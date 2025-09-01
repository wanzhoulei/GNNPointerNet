import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm

from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops
from torch.nn import Sequential as Seq, Linear as Lin, ReLU

import pickle

import torch.nn.functional as F
import torch_scatter
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GCNConv
import math

from itertools import combinations

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def build_knn_edges_batched(points_batch, k):
    """
    Vectorized batched KNN edge construction.

    Args:
        points_batch: Tensor of shape (B, N, 2)
        k: int, number of neighbors

    Returns:
        edge_index: Tensor of shape (2, total_edges)
        batch: Tensor of shape (B * N,) each element indicates which batch element each edge belongs to 
    """
    B, N, _ = points_batch.shape
    device = points_batch.device

    # (B, N, N) pairwise distances
    dists = torch.cdist(points_batch, points_batch, p=2.0)

    # Get top-k neighbors (exclude self, so take 1:k+1)
    _, knn_idx = torch.topk(dists, k=k + 1, dim=-1, largest=False)  # shape: (B, N, k+1)
    knn_idx = knn_idx[:, :, 1:]  # remove self

    # Build source and target indices for edges
    src = torch.arange(N, device=device).view(1, N, 1).repeat(B, 1, k)  # (B, N, k)
    tgt = knn_idx  # (B, N, k)

    # Batch offset
    batch_offset = (torch.arange(B, device=device) * N).view(B, 1, 1)
    src = (src + batch_offset).reshape(-1)
    tgt = (tgt + batch_offset).reshape(-1)

    # Symmetrize: add both (i, j) and (j, i)
    edge_index = torch.cat([torch.stack([src, tgt], dim=0),
                            torch.stack([tgt, src], dim=0)], dim=1)  # (2, 2*B*N*k)

    return edge_index

def make_pyg_batch_2dpoints(points_batch, k):
    """
    Args:
        points_batch: (B, N, 2)
        k: int

    Returns:
        PyG Batch object with:
            - x: shape (B*N, 2) <-- stacked 2d coordinates of all nodes
            - edge_index: shape (2, total_edge) <-- stacked edge list, first row starting node, second row ending node
            - batch: shape (B*N,), indicating which graph each node belongs to
    """
    B, N, _ = points_batch.shape
    device = points_batch.device

    # Build edge_index (2, E)
    edge_index = build_knn_edges_batched(points_batch, k)

    # Flatten all points into (B*N, 2)
    x = points_batch.reshape(B * N, 2)

    #build edge ettributes ()

    # Build batch vector
    batch = torch.arange(B, device=device).repeat_interleave(N)  # (B*N,)

    return Batch(x=x, edge_index=edge_index, batch=batch)

def make_pyg_batch_wedge(points_batch, k):
    """
    Args:
        points_batch: (B, N, 2) Tensor of 2D points
        k: int number of nearest neighbors

    Returns:
        PyG Batch object with:
            - x: (B*N, 2) node coordinates
            - edge_index: (2, E) stacked adjacency
            - edge_attr: (2*k*B*N, 4) edge embeddings
                [ midpoint_x, midpoint_y, length, orientation ]
            - batch: (B*N,) each node's subgraph ID
    """
    B, N, _ = points_batch.shape
    device = points_batch.device

    # Build edge_index
    edge_index = build_knn_edges_batched(points_batch, k)  # (2, E)

    # Flatten node coords into (B*N, 2)
    x = points_batch.view(B * N, 2)

    # Compute edge_attr
    src, tgt = edge_index
    p_src = x[src]  # (E, 2)
    p_tgt = x[tgt]  # (E, 2)

    # Midpoint
    midpoint = 0.5 * (p_src + p_tgt)         # (E, 2)
    # Length
    length = (p_tgt - p_src).norm(dim=1)     # (E,)
    # Orientation (angle)
    orientation = torch.atan2(
        (p_tgt[:, 1] - p_src[:, 1]),
        (p_tgt[:, 0] - p_src[:, 0])
    )  # (E,)

    edge_attr = torch.cat([
        midpoint, 
        length.unsqueeze(1), 
        orientation.unsqueeze(1)
    ], dim=1)  # (E, 4)

    edge_attr = edge_attr.view(2 * k * B * N, 4)

    # Build batch vector: which graph each node belongs to
    batch = torch.arange(B, device=device).repeat_interleave(N)  # (B*N,)

    # Return data with x, edge_index, edge_attr, batch
    return Batch(x=x, edge_index=edge_index, edge_attr=edge_attr, batch=batch)


class SimpleGNN(nn.Module):
    '''
    A simple GCN that only updates node embeddings.
    Incorporates batch info (even if it doesn't feed it directly to GCNConv).
    '''
    def __init__(self, in_dim=2, hidden_dim=32, num_layers=2):
        super().__init__()
        self.convs = nn.ModuleList()
        cur_dim = in_dim
        for _ in range(num_layers):
            self.convs.append(GCNConv(cur_dim, hidden_dim))
            cur_dim = hidden_dim
        self.out_dim = hidden_dim
        
    def forward(self, data_batch):
        """
        data_batch is a PyG Data (or DataBatch) with:
          - data_batch.x: (total_nodes, in_dim)  node features
          - data_batch.edge_index: (2, E)        adjacency
          - data_batch.batch: (total_nodes,)     subgraph ID for each node
        returns:
          node_emb: (total_nodes, hidden_dim)
        """
        x, edge_index, batch = data_batch.x, data_batch.edge_index, data_batch.batch

        # Even though GCNConv doesn't directly need `batch`,
        # we keep it so PyG is aware these nodes belong to multiple subgraphs.
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
        return x


class EdgeMLP(nn.Module):
    """
    For a batch of graphs, each with N nodes, produce a probability adjacency matrix A of size (B, N, N).
    - B is the batch size
    - N is the number of nodes per graph
    - node_emb: (B*N, embed_dim)
    - data_batch.batch: (B*N,) indicates subgraph membership (0..B-1)
    """

    def __init__(self, embed_dim=32, hidden_dim=64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    @staticmethod
    def pair_node_embeddings(H, B, N):
        """
        H: Tensor of shape (B*N, hidden_dim)
        Returns: Tensor of shape (B*N*(N-1)/2, 2*hidden_dim)
        """
        hidden_dim = H.shape[1]
        H = H.view(B, N, hidden_dim)  # (B, N, hidden_dim)
    
        # Create all index pairs (i, j) where i < j
        idx_pairs = list(combinations(range(N), 2))  # [(0,1), (0,2), (1,2), ...]
        idx_i = torch.tensor([i for i, j in idx_pairs])
        idx_j = torch.tensor([j for i, j in idx_pairs])
    
        # Expand to batch dimension
        idx_i = idx_i.unsqueeze(0).repeat(B, 1)  # (B, num_pairs)
        idx_j = idx_j.unsqueeze(0).repeat(B, 1)  # (B, num_pairs)
        batch_indices = torch.arange(B).unsqueeze(1).repeat(1, len(idx_pairs))  # (B, num_pairs)
    
        # Gather embeddings
        h_i = H[batch_indices, idx_i]  # (B, num_pairs, hidden_dim)
        h_j = H[batch_indices, idx_j]  # (B, num_pairs, hidden_dim)
    
        # Concatenate
        h_cat = torch.cat([h_i, h_j], dim=-1)  # (B, num_pairs, 2*hidden_dim)
        h_cat = h_cat.view(-1, 2 * hidden_dim)  # (B*num_pairs, 2*hidden_dim)
        return h_cat

    @staticmethod
    def build_batched_adj_matrix(batched_prob, B, N):
        """
        batched_prob: Tensor of shape (B * num_pairs, 1)
        Returns: Tensor of shape (B, N, N) - symmetric adjacency matrices with 0 diagonals
        """
        device = batched_prob.device
        batched_prob = batched_prob.view(B, -1)  # (B, num_pairs)
        
        # Get upper triangle indices (i < j)
        idx_pairs = list(combinations(range(N), 2))  # [(0,1), (0,2), (1,2), ...]
        idx_i = torch.tensor([i for i, j in idx_pairs], device=device)
        idx_j = torch.tensor([j for i, j in idx_pairs], device=device)
        
        # Initialize empty adjacency matrix
        adj = torch.zeros((B, N, N), device=device)
    
        # Fill the upper triangle
        adj[:, idx_i, idx_j] = batched_prob
    
        # Symmetrize (copy upper to lower)
        adj = adj + adj.transpose(1, 2)
    
        return adj
    
    def forward(self, node_emb, B, N):
        """
        Args:
            node_emb: (B*N, embed_dim)  stacked embeddings of all graphs in the batch
            batch:    (B*N,)           subgraph index for each node (range 0..B-1)
        
        Returns:
            A: (B, N, N) adjacency probability matrix for each graph.
               - A[b] is the NxN adjacency matrix for graph b.
               - A[b, i, j] = P(edge between node i, j in graph b)
               - diagonal set to 0
               - matrix is symmetrized
        """
        H_paired = self.pair_node_embeddings(node_emb, B, N) #shape (B*N*(N-1)/2, 2*hidden_dim)
        #go through the MLP
        batched_prob = self.mlp(H_paired) #shape (B*N*(N-1)/2, 1)
        batched_adj = self.build_batched_adj_matrix(batched_prob, B, N) 
        return batched_adj
        
#Combine everything
class GraphEdgePredictor(nn.Module):
    def __init__(self, k=5, gnn_hidden=32, gnn_layers=2):
        super().__init__()
        self.k = k
        self.gnn = SimpleGNN(in_dim=2, hidden_dim=gnn_hidden, num_layers=gnn_layers)
        self.edge_mlp = EdgeMLP(embed_dim=gnn_hidden, hidden_dim=64)

    def forward(self, batch_points):
        """
        batch_points: shape (B, N, 2)
          1) create a list of PyG Data for each sample
          2) batch them into one big Data
          3) do GNN => node embeddings
          4) for each subgraph, compute adjacency probabilities 
             for all i<j
          => return adjacency (B, N, N)
        """
        B, N, _ = batch_points.shape
        
        #build PYG data
        data_batch = make_pyg_batch_2dpoints(batch_points, self.k)
        #go through the convGNN to obtain node embedding
        H = self.gnn(data_batch) #shape (B*N, hidden_dim)
        #go through the MLP to obtain the adj matrix
        A = self.edge_mlp(H, B, N)
        return A

##structure for the MPGNN Pointer network
class EdgeNetwork(nn.Module):
    '''
    The MLP that takes starting vertex and ending vertex and the edge embedding
    and compute the message Phi(x_i, x_j, e_ij)
    '''
    def __init__(self, input_dim, hidden_dim, message_dim):
        '''
        Args:
            input_dim: the dimension of input: len(x_i) + len(x_j) + len(e_ij)
            hidden_dim: hidden dimension
            messgage_dim: the dimension of the message vector 
        '''
        super().__init__()
        self.edge_mlp = Seq(
            Lin(input_dim, hidden_dim),
            ReLU(),
            Lin(hidden_dim, message_dim)
        )

    def forward(self, edge_attr, x_i, x_j):
        out = torch.cat([x_i, x_j, edge_attr], dim=1)
        return self.edge_mlp(out)
    

class MPNNWithEdgeUpdate(MessagePassing):
    '''
    One layer of message passing: gamma (x_i, agg(Phi(x_i, x_j, e_ij)))
    '''
    def __init__(self, node_dim, edge_dim, hidden_dim, message_dim, new_node_dim, new_edge_dim, initial=False):
        super().__init__(aggr='add')  # sum aggregation
        
        #the neural network Phi to create the message from xi, xj and eij
        self.message_mlp = EdgeNetwork(new_edge_dim + 2 * node_dim, hidden_dim, message_dim)


        #the network to update node embedding 
        self.node_update = Seq(
            Lin(node_dim + message_dim, hidden_dim),
            ReLU(),
            Lin(hidden_dim, new_node_dim)
        )

        #the network to update edge embedding
        self.edge_network = EdgeNetwork(edge_dim + 2 * node_dim, hidden_dim, new_edge_dim)

    def forward(self, batched_data):
        '''
        forward function of one layer of message passing 

        Args:
            batched_data: pyg batched data type must contains the following attributes
                x: [B*N, node_dim] the embeddings of all the vertices in one batch
                edge_attr: [num_edges, edge_dim] the embeddings of all edges in one batch
                edge_index: [2, B*N] indices of connected edges in the batched graph

        Returns:
            node_embeddings: [B*N, new_node_embedding_dim] embeddings of all nodes in new embedding dimension
            edge_embeddings: [num_edges, new_edge_embedding_dim] embeddings of all edges in new edge embedding dimension

        '''
        # x: [num_nodes, node_dim]
        # edge_attr: [num_edges, edge_dim]
        x = batched_data.x; edge_index =batched_data.edge_index; edge_attr = batched_data.edge_attr
        
        return self.propagate(edge_index, x=x, edge_attr=edge_attr), self.updated_edge_attr

    def message(self, x_i, x_j, edge_attr):
        # Compute edge-updated features first 
        updated_edge_attr = self.edge_network(edge_attr, x_i, x_j)
        self.updated_edge_attr = updated_edge_attr
        #msg_input = torch.cat([x_i, x_j, updated_edge_attr], dim=1)
        return self.message_mlp(updated_edge_attr, x_i, x_j)

    def update(self, aggr_out, x):
        node_input = torch.cat([x, aggr_out], dim=1)
        return self.node_update(node_input)

class MP_GNN(nn.Module):
    '''
    This is the message passing graph neural network that contains several message passing layers
    '''
    def __init__(self, initial_node_dim, initial_edge_dim, embedding_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        self.initial_layer = MPNNWithEdgeUpdate(initial_node_dim, initial_edge_dim, 
                                                embedding_dim, embedding_dim, embedding_dim, embedding_dim)

        self.layers_list = nn.ModuleList()
        for _ in range(num_layers-1):
            self.layers_list.append(MPNNWithEdgeUpdate(embedding_dim, embedding_dim, 
                                                embedding_dim, embedding_dim, embedding_dim, embedding_dim))

    def forward(self, batched_data):
        ##add the self adge first
        edge_index, edge_attr = add_self_loops(batched_data.edge_index, edge_attr=batched_data.edge_attr, fill_value=0.0, 
                                               num_nodes=batched_data.x.size(0))
        batched_data.edge_index = edge_index; batched_data.edge_attr = edge_attr
        
        #goes through the first layer
        x, edge_attr = self.initial_layer(batched_data)
        batched_data.x = x; batched_data.edge_attr = edge_attr
        #goes through all the remianin layers
        for layer in self.layers_list:
            x, edge_attr = layer(batched_data)
            batched_data.x = x; batched_data.edge_attr = edge_attr
        return x, edge_attr


class EncoderLSTM(nn.Module):
    """
    LSTM Encoder:
      - input:  (batch_size, seq_len, input_dim)
      - output: (batch_size, seq_len, hidden_dim) + (h, c) final states
    """
    def __init__(self, input_dim, hidden_dim, num_layers=1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False
        )
        
    def forward(self, x):
        """
        Args:
          x: (batch_size, seq_len, input_dim)
        Returns:
          encoder_outputs: (batch_size, seq_len, hidden_dim)
          (h, c): final hidden, cell states
        """
        encoder_outputs, (h, c) = self.lstm(x)  # h,c shape: (num_layers, batch, hidden_dim)
        return encoder_outputs, (h, c)

class DecoderLSTM(nn.Module):
    """
    LSTM Decoder:
      - We run for multiple steps (each step can produce one triangle).
      - Input size = 3 * hidden_dim (since we feed a concatenation of 3 node embeddings).
      - On each step, we produce a pointer distribution over (N+1) "nodes":
        the real N nodes + 1 "end token" node.
    """
    def __init__(self, hidden_dim, num_layers=1):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # LSTMCell expects input_size = 3*hidden_dim (concatenation of 3 node embeddings)
        self.input_size = 3 * hidden_dim
        self.lstm_cell = nn.LSTMCell(self.input_size, hidden_dim)
        
        # A transform for the pointer "query" if desired
        self.query_transform = nn.Linear(hidden_dim, hidden_dim)

    def forward(self,
                encoder_outputs,
                hidden,
                cell,
                end_node_embed,
                initial_input,
                max_steps=10, 
                teacher_indices=None):
        """
        Args:
          encoder_outputs: (batch_size, N, hidden_dim)
            - includes real node embeddings for the first N-1 positions
            - the last position is the "end token" embedding
          hidden, cell: initial hidden/cell states for the decoder
          end_node_embed: (1, hidden_dim) - if you need it separately for something
          initial_input: (batch_size, 3*hidden_dim) start token embedding
          max_steps: the maximum number of triangles to decode
          teacher_indices: shape (batch_size, max_steps, 3), or None
              if None, Prediction mode
              if not None, train mode, where the input will be truth triangles
          
        Returns:
          pointer_logits_list: tensor of shape (max_steps, B, N+1)
          chosen_indices_list: tensor of shape (max_steps, B, 3)
        """
        batch_size, N, hd = encoder_outputs.shape
        
        pointer_logits_list = []
        chosen_indices_list = []

        # We'll use the given initial_input for step 0
        # for later step, it will be either from teacher enforcing or from the model's own prediction 
        input_t = initial_input  # shape (batch_size, 3*hidden_dim)

        #pad the teacher_indices if len < max_steps
        if teacher_indices is not None and teacher_indices.shape[1] < max_steps:
            teacher_indices = F.pad(teacher_indices, (0, 0, 0, max_steps-teacher_indices.shape[1]), value=-1).to(torch.int)

        for t in range(max_steps):
            # LSTMCell step
            hidden, cell = self.lstm_cell(input_t, (hidden, cell))

            # Pointer distribution
            query = self.query_transform(hidden)   # (batch_size, hidden_dim)
            query_expanded = query.unsqueeze(1).expand(-1, N, -1)  # (batch_size, N, hidden_dim)
            pointer_logits = torch.sum(query_expanded * encoder_outputs, dim=-1)  # (batch_size, N)
            pointer_logits_list.append(pointer_logits)

            _, top3_indices = torch.topk(pointer_logits, k=3, dim=1)  # (batch_size, 3)

            # Pick top-3 from pointer logits or from teacher enforcing solution 
            if teacher_indices is None:
                next_indices = top3_indices
            else:
                ##teacher forcing mode, use ground truth 
                next_indices = teacher_indices[:, t, :].to(torch.int64)  # shape (batch_size, 3)
                next_indices[next_indices == -1] = 0 # replace -1 with 0 because gather does not accept -1
                

            chosen_indices_list.append(top3_indices)

            # Build the next input by concatenating the chosen embeddings
            top3_vectors = torch.gather(
                encoder_outputs,  # (batch_size, N, hidden_dim)
                dim=1,
                index=next_indices.unsqueeze(-1).expand(-1, -1, hd)
            )
            # shape of top3_vectors: (batch_size, 3, hidden_dim)
            input_t = top3_vectors.view(batch_size, 3 * hd)  # (batch_size, 3*hidden_dim)

        return torch.stack(pointer_logits_list), torch.stack(chosen_indices_list)

class DecoderLSTM_att(nn.Module):
    """
    Decoder with causal self-attention.
    • identical public API to DecoderLSTM
    • after each LSTMCell step, the current hidden state attends over
      all previous decoder hidden states (including itself)
    """
    def __init__(self, hidden_dim, num_layers=1, num_heads=4, dropout=0.0, num_cold_start=0):
        """
        Args
        ----
        hidden_dim : int   - dimension of LSTM hidden state
        num_layers : int   - # layers in LSTMCell stack  (kept for parity)
        num_heads  : int   - heads in MultiheadAttention
        dropout    : float - dropout inside attention
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.input_size = 3 * hidden_dim

        self.lstm_cell = nn.LSTMCell(self.input_size, hidden_dim)
        self.num_cold_start = num_cold_start

        # multi‑head self‑attention; batch_first=True gives shape (B, S, D)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        self.query_transform = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, encoder_outputs, hidden, cell,
                end_node_embed, initial_input,
                max_steps=10, teacher_indices=None):

        batch_size, N, hd = encoder_outputs.shape
        pointer_logits_list, chosen_indices_list = [], []
        input_t = initial_input
        saved_hiddens = []                         # ← list of *copies*

        if teacher_indices is not None and teacher_indices.shape[1] < max_steps:
            teacher_indices = F.pad(teacher_indices,
                                    pad=(0, 0, 0, max_steps - teacher_indices.shape[1]),
                                    value=-1).to(torch.int64)
                                    

        for _ in range(self.num_cold_start):
            # LSTM step
            hidden, cell = self.lstm_cell(input_t, (hidden, cell))

            # causal self‑attention over past dec‑states
            saved_hiddens.append(hidden.clone())
            kv = torch.stack(saved_hiddens, dim=1)       # (B, s, hd)
            q  = hidden.unsqueeze(1)                    # (B, 1, hd)
            attn_out, _ = self.self_attn(q, kv, kv, need_weights=False)
            hidden_ctx = 0.5 * (hidden + attn_out.squeeze(1))

            # pointer distribution
            query = self.query_transform(hidden_ctx)
            pointer_logits = torch.sum(
                query.unsqueeze(1) * encoder_outputs, dim=-1
            )                                           # (B, N)
            #pointer_logits_list.append(pointer_logits)

            # choose next indices (top‑3, sorted) and build next input
            _, pred_idx = torch.topk(pointer_logits, k=3, dim=1)
            pred_idx, _ = torch.sort(pred_idx, dim=1)
            #chosen_indices_list.append(pred_idx)

            top3_vec = torch.gather(
                encoder_outputs,
                1,
                pred_idx.unsqueeze(-1).expand(-1, -1, hd)
            )                                           # (B, 3, hd)
            input_t = top3_vec.reshape(batch_size, 3 * hd)



        for t in range(max_steps):
            # 1) LSTM step
            hidden, cell = self.lstm_cell(input_t, (hidden, cell))   # (B, hd)

            # 2) save a copy for causal self‑attention
            saved_hiddens.append(hidden.clone())                     #   changed

            kv = torch.stack(saved_hiddens, dim=1)                   # (B, t+1, hd)
            q  = hidden.unsqueeze(1)                                 # (B, 1,  hd)
            attn_out, _ = self.self_attn(q, kv, kv, need_weights=False)
            attn_out = attn_out.squeeze(1)
            hidden_ctx = 0.5 * (hidden + attn_out)                   # residual

            # 3) pointer distribution
            query = self.query_transform(hidden_ctx)
            pointer_logits = torch.sum(
                query.unsqueeze(1) * encoder_outputs, dim=-1)
            pointer_logits_list.append(pointer_logits)

            _, top3_pred = torch.topk(pointer_logits, k=3, dim=1)
            #sort the indices
            top3_pred, _ = torch.sort(top3_pred, dim=1)

            # 4) teacher forcing
            if teacher_indices is None:
                next_indices = top3_pred
            else:
                next_indices = teacher_indices[:, t, :].to(torch.int64)
                next_indices = torch.where(next_indices == -1,
                                           torch.zeros_like(next_indices),
                                           next_indices)

            chosen_indices_list.append(top3_pred)

            # 5) build next input
            top3_vec = torch.gather(
                encoder_outputs,
                1,
                next_indices.unsqueeze(-1).expand(-1, -1, hd)
            )                                       # (B, 3, hd)
            input_t = top3_vec.reshape(batch_size, 3 * hd)

        return (torch.stack(pointer_logits_list),
                torch.stack(chosen_indices_list))

class PointerNetForTriangles(nn.Module):
    """
    Full Pointer Network that:
      - Has an LSTM encoder
      - Adds a "virtual end-node embedding" to the encoder outputs
      - Has a learnable "start token" embedding for the decoder's first input
      - Decodes up to max_steps, each step producing a triple of node indices
    """
    def __init__(self, input_dim, hidden_dim, num_layers=1, max_steps=10, attention=False, num_cold_start=0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_steps = max_steps
        
        # Encoder
        self.encoder = EncoderLSTM(input_dim, hidden_dim, num_layers)
        
        # A learnable parameter to serve as "end node"
        self.end_node_embed = nn.Parameter(torch.randn(1, hidden_dim))
        
        # A learnable "start token" that the decoder sees at time t=0
        # Because we feed 3*hidden_dim each step, define it that shape:
        self.start_token_embed = nn.Parameter(torch.randn(1, 3 * hidden_dim))
        
        # Decoder
        if not attention:
            self.decoder = DecoderLSTM(hidden_dim, num_layers)
        else:
            self.decoder = DecoderLSTM_att(hidden_dim, num_layers, num_cold_start=num_cold_start)

    def forward(self, x, teacher_indices=None, max_steps=40):
        """
        x: (batch_size, seq_len, input_dim) -- real node inputs
        Returns:
          pointer_logits_list: list of (batch_size, N)
          chosen_indices_list: list of (batch_size, 3)
        """
        batch_size, seq_len, _ = x.shape
        
        # 1) Encode
        encoder_outputs, (h_enc, c_enc) = self.encoder(x)
        # h_enc, c_enc: shape (num_layers, batch, hidden_dim)
        
        # We'll only take the top layer hidden/cell if num_layers > 1
        h_enc = h_enc[-1]  # (batch, hidden_dim)
        c_enc = c_enc[-1]  # (batch, hidden_dim)
        
        # 2) Append "end node" to encoder_outputs
        end_node_tiled = self.end_node_embed.unsqueeze(0).expand(batch_size, -1, -1)
        encoder_outputs = torch.cat([encoder_outputs, end_node_tiled], dim=1)
        # Now encoder_outputs has shape (batch, N, hidden_dim)
        # N = seq_len + 1  (the extra one is the end node)
        
        # 3) Prepare the decoder's initial input from the start token
        start_tiled = self.start_token_embed.expand(batch_size, -1)  # (batch_size, 3*hidden_dim)
        
        # 4) Run the decoder
        pointer_logits_list, chosen_indices_list = self.decoder(
            encoder_outputs,
            h_enc,
            c_enc,
            end_node_embed=self.end_node_embed,
            initial_input=start_tiled,
            max_steps=max(self.max_steps, max_steps), 
            teacher_indices = teacher_indices
        )
        
        return pointer_logits_list, chosen_indices_list

class GraphPointerNet(nn.Module):
    '''
    This is the funal node that consists of 1. Message Passing GNN 2. A LSTM as Pointer Network
    '''
    
    def __init__(self, initial_node_dim, initial_edge_dim, embedding_dim = 32, num_layers=4, num_layers_LSTM = 1,
                 hidden_dim=64, max_steps=40, attention=False, num_cold_start=0):
        '''
        Arguments:
            initial_node_dim: the initial node dim before anything, usually 2, the 2d coordinate
            initial_edge_dim: the initial edge dimension before anything, usually 4
            embedding_dim: the embedding dimension for both the node and edge in MSP GNN
            num_layers: number of layers in the message passing GNN
            hidden_dim: the hidden dimension in the LSTM, in encoder hidden state has len hidden_dim, in decoder, 
                the hidden state has shape 3*hidden_dim and takes input of shape 3*hidden_dim
            max_steps: number of steps made in the decoder, in this implementation, the model will always make
                max_steps predictions even if the end of sequence token is predicted

        '''
        super().__init__()
        self.max_steps = max_steps
        self.MP_model = MP_GNN(initial_node_dim, initial_edge_dim, embedding_dim, num_layers)
        self.Pointer_model = PointerNetForTriangles(embedding_dim, hidden_dim, max_steps=max_steps, 
                attention=attention, num_layers=num_layers_LSTM, num_cold_start=num_cold_start)
        self.embedding_dim = embedding_dim

    def forward(self, data_batch, teacher_indices=None, N=20, max_steps=40):
        '''
        Arguments:
            data_batch: a pyg data set that contains the following attributes:
                x: [B*N_nodes, initial_dim], the initial embeddings of all nodes in one batch
                edge_index: [2, (k * N_nodes * B)*2], indice pairs of directed edges in the batched graph
                edge_attr: [(k * N_nodes * B)*2, initial_edge_dim], initial edge embeddings of all the edges in the batched graph 

        '''
        
        node_embeddings, edge_embeddings = self.MP_model(data_batch)
        #reshape the node embeddings into shape (B, N, embedding_dim)
        node_embeddings = node_embeddings.reshape(-1, N, self.embedding_dim)
        logits, indices = self.Pointer_model(node_embeddings, teacher_indices=teacher_indices, max_steps=max(max_steps, self.max_steps))

        return logits, indices




