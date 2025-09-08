
'''
Author: Wanzhou Lei @ Sept 2025. Email: wanzhou_lei@berkeley.edu

This script defines the loss function used in the training. 
It also defines the two used evaluation metrics: iou and accuracy. 
Some of those functions below are not used in training or evaluation. They are legacy from previous versions. But I still want to save them here. 

'''


import torch
import torch.nn as nn
import torch.nn.functional as F

##custom loss function 
#this is the binary cross entropy between prediction and reference 
def adjacency_bce_loss(pred_adj, ref_adj, weight=1.0):
    """
    pred_adj: (B, N, N) with predicted probabilities in [0,1].
    ref_adj:  (B, N, N) with 0/1 ground truth. 
              symmetrical and diagonal=0.
    Return: scalar loss.
    """
    B, N, _ = pred_adj.shape
    # mask for upper triangle, excluding diagonal
    # shape => (N,N), boolean
    triu_mask = torch.triu(torch.ones(N, N), diagonal=1).bool().to(pred_adj.device)
        
    # gather the predicted upper triangle => shape (B, #tri_entries)
    pred_upper = pred_adj[:, triu_mask]
    # gather the reference upper triangle => shape (B, #tri_entries)
    ref_upper = ref_adj[:, triu_mask].float()
        
    # use a BCE loss 
    bce = nn.BCELoss(reduction='none')
    loss = bce(pred_upper, ref_upper)# shape (B, #tri_entries)
    
    #add more weights on false negative prediction 
    false_neg_mask = (ref_upper == 1) & (pred_upper < 0.5)
    weights = torch.ones_like(loss)
    weights[false_neg_mask] = weight #weight false negatives more
    
    weighted_loss = (weights * loss).mean()

    return weighted_loss

def unconnected_node_loss(pred_adj, thres=0.5):
    B, N, _ = pred_adj.shape
    adj = pred_adj > thres
    return ((adj.sum(axis=1) < 3).sum(axis=1)/N).sum()/B

#this loss is the MSE between predicted edge numbers and reference edge number
def MSE_edge_num(pred_adj, ref_adj, thres=0.5):
    pred_edge_num = (pred_adj > thres).sum(axis=(1, 2))
    ref_edge_num = ref_adj.sum(axis=(1, 2))
    return torch.nn.MSELoss()(pred_edge_num, ref_edge_num)

def cross_edge_penalty(pred_matrix, x, thres = 0.5):
    '''
    This function computes the number of crossing edges / number of total edges in predicted graph

    Args:
        pred_matrix: tensor (B, N, N) of floats of probability of connection 
        x_coord: tensor (B, N, 2) coordinates of points in a batch
    Returns:
        loss: scalar tensor, the number of corssing edges divided by the total number of edges in each graph, averaged by batch 
    '''

    pred_matrix = (pred_matrix > 0.5)*1
    B, N, _ = pred_matrix.shape
    penalty = 0
    for k in range(B):
        adj = pred_matrix[k]
        triu_mask = torch.triu(torch.ones(N, N), diagonal=1).bool().to(adj.device)
        adj *= triu_mask

        x_coord = x[k]
        edge_start, edge_end = torch.where(adj == 1)
        edge_tuples = [[edge_start[i].item(), edge_end[i].item()] for i in range(len(edge_start))]
        num_intersections = 0
        for i in range(len(edge_tuples)-1):
            for j in range(i+1, len(edge_tuples)):
                p1_index = edge_tuples[i]
                p2_index = edge_tuples[j]
                if not bool(set(p1_index) & set(p2_index)):
                   if intersect(x_coord[p1_index[0]], x_coord[p1_index[1]], x_coord[p2_index[0]], x_coord[p2_index[1]]):
                       num_intersections += 1
        penalty += num_intersections / len(edge_tuples)
    return penalty / B
        

#this function determines if two line segments intersect
def intersect(p1, p2, p3, p4):
    '''
    This function determines if two line segements intersect
    return true if they intersect, false if they dont

    Args:
        p1, p2, p3, p4 are all tensor of shape (2,)
        p1 and p2 are coordinates of the endpoints of the first line segment
        p3 and p4 are coordinates of the endpoints of the second line segment
    '''
    A = torch.stack([p1-p2, p4-p3]).T
    try:
        gamma = torch.linalg.solve(A, p4-p2)
    except:
        return False
    return torch.all((gamma >= 0) & (gamma <= 1)).item()

def cross_edge_penalty_vectorized(pred_matrix, coords, threshold=0.5):
    """
    Compute (number_of_crossing_edges / number_of_total_edges) per graph
    and average across the batch, in a vectorized manner.

    Args:
      pred_matrix: (B, N, N) float probabilities of connection.
      coords:      (B, N, 2) float coordinates of the nodes.
      threshold:   float threshold for binarizing the adjacency.

    Returns:
      penalty: a 0D torch.Tensor (scalar), 
               the average crossing-edge ratio across the batch.
    """

    # Binarize - Not differentiable, but yields a Torch tensor
    A = (pred_matrix > threshold).float()  # shape: (B, N, N)

    # Only consider upper triangle to avoid double counting
    B, N, _ = A.shape
    upper_mask = torch.triu(torch.ones_like(A), diagonal=1)  # shape (B,N,N)
    A = A * upper_mask  # zero out lower-triangle & diagonal

    # We'll accumulate the penalty across the batch in a Torch scalar
    # (initialized to 0 on the correct device/dtype)
    device = A.device
    penalty = torch.zeros([], device=device)  # shape ()

    # For each batch, find edges (i, j) where A[b, i, j] = 1
    # A.nonzero(as_tuple=False) => shape [#edges_total_in_batch, 3],
    # columns: [batch_idx, i, j].
    edges_all = A.nonzero(as_tuple=False)

    # We’ll loop over each graph in the batch. (We can’t trivially
    # unify them in a single big intersection test, because each
    # batch might have different coords / edges, so we group by b.)
    for b in range(B):
        # 3.1) Extract edges for this batch b
        mask_b = (edges_all[:, 0] == b) 
        # shape [E_b, 2], each row = (i, j)
        edges_b = edges_all[mask_b, 1:]
        E_b = edges_b.shape[0]
        if E_b < 2:
            # Fewer than 2 edges => no intersections possible
            continue

        # Grab the coordinates for all nodes in graph b => shape (N,2)
        coord_b = coords[b]  

        # Build p1, p2 for each edge
        #    p1[e] = coords of start node
        #    p2[e] = coords of end   node
        p1 = coord_b[edges_b[:, 0]]  # shape (E_b, 2)
        p2 = coord_b[edges_b[:, 1]]  # shape (E_b, 2)

        # We want to test all edge-pairs. We'll use upper-triangular
        #    indexing among edges themselves (to avoid double-counting).
        #    This is still O(E^2), but purely in Torch.
        idx_i, idx_j = torch.triu_indices(E_b, E_b, offset=1, device=device)
        if idx_i.numel() == 0:
            # only one edge => no intersection
            continue

        # Exclude pairs that share a vertex => no need to test
        #    edges that have common endpoints
        #    edges_b[i] = (start_i, end_i)
        #    edges_b[j] = (start_j, end_j)
        edges_i = edges_b[idx_i]  # shape (num_pairs, 2)
        edges_j = edges_b[idx_j]  # shape (num_pairs, 2)

        share_vertex = (
            (edges_i[:, 0] == edges_j[:, 0]) |
            (edges_i[:, 0] == edges_j[:, 1]) |
            (edges_i[:, 1] == edges_j[:, 0]) |
            (edges_i[:, 1] == edges_j[:, 1])
        )
        valid_mask = ~share_vertex
        idx_i = idx_i[valid_mask]
        idx_j = idx_j[valid_mask]

        # If no valid edge pairs remain, skip
        if idx_i.numel() == 0:
            continue

        # Coordinates of these valid pairs
        p1_i = p1[idx_i]  # shape (num_valid_pairs, 2)
        p2_i = p2[idx_i]
        p1_j = p1[idx_j]
        p2_j = p2[idx_j]

        # Vectorized intersection check
        # cross2D(v,w) = v.x*w.y - v.y*w.x
        def cross2D(v, w):
            return v[..., 0]*w[..., 1] - v[..., 1]*w[..., 0]

        r = p2_i - p1_i  # shape (num_valid_pairs, 2)
        s = p2_j - p1_j

        # cross(r, s)
        rxs = cross2D(r, s)

        qmp = (p1_j - p1_i)  # (q - p)
        cross_qmp_r = cross2D(qmp, r)
        cross_qmp_s = cross2D(qmp, s)

        # If rxs == 0 => parallel or colinear => skip by marking them no intersection
        eps = 1e-12
        parallel = (rxs.abs() < eps)

        # Intersection parameters
        t = cross_qmp_s / (rxs + 1e-16)  # avoid /0
        u = cross_qmp_r / (rxs + 1e-16)

        # intersection if: 
        #   not parallel, 0 < t < 1, 0 < u < 1
        intersect_mask = (~parallel) & (t >= 0) & (t <= 1) & (u >= 0) & (u <= 1)
        num_intersections = intersect_mask.sum()

        # Add to the penalty: (#intersections in this graph / total_edges_in_graph)
        # Must stay as a Torch scalar, so we do "penalty += ..."
        penalty = penalty + (num_intersections.float() / E_b)

    # Average penalty over the batch 
    return penalty / B

##define the loss functions
def loss_neg_log(logits, tri):
    '''
    Compute the following batched loss function 
    L = (sum_b=1^B sum_t=1^T_b l_bt) / number of valid tokens in referecne, where
    l_bt = - log(sum_{i \in T_bt} p(index = i))

    Arguments:
        logits: predicted logits of shape (max_steps, B, N+1), where max_steps is the max steps used in the model
        tri: truth reference of shape (B, tri_max_steps, 3), where tri_max_steps is the max steps used in data
        WE ASSUME that max_steps >= tri_max_steps

    Returns:
        A scalar tensor of the losss
     
    '''

    # tri is shape (B, T, 3) with some entries = -1
    tri = tri.to(torch.int64)
    tri_clamped = tri.clone()
    # Replace -1 with 0 so that gather won't crash
    tri_clamped[tri_clamped < 0] = 0
    
    tri_max_len = tri.shape[1]
    #cut logits to shape (tri_max_len, B, N+1)
    logits = logits[:tri_max_len, :, :].permute(1, 0, 2)

    log_probs = F.log_softmax(logits, dim=-1)
    gt_log_probs = torch.gather(log_probs, dim=2, index=tri_clamped)  # (B, T, 3)
    
    #mask out the padding
    mask = (tri != -1)
    gt_log_probs *= mask
    
    return -gt_log_probs.sum() / mask.sum()

import torch

def iou_triangles(tri: torch.Tensor, indicies: torch.Tensor) -> float:
    """
    Intersection-over-Union (IoU) between predicted and reference triangles.

    Arguments:
        tri: Tensor (B, max_len, 3)   - reference sequences, float32
                padding rows are all -1
                the LAST *valid* row is [20,20,20]  (EOS token)
        indicies: Tensor (max_len2, B, 3)  - predictions, int64
                the FIRST row that contains a 20 is EOS; rows after that
                are ignored

    Returns:
    mean_iou: float   - average IoU over the B graphs
    """

    B = tri.size(0)
    device = tri.device

    #  Build a mask of valid reference rows 
    #  a row is valid if NOT padding (all != -1) AND NOT EOS (all == 20)
    tri_int = tri.long()
    not_pad = (tri_int != -1).all(dim=-1)
    not_eos = ~(tri_int == 20).all(dim=-1)
    tri_mask = not_pad & not_eos

    # Collect reference triangles as sets of sorted tuples
    # shape after sort: (B, L, 3)
    tri_sorted = tri_int.sort(dim=-1).values
    ref_sets = [
        {tuple(row.tolist()) for row in tri_sorted[b][tri_mask[b]]}
        for b in range(B)
    ]

    # Extract valid prediction rows
    max_len2 = indicies.size(0)
    pred = indicies.permute(1, 0, 2).long()

    pred_sets = []
    for b in range(B):
        rows = pred[b]

        # stop at first row that contains a 20
        eos_idx = (rows == 20).any(dim=-1).nonzero(as_tuple=False)
        if eos_idx.numel():
            rows = rows[:eos_idx[0, 0]]

        # drop padding rows (if they exist) and sort inside the triple
        rows = rows[(rows != -1).all(dim=-1)]
        rows = rows.sort(dim=-1).values

        pred_sets.append({tuple(r.tolist()) for r in rows})

    # IoU per graph
    ious = []
    for ref, pred in zip(ref_sets, pred_sets):
        if len(ref) == len(pred) == 0:
            ious.append(1.0)
            continue
        inter = len(ref & pred)
        union = len(ref | pred)
        ious.append(inter / union)

    return float(torch.tensor(ious).mean())

def iou_accuracy(tri: torch.Tensor, indicies: torch.Tensor) -> float:
    """
    accuracy of predicted triangle: (number of correct prediction)/(number of predicted triangles)

    Arguments:
        tri: Tensor (B, max_len, 3)   - reference sequences, float32
                padding rows are all -1
                the LAST *valid* row is [20,20,20]  (EOS token)
        indicies: Tensor (max_len2, B, 3)  - predictions, int64
                the FIRST row that contains a 20 is EOS; rows after that
                are ignored

    Returns:
        mean_accuracy: float   - average accuracy over the B graphs
    """

    B = tri.size(0)
    device = tri.device

    # Build a mask of valid reference rows
    # a row is valid if NOT padding (all != -1) AND NOT EOS (all == 20)
    tri_int = tri.long()
    not_pad = (tri_int != -1).all(dim=-1)
    not_eos = ~(tri_int == 20).all(dim=-1)
    tri_mask = not_pad & not_eos 

    # Collect reference triangles as sets of sorted tuples
    # shape after sort: (B, L, 3)
    tri_sorted = tri_int.sort(dim=-1).values
    ref_sets = [
        {tuple(row.tolist()) for row in tri_sorted[b][tri_mask[b]]}
        for b in range(B)
    ]

    #  Extract valid prediction rows 
    max_len2 = indicies.size(0)
    pred = indicies.permute(1, 0, 2).long()

    pred_sets = []
    for b in range(B):
        rows = pred[b]

        # stop at first row that contains a 20
        eos_idx = (rows == 20).any(dim=-1).nonzero(as_tuple=False)
        if eos_idx.numel():
            rows = rows[:eos_idx[0, 0]] # trim after EOS

        # drop padding rows (if they exist) and sort inside the triple
        rows = rows[(rows != -1).all(dim=-1)]
        rows = rows.sort(dim=-1).values

        pred_sets.append({tuple(r.tolist()) for r in rows})

    # IoU per graph
    accuracies = []
    for ref, pred in zip(ref_sets, pred_sets):
        if len(ref) == len(pred) == 0:
            accuracies.append(1.0)
            continue
        correct = len(ref & pred)
        total = len(pred)
        accuracies.append(correct / total)

    return float(torch.tensor(accuracies).mean())