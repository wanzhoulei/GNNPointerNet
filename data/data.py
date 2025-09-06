import numpy as np
from scipy.spatial import Delaunay
from scipy.spatial import ConvexHull

import matplotlib.pyplot as plt
import torch

import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm

from collections import deque
from typing import Sequence

#this function creates and retunrs an adjacency matrix by taking the triangle list
def get_adj_mtx(tri, N):
    adj = np.zeros((N, N))
    for triangle in tri:
        adj[triangle[0], triangle[1]] = 1
        adj[triangle[1], triangle[0]] = 1
        adj[triangle[0], triangle[2]] = 1
        adj[triangle[2], triangle[0]] = 1
        adj[triangle[1], triangle[2]] = 1
        adj[triangle[2], triangle[1]] = 1
    return adj

def sample_delauney(N=20, randomize_start=False):
    points = np.random.rand(N, 2)
    tri = Delaunay(points).simplices
    adj = get_adj_mtx(tri, N)

    DFSorder = dfs_order(adj, 0)
    
    #reorder the datas
    points = points[DFSorder]
    adj = adj[np.ix_(DFSorder, DFSorder)]
    
    N = len(DFSorder)
    inverse_map = np.empty(N, dtype=int)
    
    for new_idx, old_idx in enumerate(DFSorder):
        inverse_map[old_idx] = new_idx
    tri = inverse_map[tri]
    
    #create the dual graph
    tri_adj = np.zeros((tri.shape[0], tri.shape[0]))
    for i in range(len(tri)):
        for j in range(i, len(tri)):
            if len(np.intersect1d(tri[i], tri[j])) > 1:
                tri_adj[i][j] = 1
                tri_adj[j][i] = 1

    # Compute the convex hull
    hull = ConvexHull(points)
    # Indices of the points forming the convex hull (in counter-clockwise order)
    hull_indices = hull.vertices

    boundary_tri = find_boundary_triangles(tri, hull_indices)

    if not randomize_start:
        #dfs traversal of the dual graph
        tri_dfs_order = dfs_order(tri_adj, boundary_tri[0])
        #reorder the points and triangles
        tri = tri[tri_dfs_order]
        ##add the stop sign to tri
        tri = np.vstack((tri, [N, N, N]))  
        return points, adj, tri
    else:
        tri_list = []
        for i in range(len(boundary_tri)):
            #dfs traversal of the dual graph
            tri_dfs_order = dfs_order(tri_adj, boundary_tri[i])
            #reorder the points and triangles
            tri_i = tri[tri_dfs_order]
            ##add the stop sign to tri
            tri_i = np.vstack((tri_i, [N, N, N])) 
            tri_list.append(tri_i)
        return points, adj, tri_list

def dfs_order(adj_matrix, start_index) -> list:
    """
    Perform a Depth-First Search (DFS) over the graph (with adjacency matrix adj_matrix),
    returning the order in which nodes are first visited.

    Args:
        adj_matrix: np.ndarray of shape (N, N), adjacency matrix of the graph.
                    - adj_matrix[u, v] != 0 means an edge from u to v.
                    - No self-connections => diag(adj_matrix) = 0.
        start_indices: the index of the starting triangle

    Returns:
        A list of node indices in the order they are visited by DFS.
        If the graph is disconnected, the DFS restarts from the next unvisited node
        until all nodes are covered.
    """
    N = adj_matrix.shape[0]
    visited = [False] * N
    dfs_visit_order = []

    def dfs_iter(start_node: int):
        stack = [start_node]
        while stack:
            node = stack.pop()
            if not visited[node]:
                visited[node] = True
                dfs_visit_order.append(node)
                # Push neighbors in reverse order so that lower-index neighbors
                # are popped/visited first (this is optional, just for deterministic ordering).
                neighbors = []
                for neighbor in range(N):
                    if adj_matrix[node, neighbor] != 0 and not visited[neighbor]:
                        neighbors.append(neighbor)
                # Reverse the neighbor list so that we visit them in ascending order
                for n in reversed(neighbors):
                    stack.append(n)

    # Run DFS from each unvisited node (this covers all components if graph is disconnected)
    dfs_iter(start_index)

    return dfs_visit_order

def bfs_order(adj_matrix, start_index):
    """
    Perform a Breadth‑First Search (BFS) over a graph given its adjacency matrix,
    returning the order in which nodes are first visited.

    Parameters
    ----------
    adj_matrix : np.ndarray, shape (N, N)
        Adjacency matrix of the graph.  Non‑zero entry adj_matrix[u, v] means an
        edge from node u to node v.  Diagonal is assumed to be zero
        (no self‑loops).
    start_index : int
        Index of the node from which BFS starts.

    Returns
    -------
    visit_order : List[int]
        A list of node indices in the order they are *first* visited by BFS.
        (If you need a complete traversal of *all* components, call this
        function separately for each disconnected component.)
    """
    if adj_matrix.ndim != 2 or adj_matrix.shape[0] != adj_matrix.shape[1]:
        raise ValueError("`adj_matrix` must be a square 2‑D array.")
    if not (0 <= start_index < adj_matrix.shape[0]):
        raise IndexError("`start_index` out of range.")

    N = adj_matrix.shape[0]
    visited = [False] * N
    visit_order: List[int] = []

    queue = deque([start_index])
    visited[start_index] = True

    while queue:
        node = queue.popleft()
        visit_order.append(node)

        # Collect unvisited neighbours, sorted for deterministic order
        neighbours = [
            nbr for nbr in range(N)
            if adj_matrix[node, nbr] != 0 and not visited[nbr]
        ]

        for nbr in neighbours:
            visited[nbr] = True
            queue.append(nbr)

    return visit_order

def find_boundary_triangles(tri, hull_indices):
    n = len(hull_indices)
    boundary_sides = [(hull_indices[i], hull_indices[(i+1)%n]) for i in range(n)]

    results = []

    for side in boundary_sides:
        p1, p2 = side
        # Find triangles containing both p1 and p2
        mask = np.array([
            (p1 in triangle) and (p2 in triangle)
            for triangle in tri
        ])
        matched_tri_indices = np.where(mask)[0][0]
        results.append(matched_tri_indices)

    return results

def build_dataset(num_samples=10000, N=20, seed=42, padding_len=35, padding_val=-1, order=False):
    """
    Build a dataset of (points, adjacency) pairs for GNN training.
    
    Parameters
    ----------
    num_samples : int
        How many samples you want in your dataset.
    N : int
        Number of points per sample.
    seed : int
        Random seed for reproducibility.
    order: bool
        whether to order the indices in the triangles
        
    Returns
    -------
    X : np.ndarray
        Shape (num_samples, N, 2), each row is the set of 2D points.
    Y : np.ndarray
        Shape (num_samples, N, N), each corresponding adjacency matrix
    TRI: np.ndarray
        Shape (num_samples, padding_len, 3)
    """
    # Optional: Set the random seed for reproducibility
    np.random.seed(seed)
    
    # Preallocate arrays
    X = np.zeros((num_samples, N, 2), dtype=np.float32)
    Y = np.zeros((num_samples, N, N), dtype=np.float32)
    TRI = np.zeros((num_samples, padding_len, 3), dtype=np.int32)
    
    for i in range(num_samples):
        points, adj, tri = sample_delauney(N=N)
        if order:
            tri = np.sort(tri, axis=1)
        
        X[i] = points
        Y[i] = adj
        
        N_tri = tri.shape[0]
        if N_tri > padding_len:
            raise ValueError(f"Input has {N_tri} rows, which exceeds the target length {padding_len}.")
            
        pad_rows = padding_len - N_tri
        padding = np.full((pad_rows, 3), padding_val, dtype=tri.dtype)
        tri = np.vstack([tri, padding])
        TRI[i] = tri
            
    
    return X, Y, TRI

def build_dataset_randomized(num_samples=10000, N=20, seed=42, padding_len=35, padding_val=-1, order=False):
    '''
    Basically the same as before, but the data set will be larger
    the traversal order of triangles will be different, with randomized starting boundary triangles

    Parameters
    ----------
    num_samples : int
        How many samples you want in your dataset.
    N : int
        Number of points per sample.
    seed : int
        Random seed for reproducibility.
    order: bool
        whether to order the indices in the triangles
        
    Returns
    -------
    X : np.ndarray
        Shape (unknown_num_samples, N, 2), each row is the set of 2D points.
    Y : np.ndarray
        Shape (unknown_num_samples, N, N), each corresponding adjacency matrix.
    TRI: np.ndarray
        Shape (unknwon_num_samples, padding_len, 3)
    '''
    
    # Optional: Set the random seed for reproducibility
    np.random.seed(seed)
    
    #container
    X = []
    Y = []
    TRI = []
    
    for i in tqdm(range(num_samples)):
        points, adj, tri_list = sample_delauney(N=N, randomize_start=True)
        tri_list = np.array(tri_list)
        if order:
            tri_list = np.sort(tri_list, axis=2)

        for k in range(tri_list.shape[0]):
            X.append(points)
            Y.append(points)
        
            N_tri = tri_list[k].shape[0]
            if N_tri > padding_len:
                raise ValueError(f"Input has {N_tri} rows, which exceeds the target length {padding_len}.")
            
            pad_rows = padding_len - N_tri
            padding = np.full((pad_rows, 3), padding_val, dtype=tri_list.dtype)
            tri = np.vstack([tri_list[k], padding])
            TRI.append(tri)
    return np.array(X), np.array(Y), np.array(TRI)

#define custom dataset and data loader
class GraphDataSet(Dataset):
    def __init__(self, X_data, Y_data, tri_data=None):
        self.X_data = X_data
        self.Y_data = Y_data
        if tri_data != None:
            self.tri_data = tri_data

    def __len__(self):
        return self.X_data.shape[0]

    def __getitem__(self, index):
        if self.tri_data != None:
            return (self.X_data[index], self.Y_data[index], self.tri_data[index])
        else:
            return (self.X_data[index], self.Y_data[index])

class GraphDataSet_new(Dataset):
    def __init__(self, X_data, tri_data):
        self.X_data = X_data
        self.tri_data = tri_data

    def __len__(self):
        return self.X_data.shape[0]

    def __getitem__(self, index):
        
        return (self.X_data[index], self.tri_data[index])
    
def build_knn_graph(points, k = 5):
    """
    Build a k‑nearest‑neighbour (k‑NN) graph for 2‑D points and return its adjacency matrix.

    Parameters
    ----------
        points : np.ndarray, shape (N, 2)
            Array of N points in the plane, one per row.
        k      : int, default=5
            Number of nearest neighbours each node is connected to
            (excluding the point itself).

    Returns
    -------
    adj : np.ndarray, shape (N, N)
        Binary (0/1) adjacency matrix.  adj[i, j] == 1 means point *i*
        is connected to point *j*.  The matrix is symmetric.
    """
    if k < 1:
        raise ValueError("k must be a positive integer.")
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("`points` must have shape (N, 2).")

    N = points.shape[0]
    
    diff = points[:, None, :] - points[None, :, :]
    dist2 = np.einsum("ijk,ijk->ij", diff, diff)    #

    np.fill_diagonal(dist2, np.inf)

    knn_idx = np.argpartition(dist2, kth=k, axis=1)[:, :k]   # shape (N, k)

    adj = np.zeros((N, N), dtype=np.uint8)
    rows = np.repeat(np.arange(N), k)
    cols = knn_idx.reshape(-1)
    adj[rows, cols] = 1
    adj |= adj.T  

    return adj

def plot_knn_graph(points, adj, index=None):
    """
    Visualise a k‑NN graph with Matplotlib.

    Parameters
    ----------
    points : np.ndarray, shape (N, 2)
        2‑D coordinates of N nodes.
    adj    : np.ndarray, shape (N, N)
        Binary adjacency matrix returned by `build_knn_graph`.
        Non‑zero entry adj[i, j] means an edge between i and j.
    """
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("`points` must have shape (N, 2).")
    if adj.shape[0] != adj.shape[1] or adj.shape[0] != points.shape[0]:
        raise ValueError("`adj` must be a square matrix with the same number of rows as `points`.")

    N = points.shape[0]

    fig, ax = plt.subplots()
    ax.scatter(points[:, 0], points[:, 1], s=40, color="tab:blue", zorder=3)

    # draw each edge once (i < j to avoid duplicates)
    for i in range(N):
        for j in range(i + 1, N):
            if adj[i, j]:
                ax.plot(
                    [points[i, 0], points[j, 0]],
                    [points[i, 1], points[j, 1]],
                    color="lightgray",
                    linewidth=1,
                    zorder=1,
                )

    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("k‑NN graph")

    if index is not None:
        ax.scatter(points[:, 0][:index], points[:, 1][:index], color='red', zorder=10)

def is_connected(points, adj_matrix) -> bool:
    """
    Check in O(N + E) time whether an undirected graph is connected.

    Parameters
    ----------
    points : np.ndarray, shape (N, 2)
        Coordinates of N 2‑D points (only used to sanity‑check N).
    adj_matrix : np.ndarray, shape (N, N)
        Adjacency matrix (non‑zero ⇒ edge).  Assumed symmetric
        and with zero diagonal.

    Returns
    -------
    connected : bool
        True  → every node is reachable from every other node  
        False → the graph has ≥ 2 connected components
    """
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("`points` must have shape (N, 2).")
    if adj_matrix.shape[0] != adj_matrix.shape[1]:
        raise ValueError("`adj_matrix` must be square.")
    if adj_matrix.shape[0] != points.shape[0]:
        raise ValueError("`points` and `adj_matrix` disagree on N.")

    N = adj_matrix.shape[0]
    if N <= 1:
        return True          # empty / single‑node graph is trivially connected

    visited = np.zeros(N, dtype=bool)
    queue = deque([0])
    visited[0] = True

    while queue:
        u = queue.popleft()
        # fast NumPy lookup of neighbours
        nbrs = np.nonzero(adj_matrix[u])[0]
        unvisited = nbrs[~visited[nbrs]]
        visited[unvisited] = True
        queue.extend(unvisited.tolist())

        if visited.all():    # early exit: all nodes reached
            return True

    return False             # some nodes never reached
    

def sample_delauney_new(N=20, order=bfs_order, k=5, train=True):
    
    ##generate N random nodes in 2d, arbitrary orders and make sure the knn graph is connected
    points = np.random.rand(N, 2)
    #create the KNN graph and find its adjacency matrix
    adj = build_knn_graph(points, k=k)

    while not is_connected(points, adj):
        print("Encountered Disconnected KNN graph for k = ", k)
        points = np.random.rand(N, 2)
        #create the KNN graph and find its adjacency matrix
        adj = build_knn_graph(points, k=k)
    
    #find the convex hull of these points
    hull = ConvexHull(points).vertices
    
    #build the dual graph of triangles using the old index
    tri = Delaunay(points).simplices
    delaunay_adj = get_adj_mtx(tri, N)
    
    #create the dual graph
    tri_adj = np.zeros((tri.shape[0], tri.shape[0]))
    for i in range(len(tri)):
        for j in range(i, len(tri)):
            if len(np.intersect1d(tri[i], tri[j])) > 1:
                tri_adj[i][j] = 1
                tri_adj[j][i] = 1
    
    #starting from each hull point do bfs and reorder the points, then delaunay on the reordered points 
    order_func = order
    chosen = []
    points_list = []
    tri_list = []
    for i in hull:
        #order of indices
        order = order_func(adj, i)
        
        #order these delaunay triangles 
        #first of all find a starting point for those triangles 
        nonzero_hull_values = [v for v in hull if v != i]
        
        rows = np.where((tri == i).any(axis=1) & np.isin(tri, nonzero_hull_values).any(axis=1))[0] #indices of rows that are neigh bor i
        #choose a starting triangle that is on the boundary, neighbors i, and not chosen before 
        choice = rows[0]
        if choice in chosen and len(rows) > 1:
            choice = rows[1]
        chosen.append(choice)

        #order the triangle staring from the choisen starting point
        order_tri = order_func(tri_adj, choice)
    
        #reorder points
        points_reordered = points[order]
        #reorder the triangles
        tri_reordered = tri[order_tri]
        #replace the indices in the tri sequence by the new indices 
        #old to new index dictionary 
        dict = {}
        for i, val in enumerate(order):
            dict[val] = i
        try:
            tri_reordered = np.sort(np.vectorize(dict.__getitem__)(tri_reordered), axis=1)
            tri_reordered = np.vstack((tri_reordered, [N, N, N]))
        except:
            plot_knn_graph(points, adj)
    
        points_list.append(points_reordered)
        tri_list.append(tri_reordered)
    if train:
        return points_list, tri_list
    else:
        return [points_list[0]], [tri_list[0]]

def build_dataset_new(num_samples=10000, N=20, seed=42, padding_len=35, padding_val=-1, order=bfs_order, k=5, train=True):
    # Optional: Set the random seed for reproducibility
    np.random.seed(seed)
    
    #container
    X = []
    TRI = []
    
    for i in tqdm(range(num_samples)):
        points_list, tri_list = sample_delauney_new(N=N, order=order, k=k, train=train)
        tri_list = np.array(tri_list)

        for j in range(tri_list.shape[0]):
            X.append(points_list[j])
        
            N_tri = tri_list[j].shape[0]
            if N_tri > padding_len:
                raise ValueError(f"Input has {N_tri} rows, which exceeds the target length {padding_len}.")
            
            pad_rows = padding_len - N_tri
            padding = np.full((pad_rows, 3), padding_val, dtype=tri_list.dtype)
            tri = np.vstack([tri_list[j], padding])
            TRI.append(tri)
    return np.array(X), np.array(TRI)