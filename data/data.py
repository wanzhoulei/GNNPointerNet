import numpy as np
from scipy.spatial import Delaunay

import matplotlib.pyplot as plt
import torch

import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm

#this function creates and retunrs an adjacency matrix by taking the triangle list
def get_adj_mtx(tri, N):
    adj = np.zeros((N, N))
    for triangle in tri.simplices:
        adj[triangle[0], triangle[1]] = 1
        adj[triangle[1], triangle[0]] = 1
        adj[triangle[0], triangle[2]] = 1
        adj[triangle[2], triangle[0]] = 1
        adj[triangle[1], triangle[2]] = 1
        adj[triangle[2], triangle[1]] = 1
    return adj

def sample_delauney(N=20):
    '''
    This function samples a 2d Delauney triangularization of N points

    '''
    points = np.random.rand(N, 2)
    tri = Delaunay(points)
    adj = get_adj_mtx(tri, N)
    return points, adj

def build_dataset(num_samples=10000, N=20, seed=42):
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
        
    Returns
    -------
    X : np.ndarray
        Shape (num_samples, N, 2), each row is the set of 2D points.
    Y : np.ndarray
        Shape (num_samples, N, N), each corresponding adjacency matrix.
    """
    # Optional: Set the random seed for reproducibility
    np.random.seed(seed)
    
    # Preallocate arrays
    X = np.zeros((num_samples, N, 2), dtype=np.float32)
    Y = np.zeros((num_samples, N, N), dtype=np.float32)
    
    for i in range(num_samples):
        points, adj = sample_delauney(N=N)
        X[i] = points
        Y[i] = adj
    
    return X, Y

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

