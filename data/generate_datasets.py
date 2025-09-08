## Author: Wanzhou Lei @ Sept 2025. Email: wanzhou_lei@berkeley.edu

## Script generates the two train and test dataset we need for training and evaluation. 
## The training set contains 10,000 delaunay graphs of 20 points.
## The testing set contains 5,000 delaunay graphs of 20 points. 
## Both uses a knn graph of k = 5 to traverse the coordinates. 
## Note that one graph can be traversed in many different orders, resulting in more sequences in the dataset. 

import numpy as np
from data import *

def main():
    # Training set
    X, TRI = build_dataset_new(num_samples=10000, N=20, seed=41,
                               padding_len=35, padding_val=-1,
                               order=bfs_order, k=5)
    print(X.shape)
    print(TRI.shape)
    np.savez("../datasets/train_10000_bfs_k5_N20.npz", X=X, tri=TRI)

    # Test set
    X, TRI = build_dataset_new(num_samples=5000, N=20, seed=43,
                               padding_len=35, padding_val=-1,
                               order=bfs_order, k=5, train=False)
    print(X.shape)
    print(TRI.shape)
    np.savez("../datasets/test_5000_bfs_k5_N20.npz", X=X, tri=TRI)

if __name__ == "__main__":
    main()
