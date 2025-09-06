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
    X, TRI = build_dataset_new(num_samples=5000, N=15, seed=43,
                               padding_len=25, padding_val=-1,
                               order=bfs_order, k=5, train=False)
    print(X.shape)
    print(TRI.shape)
    np.savez("../datasets/test_5000_bfs_k5_N15.npz", X=X, tri=TRI)

if __name__ == "__main__":
    main()
