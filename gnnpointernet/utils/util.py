from matplotlib import pyplot as plt
import torch
import numpy as np

def plot_graph(x: torch.Tensor, adj: torch.Tensor):
    """
    Plots a 2D graph based on node coordinates and an adjacency matrix.

    Args:
        x (torch.Tensor): Tensor of shape (N, 2), where each row is a 2D coordinate.
        adj (torch.Tensor): Tensor of shape (N, N), adjacency matrix.
    """
    x = x.cpu().numpy()
    adj = adj.cpu().numpy()
    num_nodes = x.shape[0]

    plt.figure(figsize=(6, 6))

    # Plot edges
    for i in range(num_nodes):
        for j in range(i+1, num_nodes):  # avoid plotting both (i,j) and (j,i)
            if adj[i, j] != 0:
                x_coords = [x[i, 0], x[j, 0]]
                y_coords = [x[i, 1], x[j, 1]]
                plt.plot(x_coords, y_coords, 'k-', linewidth=0.5)  # edge as black line

    # Plot nodes
    plt.scatter(x[:, 0], x[:, 1], c='blue', s=30, zorder=3)

    for i in range(num_nodes):
        plt.text(x[i, 0], x[i, 1], str(i), fontsize=8, ha='center', va='center', color='white',
                 bbox=dict(facecolor='blue', edgecolor='none', boxstyle='circle,pad=0.2'))

    plt.title("Graph Visualization")
    plt.axis('equal')
    plt.grid(True)
    plt.show()



