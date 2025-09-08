# experiments/train.py

"""
Author: Wanzhou Lei @ Sept 2025. Email: wanzhou_lei@berkeley.edu

This script trains the GNNPointerNet model on pre-generated datasets of 2D point clouds and Delaunay triangulations.
It supports configurable model hyperparameters and training options via command-line arguments. Each training run creates
a timestamped subdirectory inside the checkpoints folder that stores the model weights, logs, and configuration files.

USAGE
From the repository root, run:
    python -m experiments.train [OPTIONS]
Examples:
    # Run with default settings
    python -m experiments.train

    # Train with custom hyperparameters
    python -m experiments.train --embedding_dim 64 --num_layers 6 --hidden_dim 128 --attention true --epochs 100 --save_dir experiments/checkpoints

ARGUMENTS
Dataset arguments:
    --dataset_dir       Path to dataset folder (default: "datasets")
    --train_file        Training dataset file (default: "train_10000_bfs_k5_N20.npz")
    --test_file         Test dataset file (default: "test_5000_bfs_k5_N15.npz")
    --k                 k for kNN graph construction (default: 5)
Training arguments:
    --batch_size        Training batch size (default: 256)
    --epochs            Number of training epochs (default: 200)
    --lr                Learning rate (default: 1e-3)
Model hyperparameters:
    --embedding_dim     Node embedding dimension (default: 32)
    --num_layers        Number of GNN message-passing layers (default: 5)
    --num_layers_LSTM   Number of LSTM layers in pointer network (default: 1)
    --hidden_dim        Hidden dimension of LSTM (default: 256)
    --max_steps         Maximum decoding steps (default: 40)
    --attention         Whether to use attention in decoder (default: True)
    --num_cold_start    Cold-start tokens (default: 0)
Logging / checkpoint arguments:
    --save_dir          Root folder to store outputs (default: "experiments/checkpoints")
    --save_every        Save checkpoint every N epochs (default: 50)

OUTPUT
For each run, the script creates a timestamped subfolder under the save directory (e.g. "experiments/checkpoints/2025-09-06-12-34/"):
    - config.json              : JSON file of all run arguments
    - training_history.pkl     : Pickled dict with loss/IoU trace + config
    - model_epoch{N}.pt        : Checkpoint(s) saved every N epochs
    - gnnpointernet_final.pt   : Final trained model weights

NOTES
- Run the script from the repository root so imports resolve correctly.
- Datasets (.npz files) should be placed under the folder specified by --dataset_dir.
"""


import argparse
import os
import json
import pickle
import numpy as np
from datetime import datetime
from tqdm import tqdm

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from data.data import *
from gnnpointernet.models.model import *
from gnnpointernet.losses.loss_functions import *


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    if v.lower() in ("no", "false", "f", "n", "0"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def train_one_epoch(model, loader, optimizer, scheduler, device, k):
    """Train model for one epoch and return avg loss and IoU."""
    model.train()
    epoch_loss, epoch_iou = [], []

    for x, tri in tqdm(loader, desc="Training", leave=False):
        x, tri = x.to(device), tri.to(device)

        # prepare pyg batch
        data_batch = make_pyg_batch_wedge(x, k)

        # forward (teacher forcing)
        logits, indices = model(data_batch, teacher_indices=tri)

        # loss
        loss = loss_neg_log(logits, tri)

        # backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step(loss.item())

        epoch_loss.append(loss.item())
        epoch_iou.append(iou_triangles(tri, indices))

    return np.mean(epoch_loss), np.mean(epoch_iou)


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Prepare run directory
    # e.g., experiments/checkpoints/2025-09-06-11-42
    run_stamp = datetime.now().strftime("%Y-%m-%d-%H-%M")
    run_dir = os.path.join(args.save_dir, run_stamp)
    os.makedirs(run_dir, exist_ok=True)
    print(f"Logs & checkpoints will be saved to: {run_dir}")

    # Load datasets
    train_data = np.load(os.path.join(args.dataset_dir, args.train_file))
    test_data = np.load(os.path.join(args.dataset_dir, args.test_file))
    X_train, tri_train = torch.Tensor(train_data["X"]), torch.Tensor(train_data["tri"])
    train_dataset = GraphDataSet_new(X_train, tri_train)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    # Build model (using CLI hyperparams)
    model = GraphPointerNet(
        2,  # in_dim
        4,  # edge_dim
        embedding_dim=args.embedding_dim,
        num_layers=args.num_layers,
        num_layers_LSTM=args.num_layers_LSTM,
        hidden_dim=args.hidden_dim,
        max_steps=args.max_steps,
        attention=args.attention,
        num_cold_start=args.num_cold_start,
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2000, verbose=False
    )

    # Save run config (hyperparams + metadata)
    run_config = {
        "device": str(device),
        "dataset": {
            "dataset_dir": args.dataset_dir,
            "train_file": args.train_file,
            "test_file": args.test_file,
            "k": args.k,
            "batch_size": args.batch_size,
        },
        "training": {
            "epochs": args.epochs,
            "lr": args.lr,
            "save_every": args.save_every,
        },
        "model": {
            "embedding_dim": args.embedding_dim,
            "num_layers": args.num_layers,
            "num_layers_LSTM": args.num_layers_LSTM,
            "hidden_dim": args.hidden_dim,
            "max_steps": args.max_steps,
            "attention": args.attention,
            "num_cold_start": args.num_cold_start,
            "in_dim": 2,
            "edge_dim": 4,
        },
        "run_stamp": run_stamp,
    }
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(run_config, f, indent=2)
    print(f"Saved run config to {os.path.join(run_dir, 'config.json')}")

    # Training loop
    history = {"train_loss": [], "train_iou": []}

    for epoch in range(args.epochs):
        avg_loss, avg_iou = train_one_epoch(
            model, train_loader, optimizer, scheduler, device, args.k
        )

        history["train_loss"].append(float(avg_loss))
        history["train_iou"].append(float(avg_iou))

        print(f"Epoch {epoch+1}/{args.epochs} | Loss: {avg_loss:.4f} | IoU: {avg_iou:.4f}")

        # save checkpoint every few epochs
        if (epoch + 1) % args.save_every == 0:
            ckpt_path = os.path.join(run_dir, f"model_epoch{epoch+1}.pt")
            torch.save(model.state_dict(), ckpt_path)
            print(f"Saved checkpoint: {ckpt_path}")

    # Save final model and logs
    final_model_path = os.path.join(run_dir, "gnnpointernet_final.pt")
    torch.save(model.state_dict(), final_model_path)
    print(f"Training complete! Model saved to {final_model_path}")

    history_path = os.path.join(run_dir, "training_history.pkl")
    with open(history_path, "wb") as f:
        pickle.dump(
            {
                "history": history,
                "config": run_config, 
            },
            f,
        )
    print(f"Training history saved to {history_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # dataset args
    parser.add_argument("--dataset_dir", type=str, default="datasets")
    parser.add_argument("--train_file", type=str, default="train_10000_bfs_k5_N20.npz")
    parser.add_argument("--test_file", type=str, default="test_5000_bfs_k5_N15.npz")
    parser.add_argument("--k", type=int, default=5, help="k for kNN graph construction")

    # training args
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)

    # model hyperparameters (now overridable via CLI)
    parser.add_argument("--embedding_dim", type=int, default=32)
    parser.add_argument("--num_layers", type=int, default=5)
    parser.add_argument("--num_layers_LSTM", type=int, default=1)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--max_steps", type=int, default=40)
    parser.add_argument("--attention", type=str2bool, default=True)
    parser.add_argument("--num_cold_start", type=int, default=0)

    # logging/checkpoint args
    parser.add_argument("--save_dir", type=str, default="experiments/checkpoints")
    parser.add_argument("--save_every", type=int, default=50)

    args = parser.parse_args()
    main(args)
