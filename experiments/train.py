# experiments/train.py

##usage 
#basic run: python -m experiments.train
#change dataset and run fewer epochs: python -m experiments.train --train_file train_small.npz --test_file test_small.npz --epochs 20
#with smaller batch size: python -m experiments.train --batch_size 64
#change save directory and frequency: python -m experiments.train --save_dir results/run1 --save_every 10


import argparse
import os
import pickle
import numpy as np
from tqdm import tqdm

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from data.data import *
from gnnpointernet.models.model import *
from gnnpointernet.losses.loss_functions import *


def train_one_epoch(model, loader, optimizer, scheduler, device, k):
    """Train model for one epoch and return avg loss and IoU."""
    model.train()
    epoch_loss, epoch_iou = [], []

    for x, tri in tqdm(loader, desc="Training", leave=False):
        x, tri = x.to(device), tri.to(device)

        # prepare pyg batch
        data_batch = make_pyg_batch_wedge(x, k)

        # forward
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

    # Load datasets
    train_data = np.load(os.path.join(args.dataset_dir, args.train_file))
    test_data = np.load(os.path.join(args.dataset_dir, args.test_file))

    X_train, tri_train = torch.Tensor(train_data["X"]), torch.Tensor(train_data["tri"])
    X_test, tri_test = torch.Tensor(test_data["X"]), torch.Tensor(test_data["tri"])

    train_dataset = GraphDataSet_new(X_train, tri_train)
    test_dataset = GraphDataSet_new(X_test, tri_test)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # Build model
    model = GraphPointerNet(2, 4,
        embedding_dim=32,
        num_layers=5,
        num_layers_LSTM=1,
        hidden_dim=256,
        max_steps=40,
        attention=True,
        num_cold_start=0,
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2000, verbose=False
    )

    # Training loop
    history = {"train_loss": [], "train_iou": []}

    for epoch in range(args.epochs):
        avg_loss, avg_iou = train_one_epoch(model, train_loader, optimizer, scheduler, device, args.k)

        history["train_loss"].append(avg_loss)
        history["train_iou"].append(avg_iou)

        print(f"Epoch {epoch+1}/{args.epochs} | Loss: {avg_loss:.4f} | IoU: {avg_iou:.4f}")

        # save checkpoint every few epochs
        if (epoch + 1) % args.save_every == 0:
            ckpt_path = os.path.join(args.save_dir, f"model_epoch{epoch+1}.pt")
            torch.save(model.state_dict(), ckpt_path)
            print(f"✅ Saved checkpoint: {ckpt_path}")

    # Save final model + logs
    os.makedirs(args.save_dir, exist_ok=True)

    final_model_path = os.path.join(args.save_dir, "gnnpointernet_final.pt")
    torch.save(model.state_dict(), final_model_path)
    print(f"Training complete! Model saved to {final_model_path}")

    history_path = os.path.join(args.save_dir, "training_history.pkl")
    with open(history_path, "wb") as f:
        pickle.dump(history, f)
    print(f"Training history saved to {history_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # dataset args
    parser.add_argument("--dataset_dir", type=str, default="datasets")
    parser.add_argument("--train_file", type=str, default="train_10000_bfs_k5_N20.npz")
    parser.add_argument("--test_file", type=str, default="test_5000_bfs_k5_N15.npz")

    # training args
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--k", type=int, default=5)

    # logging/checkpoint args
    parser.add_argument("--save_dir", type=str, default="experiments/checkpoints")
    parser.add_argument("--save_every", type=int, default=50)

    args = parser.parse_args()
    main(args)
