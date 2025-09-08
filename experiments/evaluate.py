# experiments/evaluate.py

"""
Author: Wanzhou Lei @ Sept 2025. Email: wanzhou_lei@berkeley.edu
 GNNPointerNet Evaluation Script (experiments/evaluate.py)

This script:
  1) Loads a training run from a checkpoint directory, plots the loss/IoU trace and saves the figure in-place.
  2) Loads all intermediate checkpoints (model_epoch*.pt) and the final model (gnnpointernet_final.pt).
  3) Evaluates each model on train & test datasets w/o teacher forcing using IoU and accuracy; prints and logs results to evaluation.txt.

USAGE (from repo root):
    python -m experiments.evaluate \
        --checkpoint_dir experiments/checkpoints/2025-09-06_12-34-56 \
        --train_path datasets/train_10000_bfs_k5_N20.npz \
        --test_path  datasets/test_5000_bfs_k5_N20.npz
"""

import argparse
import os
import re
import json
import pickle
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from data.data import *
from gnnpointernet.models.model import *
from gnnpointernet.utils.util import *
from gnnpointernet.losses.loss_functions import *


def load_history_and_plot(checkpoint_dir):
    """Load training_history.pkl and (optionally) config.json, then plot traces."""

    history_path = os.path.join(checkpoint_dir, "training_history.pkl")
    if not os.path.isfile(history_path):
        raise FileNotFoundError(f"training_history.pkl not found at {history_path}")

    with open(history_path, "rb") as f:
        payload = pickle.load(f)

    # history may be nested under "history"; keep both cases robust
    if isinstance(payload, dict) and "history" in payload:
        history = payload["history"]
        config = payload.get("config", None)
    else:
        history = payload
        config = None

    # Try reading config.json if not bundled in history
    if config is None:
        cfg_path = os.path.join(checkpoint_dir, "config.json")
        if os.path.isfile(cfg_path):
            with open(cfg_path, "r") as cf:
                config = json.load(cf)

    # Extract traces
    loss_trace = history.get("train_loss", [])
    iou_trace = history.get("train_iou", [])

    # Plot as requested
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))

    ax[0].plot(loss_trace)
    ax[0].grid()
    ax[0].set_title("Loss Trace")
    ax[0].set_xlabel("Number of Updates")
    ax[0].set_ylabel("Negative Log Probability Losss")

    ax[1].plot(iou_trace)
    ax[1].grid()
    ax[1].set_title("IOU")
    ax[1].set_xlabel("Number of Updates")
    ax[1].set_ylabel("IOU")

    out_path = os.path.join(checkpoint_dir, "training_curves.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)

    return history, config, out_path


def build_model_from_config(config: dict, device: torch.device):
    """Instantiate GraphPointerNet using saved config (fallback to sensible defaults)."""
    
    embedding_dim = 32
    num_layers = 4
    num_layers_LSTM = 1
    hidden_dim = 64
    max_steps = 40
    attention = False
    num_cold_start = 0

    if config and "model" in config:
        m = config["model"]
        embedding_dim = m.get("embedding_dim", embedding_dim)
        num_layers = m.get("num_layers", num_layers)
        num_layers_LSTM = m.get("num_layers_LSTM", num_layers_LSTM)
        hidden_dim = m.get("hidden_dim", hidden_dim)
        max_steps = m.get("max_steps", max_steps)
        attention = m.get("attention", attention)
        num_cold_start = m.get("num_cold_start", num_cold_start)

    model = GraphPointerNet(
        2,  # in_dim
        4,  # edge_dim
        embedding_dim=embedding_dim,
        num_layers=num_layers,
        num_layers_LSTM=num_layers_LSTM,
        hidden_dim=hidden_dim,
        max_steps=max_steps,
        attention=attention,
        num_cold_start=num_cold_start,
    ).to(device)

    return model


@torch.no_grad()
def evaluate_loader(model, loader, k_for_knn, max_steps_decode, device):
    """
    Compute mean IoU and accuracy over a dataloader without teacher forcing.
    Robust to occasional batch-size mismatches between ground truth and predictions.
    """

    model.eval()
    iou_list = []
    acc_list = []

    for x, tri in tqdm(loader, desc="Evaluating", leave=False):
        x = x.to(device)
        tri = tri.to(device)

        data_batch = make_pyg_batch_wedge(x, k_for_knn)

        # Forward decode without teacher forcing
        logits, indices = model(data_batch, max_steps=max_steps_decode)

        # Determine batch sizes returned by model vs. ground truth
        if isinstance(indices, (list, tuple)):
            b_pred = len(indices)
        elif isinstance(indices, torch.Tensor):
            b_pred = indices.shape[0]
        else:
            # Fallback: try to coerce to tensor and read first dim
            indices = torch.as_tensor(indices)
            b_pred = indices.shape[0]

        b_true = tri.shape[0]
        if b_pred != b_true:
            # Keep only the overlapping portion
            n = min(b_pred, b_true)
            # Slice tri
            tri = tri[:n]
            # Slice indices
            if isinstance(indices, (list, tuple)):
                indices = indices[:n]
            else:
                indices = indices[:n]

        # Metrics
        iou_list.append(iou_triangles(tri, indices))
        acc_list.append(iou_accuracy(tri, indices))

    mean_iou = float(np.mean(iou_list)) if iou_list else 0.0
    mean_acc = float(np.mean(acc_list)) if acc_list else 0.0
    return mean_iou, mean_acc


def collect_checkpoint_paths(checkpoint_dir):
    """Collect all model_epoch*.pt and gnnpointernet_final.pt paths, sorted by epoch order."""

    files = os.listdir(checkpoint_dir)

    # Match model_epoch{N}.pt
    epoch_ckpts = []
    pattern = re.compile(r"model_epoch(\d+)\.pt$")
    for fname in files:
        m = pattern.match(fname)
        if m:
            epoch = int(m.group(1))
            epoch_ckpts.append((epoch, os.path.join(checkpoint_dir, fname)))
    epoch_ckpts.sort(key=lambda t: t[0])

    final_path = None
    final_fname = "gnnpointernet_final.pt"
    if final_fname in files:
        final_path = os.path.join(checkpoint_dir, final_fname)

    ordered = [p for _, p in epoch_ckpts]
    if final_path:
        ordered.append(final_path)
    return ordered


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    checkpoint_dir = args.checkpoint_dir
    if not os.path.isdir(checkpoint_dir):
        raise NotADirectoryError(f"{checkpoint_dir} is not a valid directory")

    #Load history & plot
    history, config, plot_path = load_history_and_plot(checkpoint_dir)
    print(f"Saved training curves to: {plot_path}")

    #Load datasets
    train_npz = np.load(args.train_path)
    test_npz = np.load(args.test_path)

    X_train, tri_train = torch.Tensor(train_npz["X"]), torch.Tensor(train_npz["tri"])
    X_test, tri_test = torch.Tensor(test_npz["X"]), torch.Tensor(test_npz["tri"])

    train_dataset = GraphDataSet_new(X_train, tri_train)
    test_dataset = GraphDataSet_new(X_test, tri_test)

    # Use a reasonable batch size for eval; can also read from config if desired
    eval_bs = 256
    train_loader = DataLoader(train_dataset, batch_size=eval_bs, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=eval_bs, shuffle=False)

    # Decide k and max_steps from config if available; fallback to 5 and 50
    k_for_knn = 5
    if config and "dataset" in config:
        k_for_knn = config["dataset"].get("k", k_for_knn)
    max_steps_decode = 50
    if config and "model" in config:
        max_steps_decode = config["model"].get("max_steps", max_steps_decode)

    # Collect model checkpoints to evaluate
    model_paths = collect_checkpoint_paths(checkpoint_dir)
    if not model_paths:
        raise FileNotFoundError(f"No model checkpoints found in {checkpoint_dir}")

    # Prepare evaluation log
    lines = []
    lines.append(f"Checkpoint directory: {checkpoint_dir}")
    lines.append(f"Train NPZ: {args.train_path}")
    lines.append(f"Test  NPZ: {args.test_path}")
    lines.append(f"k for kNN: {k_for_knn}")
    lines.append(f"max_steps (decode): {max_steps_decode}")
    lines.append("")

    # Evaluate each checkpoint
    for mp in model_paths:
        # build model with matching hyperparams from config
        model = build_model_from_config(config, device)

        # load weights
        state = torch.load(mp, map_location=device)
        model.load_state_dict(state)

        print(f"\nEvaluating checkpoint: {os.path.basename(mp)}")
        lines.append(f"== {os.path.basename(mp)} ==")

        # Train set (no teacher forcing)
        tr_iou, tr_acc = evaluate_loader(model, train_loader, k_for_knn, max_steps_decode, device)
        print(f"Train  | IoU: {tr_iou:.4f} | Acc: {tr_acc:.4f}")
        lines.append(f"Train:  IoU={tr_iou:.6f}, Acc={tr_acc:.6f}")

        # Test set (no teacher forcing)
        te_iou, te_acc = evaluate_loader(model, test_loader, k_for_knn, max_steps_decode, device)
        print(f"Test   | IoU: {te_iou:.4f} | Acc: {te_acc:.4f}")
        lines.append(f"Test:   IoU={te_iou:.6f}, Acc={te_acc:.6f}")
        lines.append("")

    # Write evaluation.txt
    eval_txt_path = os.path.join(checkpoint_dir, "evaluation.txt")
    with open(eval_txt_path, "w") as f:
        f.write("\n".join(lines))

    print(f"\n Evaluation summary written to: {eval_txt_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        required=True,
        help="Path to a specific run directory under checkpoints (contains .pt, training_history.pkl).",
    )
    parser.add_argument(
        "--train_path",
        type=str,
        required=False,
        default = "datasets/train_10000_bfs_k5_N20.npz",
        help="Path to training .npz file (e.g., datasets/train_10000_bfs_k5_N20.npz).",
    )
    parser.add_argument(
        "--test_path",
        type=str,
        required=False,
        default = "datasets/test_5000_bfs_k5_N20.npz",
        help="Path to test .npz file (e.g., datasets/test_5000_bfs_k5_N20.npz).",
    )
    args = parser.parse_args()
    main(args)
