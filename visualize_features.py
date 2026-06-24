"""
MoodLens - Feature Map Visualizer

Run AFTER training to understand what each conv layer learned.

Usage:
    python visualize_features.py --ckpt ./checkpoints/best_model.pth --image path/to/face.jpg

Saves:
    outputs/feature_maps_block1.png
    outputs/feature_maps_block2.png
    outputs/feature_maps_block3.png
    outputs/filters_block1.png
"""

import argparse
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image
from torchvision import transforms

from model import EmotionCNN
from dataset import get_val_transform


# ── Hook to capture intermediate activations ──────────────────────────────────

class FeatureExtractor:
    def __init__(self, model: EmotionCNN):
        self.model = model
        self.features = {}
        self._register_hooks()

    def _register_hooks(self):
        def make_hook(name):
            def hook(module, input, output):
                self.features[name] = output.detach().cpu()
            return hook

        self.model.block1.register_forward_hook(make_hook("block1"))
        self.model.block2.register_forward_hook(make_hook("block2"))
        self.model.block3.register_forward_hook(make_hook("block3"))

    def forward(self, x):
        with torch.no_grad():
            out = self.model(x)
        return out


# ── Plotting helpers ──────────────────────────────────────────────────────────

def plot_feature_maps(feature_tensor, block_name, save_path, max_channels=32):
    """
    feature_tensor: (1, C, H, W)  — single image
    Plots the first max_channels feature maps in a grid.
    """
    maps = feature_tensor[0]  # (C, H, W)
    n = min(maps.shape[0], max_channels)
    cols = 8
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.5, rows * 1.5))
    fig.suptitle(f"Feature Maps — {block_name}", fontsize=14, fontweight="bold")

    for i in range(rows * cols):
        ax = axes[i // cols][i % cols] if rows > 1 else axes[i % cols]
        ax.axis("off")
        if i < n:
            fmap = maps[i].numpy()
            ax.imshow(fmap, cmap="viridis")
            ax.set_title(f"ch {i}", fontsize=6)

    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def plot_conv_filters(model: EmotionCNN, save_path):
    """Visualize the first conv layer's 32 learned filters (3×3 kernels)."""
    # First conv in block1
    first_conv = model.block1[0]
    weights = first_conv.weight.detach().cpu()  # (32, 1, 3, 3)

    n = weights.shape[0]
    cols = 8
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.2, rows * 1.2))
    fig.suptitle("Learned Conv Filters — Block 1, Layer 1", fontsize=13, fontweight="bold")

    for i in range(rows * cols):
        ax = axes[i // cols][i % cols] if rows > 1 else axes[i % cols]
        ax.axis("off")
        if i < n:
            f = weights[i, 0].numpy()
            # Normalize to [0,1] for display
            f = (f - f.min()) / (f.max() - f.min() + 1e-8)
            ax.imshow(f, cmap="gray", interpolation="nearest")
            ax.set_title(f"f{i}", fontsize=6)

    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def plot_training_curves(log_path, save_path):
    """Plot loss & accuracy curves from training_log.json."""
    import json
    with open(log_path) as f:
        log = json.load(f)

    epochs = range(1, len(log["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(epochs, log["train_loss"], label="Train Loss", color="#e74c3c")
    ax1.plot(epochs, log["val_loss"],   label="Val Loss",   color="#3498db")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
    ax1.set_title("Loss Curves"); ax1.legend(); ax1.grid(alpha=0.3)

    ax2.plot(epochs, log["train_acc"], label="Train Acc", color="#e74c3c")
    ax2.plot(epochs, log["val_acc"],   label="Val Acc",   color="#3498db")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Accuracy (%)")
    ax2.set_title("Accuracy Curves"); ax2.legend(); ax2.grid(alpha=0.3)

    fig.suptitle("MoodLens Training Curves", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt",  required=True, help="Path to best_model.pth")
    parser.add_argument("--image", required=True, help="Path to a face image")
    parser.add_argument("--log",   default="./logs/training_log.json")
    parser.add_argument("--out",   default="./outputs")
    args = parser.parse_args()

    Path(args.out).mkdir(parents=True, exist_ok=True)

    # Load model
    device = torch.device("cpu")
    model = EmotionCNN()
    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"Loaded checkpoint (val_acc={ckpt['val_acc']:.2f}%)")

    # Load & preprocess image
    img = Image.open(args.image).convert("RGB")
    transform = get_val_transform()
    tensor = transform(img).unsqueeze(0)  # (1,1,48,48)

    # Predict
    idx, emotion, probs = model.predict(tensor)
    print(f"\nPrediction: {emotion}")
    print("Probabilities:")
    for k, v in sorted(probs.items(), key=lambda x: -x[1]):
        bar = "█" * int(v / 2)
        print(f"  {k:<10} {v:5.1f}%  {bar}")

    # Extract feature maps
    extractor = FeatureExtractor(model)
    extractor.forward(tensor)

    # Plot
    plot_conv_filters(model, f"{args.out}/filters_block1.png")
    plot_feature_maps(extractor.features["block1"], "Block 1 (64ch, 24×24)", f"{args.out}/feature_maps_block1.png")
    plot_feature_maps(extractor.features["block2"], "Block 2 (128ch, 12×12)", f"{args.out}/feature_maps_block2.png")
    plot_feature_maps(extractor.features["block3"], "Block 3 (256ch, 6×6)",   f"{args.out}/feature_maps_block3.png")

    # Training curves (if log exists)
    if Path(args.log).exists():
        plot_training_curves(args.log, f"{args.out}/training_curves.png")
    else:
        print(f"(No training log found at {args.log} — skipping curves)")

    print(f"\nAll visualizations saved to {args.out}/")


if __name__ == "__main__":
    main()
