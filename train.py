"""
MoodLens - Training Script

Usage:
    python train.py --data ./fer2013 --epochs 50 --batch 64 --lr 1e-3

After training you'll get:
    checkpoints/best_model.pth   ← use this in the Streamlit app
    checkpoints/last_model.pth
    logs/training_log.json       ← loss/acc curves for Day 3 UI
"""

import argparse
import json
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from model import EmotionCNN
from dataset import get_dataloaders, IDX2LABEL


# ── Argument parser ───────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Train MoodLens EmotionCNN")
    p.add_argument("--data",    default="./fer2013", help="Path to fer2013/ folder")
    p.add_argument("--epochs",  type=int,   default=50)
    p.add_argument("--batch",   type=int,   default=64)
    p.add_argument("--lr",      type=float, default=1e-3)
    p.add_argument("--workers", type=int,   default=4)
    p.add_argument("--dropout", type=float, default=0.5)
    p.add_argument("--patience",type=int,   default=8,  help="Early stopping patience")
    p.add_argument("--ckpt",    default="./checkpoints")
    p.add_argument("--log",     default="./logs")
    p.add_argument("--resume",  default=None, help="Path to checkpoint to resume from")
    return p.parse_args()


# ── Training loop helpers ─────────────────────────────────────────────────────

def train_one_epoch(model, loader, criterion, optimizer, device, epoch):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (imgs, labels) in enumerate(loader):
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()
        logits = model(imgs)
        loss = criterion(logits, labels)
        loss.backward()

        # Gradient clipping — helps with unstable early training
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        total_loss += loss.item() * imgs.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += imgs.size(0)

        if (batch_idx + 1) % 50 == 0:
            running_acc = correct / total * 100
            print(f"  Epoch {epoch} [{batch_idx+1}/{len(loader)}] "
                  f"loss={loss.item():.4f}  acc={running_acc:.1f}%")

    return total_loss / total, correct / total * 100


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    # Per-class accuracy
    class_correct = torch.zeros(7)
    class_total   = torch.zeros(7)

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        logits = model(imgs)
        loss = criterion(logits, labels)

        total_loss += loss.item() * imgs.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += imgs.size(0)

        for c in range(7):
            mask = labels == c
            class_correct[c] += (preds[mask] == labels[mask]).sum().item()
            class_total[c]   += mask.sum().item()

    per_class = {
        IDX2LABEL[i]: round((class_correct[i] / (class_total[i] + 1e-6) * 100).item(), 1)
        for i in range(7)
    }

    return total_loss / total, correct / total * 100, per_class


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # Dirs
    Path(args.ckpt).mkdir(parents=True, exist_ok=True)
    Path(args.log).mkdir(parents=True, exist_ok=True)

    # Device
    device = (
        torch.device("cuda") if torch.cuda.is_available()
        else torch.device("mps") if torch.backends.mps.is_available()
        else torch.device("cpu")
    )
    print(f"\n{'='*50}")
    print(f"  MoodLens Training")
    print(f"  Device  : {device}")
    print(f"  Epochs  : {args.epochs}")
    print(f"  Batch   : {args.batch}")
    print(f"  LR      : {args.lr}")
    print(f"{'='*50}\n")

    # Data
    train_loader, val_loader, class_weights = get_dataloaders(
        args.data, batch_size=args.batch, num_workers=args.workers
    )
    class_weights = class_weights.to(device)

    # Model
    model = EmotionCNN(num_classes=7, dropout=args.dropout).to(device)

    # Loss — weighted to handle FER-2013 class imbalance
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Optimizer + scheduler
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3, verbose=True
    )

    # Resume
    start_epoch = 1
    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optim_state"])
        start_epoch = ckpt["epoch"] + 1
        print(f"Resumed from epoch {ckpt['epoch']} (val_acc={ckpt['val_acc']:.2f}%)")

    # Training log
    log = {
        "train_loss": [], "train_acc": [],
        "val_loss":   [], "val_acc":   [],
        "per_class_acc": [],
        "lr": [],
    }

    best_val_acc = 0.0
    patience_counter = 0

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        val_loss, val_acc, per_class = evaluate(
            model, val_loader, criterion, device
        )
        scheduler.step(val_acc)

        elapsed = time.time() - t0
        current_lr = optimizer.param_groups[0]["lr"]

        print(f"\nEpoch {epoch:03d}/{args.epochs} ({elapsed:.1f}s) | "
              f"LR={current_lr:.2e} | "
              f"Train loss={train_loss:.4f} acc={train_acc:.2f}% | "
              f"Val loss={val_loss:.4f} acc={val_acc:.2f}%")
        print(f"  Per-class val acc: {per_class}")

        # Log
        log["train_loss"].append(round(train_loss, 4))
        log["train_acc"].append(round(train_acc, 2))
        log["val_loss"].append(round(val_loss, 4))
        log["val_acc"].append(round(val_acc, 2))
        log["per_class_acc"].append(per_class)
        log["lr"].append(current_lr)

        with open(f"{args.log}/training_log.json", "w") as f:
            json.dump(log, f, indent=2)

        # Save last
        torch.save({
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optim_state": optimizer.state_dict(),
            "val_acc": val_acc,
            "config": vars(args),
        }, f"{args.ckpt}/last_model.pth")

        # Save best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optim_state": optimizer.state_dict(),
                "val_acc": val_acc,
                "config": vars(args),
            }, f"{args.ckpt}/best_model.pth")
            print(f"  ✅ New best! val_acc={val_acc:.2f}%  → saved best_model.pth")
        else:
            patience_counter += 1
            print(f"  No improvement. Patience: {patience_counter}/{args.patience}")

        # Early stopping
        if patience_counter >= args.patience:
            print(f"\nEarly stopping at epoch {epoch}. Best val_acc={best_val_acc:.2f}%")
            break

    print(f"\n{'='*50}")
    print(f"Training complete.")
    print(f"Best val accuracy : {best_val_acc:.2f}%")
    print(f"Checkpoint saved  : {args.ckpt}/best_model.pth")
    print(f"Training log      : {args.log}/training_log.json")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    main()
