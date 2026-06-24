"""
MoodLens - FER-2013 Dataset Loader

FER-2013 Kaggle structure after download:
  fer2013/
    train/
      angry/   disgust/   fear/   happy/   neutral/   sad/   surprise/
    test/
      angry/   disgust/   fear/   happy/   neutral/   sad/   surprise/

Download: https://www.kaggle.com/datasets/msambare/fer2013
"""

import os
from pathlib import Path

import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


# ── Label map (matches folder names) ─────────────────────────────────────────
EMOTIONS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
LABEL2IDX = {e: i for i, e in enumerate(EMOTIONS)}
IDX2LABEL = {i: e.capitalize() for i, e in enumerate(EMOTIONS)}


# ── Augmentations ─────────────────────────────────────────────────────────────

def get_train_transform():
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((48, 48)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),                         # [0,1]
        transforms.Normalize(mean=[0.5], std=[0.5]),   # [-1,1]
    ])


def get_val_transform():
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((48, 48)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])


# ── Dataset class ─────────────────────────────────────────────────────────────

class FER2013Dataset(Dataset):
    """
    Loads FER-2013 from the folder structure above.
    Each image is returned as (tensor[1,48,48], int label).
    """

    def __init__(self, root: str, split: str = "train", transform=None):
        """
        Args:
            root:      path to fer2013/ folder
            split:     'train' or 'test'
            transform: torchvision transform pipeline
        """
        self.root = Path(root) / split
        self.transform = transform
        self.samples = []  # list of (path, label_idx)

        if not self.root.exists():
            raise FileNotFoundError(
                f"Dataset not found at {self.root}\n"
                "Download from: https://www.kaggle.com/datasets/msambare/fer2013\n"
                "Then unzip so the path looks like:  fer2013/train/happy/..."
            )

        for emotion in EMOTIONS:
            emotion_dir = self.root / emotion
            if not emotion_dir.exists():
                continue
            for img_path in emotion_dir.iterdir():
                if img_path.suffix.lower() in (".jpg", ".jpeg", ".png"):
                    self.samples.append((str(img_path), LABEL2IDX[emotion]))

        print(f"[{split}] Loaded {len(self.samples)} samples across {len(EMOTIONS)} classes")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")  # convert to RGB first for ColorJitter
        if self.transform:
            img = self.transform(img)
        return img, label


# ── Class weight helper (handles FER-2013 imbalance) ─────────────────────────

def compute_class_weights(dataset: FER2013Dataset) -> torch.Tensor:
    """
    Returns inverse-frequency weights for CrossEntropyLoss(weight=...).
    FER-2013 has severe class imbalance (disgust ~1.5%, happy ~25%).
    """
    counts = np.zeros(len(EMOTIONS))
    for _, label in dataset.samples:
        counts[label] += 1
    weights = 1.0 / (counts + 1e-6)
    weights = weights / weights.sum() * len(EMOTIONS)
    return torch.FloatTensor(weights)


# ── DataLoader factory ────────────────────────────────────────────────────────

def get_dataloaders(data_root: str, batch_size: int = 64, num_workers: int = 4):
    train_ds = FER2013Dataset(data_root, "train", transform=get_train_transform())
    val_ds   = FER2013Dataset(data_root, "test",  transform=get_val_transform())

    class_weights = compute_class_weights(train_ds)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, class_weights


# ── Quick sanity check ────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    root = sys.argv[1] if len(sys.argv) > 1 else "./fer2013"
    train_loader, val_loader, weights = get_dataloaders(root, batch_size=32, num_workers=0)

    imgs, labels = next(iter(train_loader))
    print(f"Batch shape : {imgs.shape}")    # (32, 1, 48, 48)
    print(f"Labels      : {labels[:8]}")
    print(f"Class weights: {weights.round(decimals=2)}")
