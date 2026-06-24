"""
MoodLens - CNN Model Architecture
FER-2013 Face Emotion Recognition
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class EmotionCNN(nn.Module):
    """
    CNN trained from scratch on FER-2013.
    Input:  (B, 1, 48, 48) grayscale face crops
    Output: (B, 7) logits for 7 emotion classes
    """

    EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

    def __init__(self, num_classes=7, dropout=0.5):
        super(EmotionCNN, self).__init__()

        # ── Block 1 ───────────────────────────────────────────────────────────
        # 48×48 → 48×48 → 24×24
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),   # (B,32,48,48)
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # (B,64,48,48)
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),                           # (B,64,24,24)
            nn.Dropout2d(0.25),
        )

        # ── Block 2 ───────────────────────────────────────────────────────────
        # 24×24 → 24×24 → 12×12
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),  # (B,128,24,24)
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1), # (B,128,24,24)
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),                            # (B,128,12,12)
            nn.Dropout2d(0.25),
        )

        # ── Block 3 ───────────────────────────────────────────────────────────
        # 12×12 → 12×12 → 6×6
        self.block3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),  # (B,256,12,12)
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),  # (B,256,12,12)
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),                             # (B,256,6,6)
            nn.Dropout2d(0.25),
        )

        # ── Classifier ────────────────────────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.Flatten(),                    # (B, 256*6*6) = (B, 9216)
            nn.Linear(256 * 6 * 6, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(1024, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout / 2),
            nn.Linear(256, num_classes),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.classifier(x)
        return x

    def predict(self, x):
        """Returns (class_idx, class_name, probabilities_dict)"""
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probs = F.softmax(logits, dim=1)
            idx = probs.argmax(dim=1).item()
            prob_dict = {
                emotion: round(probs[0][i].item() * 100, 2)
                for i, emotion in enumerate(self.EMOTIONS)
            }
        return idx, self.EMOTIONS[idx], prob_dict


# ── Sanity check ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    model = EmotionCNN()
    print(model)
    total = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {total:,}")

    dummy = torch.randn(4, 1, 48, 48)
    out = model(dummy)
    print(f"Output shape: {out.shape}")  # (4, 7)
