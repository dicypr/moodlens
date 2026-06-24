# MoodLens 🎭 — Face Emotion Recognition CNN

A CNN trained from scratch on FER-2013 to classify 7 emotions:  
**Angry · Disgust · Fear · Happy · Neutral · Sad · Surprise**

---

## Project Structure

```
moodlens/
├── model.py               # CNN architecture (EmotionCNN)
├── dataset.py             # FER-2013 dataloader + augmentations
├── train.py               # Training script with checkpointing
├── visualize_features.py  # Feature maps + filter visualizer (run after training)
├── requirements.txt
└── README.md
```

---

## Day 1 — Setup & Train

### Step 1 — Install dependencies

```bash
pip install torch torchvision Pillow numpy matplotlib
```

### Step 2 — Download FER-2013 dataset

1. Go to: https://www.kaggle.com/datasets/msambare/fer2013
2. Download and unzip so the structure looks like:

```
fer2013/
  train/
    angry/      (3995 images)
    disgust/    (436 images)
    fear/       (4097 images)
    happy/      (7215 images)
    neutral/    (4965 images)
    sad/        (4830 images)
    surprise/   (3171 images)
  test/
    angry/  disgust/  fear/  ...
```

### Step 3 — Verify dataset loads

```bash
python dataset.py ./fer2013
# Expected output:
# [train] Loaded 28709 samples across 7 classes
# [test]  Loaded 7178 samples across 7 classes
```

### Step 4 — Train

```bash
# On CPU (slow but works, ~30min/epoch):
python train.py --data ./fer2013 --epochs 50 --batch 32 --workers 0

# On GPU (recommended, ~3-5min/epoch):
python train.py --data ./fer2013 --epochs 50 --batch 64 --workers 4

# On Google Colab (free GPU) — recommended:
# Upload this folder to Colab, mount Drive, run the command above
```

Training saves:
- `checkpoints/best_model.pth` — use this in the Streamlit app
- `logs/training_log.json`     — loss/acc curves

### Step 5 — Visualize what the CNN learned

```bash
python visualize_features.py \
  --ckpt ./checkpoints/best_model.pth \
  --image path/to/any/face.jpg

# Saves to outputs/:
#   filters_block1.png         ← 32 learned conv filters
#   feature_maps_block1.png    ← what block 1 activates on
#   feature_maps_block2.png
#   feature_maps_block3.png
#   training_curves.png
```

---

## Expected Results

| Metric | Expected |
|---|---|
| Training accuracy | ~85–90% |
| Validation accuracy | ~63–68% |
| Human-level on FER-2013 | ~65% |

FER-2013 is a hard dataset — 65% val acc is actually matching human performance.  
"Disgust" will have the lowest per-class acc (~30–40%) due to very few samples (436 train).

---

## Architecture Summary

```
Input (1×48×48)
  Block 1: Conv32 → BN → ReLU → Conv64 → BN → ReLU → MaxPool → Dropout
  Block 2: Conv128 → BN → ReLU → Conv128 → BN → ReLU → MaxPool → Dropout  
  Block 3: Conv256 → BN → ReLU → Conv256 → BN → ReLU → MaxPool → Dropout
  Flatten → Dense(1024) → Dropout → Dense(256) → Dropout → Dense(7)
  
Total params: ~9.2M
```

Key design decisions:
- **BatchNorm** after every Conv — stabilizes training, allows higher LR
- **Kaiming init** for Conv layers — standard for ReLU networks
- **Class-weighted CrossEntropy** — handles FER-2013 imbalance (disgust << happy)
- **ReduceLROnPlateau** — halves LR when val acc plateaus
- **Early stopping** (patience=8) — avoids overfit

---

## Day 2 (tomorrow)
- OpenCV face detection (Haar cascade)
- Grad-CAM heatmap implementation
- Per-image inference pipeline

## Day 3
- Streamlit UI (upload + webcam + Grad-CAM + confidence bars)
- Deploy to Streamlit Cloud
