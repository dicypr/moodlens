# MoodLens 🎭 — Deploy to Streamlit Cloud

## Final file structure

```
moodlens/
├── app.py                        ← Streamlit app
├── model.py                      ← CNN architecture
├── dataset.py                    ← Data loader
├── inference.py                  ← Full pipeline
├── gradcam.py                    ← Grad-CAM
├── face_detector.py              ← OpenCV face detection
├── requirements.txt              ← Python deps
├── packages.txt                  ← System deps (for Streamlit Cloud)
├── .streamlit/
│   └── config.toml               ← Dark theme config
└── checkpoints/
    └── best_model.pth            ← Your trained model ← REQUIRED
```

---

## Step 1 — Test locally first

```bash
pip install -r requirements.txt
streamlit run app.py
```

Opens at http://localhost:8501

---

## Step 2 — Push to GitHub

```bash
git init
git add .
git commit -m "feat: MoodLens emotion recognition app"
git remote add origin https://github.com/YOUR_USERNAME/moodlens.git
git push -u origin main
```

⚠️ `best_model.pth` is ~40MB. Add it with Git LFS:
```bash
git lfs install
git lfs track "*.pth"
git add .gitattributes
git add checkpoints/best_model.pth
git commit -m "add model checkpoint"
git push
```

---

## Step 3 — Deploy on Streamlit Cloud

1. Go to https://share.streamlit.io
2. Click **New app**
3. Select your GitHub repo
4. Set **Main file path** → `app.py`
5. Click **Deploy**

Done. You'll get a public URL like:
`https://YOUR_USERNAME-moodlens-app-XXXX.streamlit.app`

---

## What the app shows

- Upload any photo → auto-detects faces
- Shows annotated image with bounding box + emotion label
- Face crop panel
- Grad-CAM heatmap (what the CNN focused on)
- Confidence bar chart for all 7 emotions
- Training history curves in sidebar
- Handles multiple faces in one photo
