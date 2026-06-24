"""
MoodLens — Face Emotion Recognition
Streamlit App (Day 3)
"""

import io
import json
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image
import plotly.graph_objects as go

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="MoodLens",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=Space+Mono:wght@400;700&display=swap');

/* ── Reset & base ── */
html, body, [class*="css"] {
    font-family: 'Space Grotesk', sans-serif;
}
.stApp {
    background: #0a0a0f;
    color: #e8e8f0;
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 3rem 4rem; max-width: 1200px; }

/* ── Header ── */
.ml-header {
    display: flex;
    align-items: baseline;
    gap: 16px;
    margin-bottom: 4px;
}
.ml-logo {
    font-family: 'Space Mono', monospace;
    font-size: 2.4rem;
    font-weight: 700;
    color: #ffffff;
    letter-spacing: -1px;
}
.ml-logo span {
    color: #7c6af7;
}
.ml-tagline {
    font-size: 0.85rem;
    color: #6b6b80;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    font-weight: 500;
}
.ml-divider {
    height: 1px;
    background: linear-gradient(90deg, #7c6af7 0%, #c084fc 40%, transparent 100%);
    margin: 16px 0 32px;
}

/* ── Upload zone ── */
.upload-label {
    font-size: 0.75rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #6b6b80;
    margin-bottom: 8px;
    font-weight: 600;
}

/* ── Emotion badge ── */
.emotion-badge {
    display: inline-flex;
    align-items: center;
    gap: 10px;
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    border: 1px solid #7c6af7;
    border-radius: 12px;
    padding: 16px 24px;
    margin: 8px 0 20px;
}
.emotion-emoji { font-size: 2.2rem; line-height: 1; }
.emotion-text { }
.emotion-name {
    font-family: 'Space Mono', monospace;
    font-size: 1.5rem;
    font-weight: 700;
    color: #fff;
    letter-spacing: -0.5px;
}
.emotion-conf {
    font-size: 0.8rem;
    color: #a78bfa;
    font-weight: 500;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}

/* ── Section label ── */
.section-label {
    font-size: 0.7rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #6b6b80;
    font-weight: 600;
    margin-bottom: 10px;
    border-left: 2px solid #7c6af7;
    padding-left: 8px;
}

/* ── Image panels ── */
.img-panel {
    background: #111118;
    border: 1px solid #1e1e2e;
    border-radius: 10px;
    overflow: hidden;
}

/* ── Metric card ── */
.metric-row {
    display: flex;
    gap: 12px;
    margin-bottom: 24px;
}
.metric-card {
    flex: 1;
    background: #111118;
    border: 1px solid #1e1e2e;
    border-radius: 10px;
    padding: 16px 20px;
}
.metric-val {
    font-family: 'Space Mono', monospace;
    font-size: 1.6rem;
    font-weight: 700;
    color: #fff;
    line-height: 1;
    margin-bottom: 4px;
}
.metric-key {
    font-size: 0.72rem;
    color: #6b6b80;
    text-transform: uppercase;
    letter-spacing: 0.1em;
}

/* ── Footer ── */
.ml-footer {
    margin-top: 48px;
    padding-top: 20px;
    border-top: 1px solid #1e1e2e;
    font-size: 0.75rem;
    color: #3a3a4a;
    text-align: center;
    font-family: 'Space Mono', monospace;
}

/* ── Streamlit overrides ── */
.stFileUploader > div {
    background: #111118 !important;
    border: 1px dashed #2e2e42 !important;
    border-radius: 10px !important;
}
.stFileUploader > div:hover {
    border-color: #7c6af7 !important;
}
div[data-testid="stImage"] img {
    border-radius: 8px;
}
</style>
""", unsafe_allow_html=True)


# ── Model loading ─────────────────────────────────────────────────────────────

@st.cache_resource
def load_pipeline():
    from inference import MoodLensPipeline
    ckpt = Path("checkpoints/best_model.pth")
    if not ckpt.exists():
        return None
    return MoodLensPipeline(str(ckpt), device="cpu")


# ── Helpers ───────────────────────────────────────────────────────────────────

EMOTION_EMOJI = {
    "Angry": "😠", "Disgust": "🤢", "Fear": "😨",
    "Happy": "😊", "Neutral": "😐", "Sad": "😢", "Surprise": "😲"
}

EMOTION_COLOR = {
    "Angry": "#ef4444", "Disgust": "#84cc16", "Fear": "#8b5cf6",
    "Happy": "#f59e0b", "Neutral": "#6b7280", "Sad": "#3b82f6", "Surprise": "#ec4899"
}

def prob_chart(probs: dict, top_emotion: str):
    emotions = list(probs.keys())
    values = [probs[e] for e in emotions]
    colors = [EMOTION_COLOR.get(e, "#7c6af7") for e in emotions]
    bar_colors = [c if e == top_emotion else "#1e1e2e" for e, c in zip(emotions, colors)]
    text_colors = [c if e == top_emotion else "#4a4a5a" for e, c in zip(emotions, colors)]

    fig = go.Figure(go.Bar(
        x=values,
        y=emotions,
        orientation="h",
        marker=dict(
            color=bar_colors,
            line=dict(color=[c for c in colors], width=1.5),
        ),
        text=[f"{v:.1f}%" for v in values],
        textposition="outside",
        textfont=dict(
            family="Space Mono, monospace",
            size=11,
            color=text_colors,
        ),
    ))

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=60, t=0, b=0),
        height=240,
        xaxis=dict(
            showgrid=False, showticklabels=False,
            range=[0, max(values) * 1.25],
            zeroline=False,
        ),
        yaxis=dict(
            showgrid=False,
            tickfont=dict(family="Space Grotesk", size=13, color="#9090a8"),
            autorange="reversed",
        ),
        showlegend=False,
    )
    return fig


def training_curves_chart(log_path="logs/training_log.json"):
    if not Path(log_path).exists():
        return None
    with open(log_path) as f:
        log = json.load(f)
    epochs = list(range(1, len(log["train_acc"]) + 1))

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=epochs, y=log["train_acc"], name="Train",
        line=dict(color="#7c6af7", width=2),
        mode="lines",
    ))
    fig.add_trace(go.Scatter(
        x=epochs, y=log["val_acc"], name="Val",
        line=dict(color="#c084fc", width=2, dash="dot"),
        mode="lines",
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=0, t=8, b=0),
        height=180,
        xaxis=dict(
            showgrid=True, gridcolor="#1a1a2a", tickfont=dict(color="#6b6b80", size=10),
            title=dict(text="Epoch", font=dict(color="#6b6b80", size=10)),
        ),
        yaxis=dict(
            showgrid=True, gridcolor="#1a1a2a", tickfont=dict(color="#6b6b80", size=10),
            title=dict(text="Accuracy %", font=dict(color="#6b6b80", size=10)),
        ),
        legend=dict(font=dict(color="#9090a8", size=10), bgcolor="rgba(0,0,0,0)"),
    )
    return fig


# ── UI ────────────────────────────────────────────────────────────────────────

# Header
st.markdown("""
<div class="ml-header">
    <div class="ml-logo">Mood<span>Lens</span></div>
    <div class="ml-tagline">CNN-based face emotion recognition</div>
</div>
<div class="ml-divider"></div>
""", unsafe_allow_html=True)

# Load model
pipeline = load_pipeline()

if pipeline is None:
    st.error("⚠️ Checkpoint not found at `checkpoints/best_model.pth`. Make sure the file exists.")
    st.stop()

# ── Upload ────────────────────────────────────────────────────────────────────

st.markdown('<div class="upload-label">Upload a photo</div>', unsafe_allow_html=True)
uploaded = st.file_uploader(
    "Drop an image here",
    type=["jpg", "jpeg", "png", "webp"],
    label_visibility="collapsed",
)

if uploaded is None:
    # Landing state
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-val">7</div>
            <div class="metric-key">Emotion classes</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-val">63.9%</div>
            <div class="metric-key">Validation accuracy</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-val">28.7k</div>
            <div class="metric-key">Training images</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Training curves if log exists
    fig = training_curves_chart()
    if fig:
        st.markdown('<div class="section-label">Training history</div>', unsafe_allow_html=True)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    st.markdown("""
    <div class="ml-footer">
        Built from scratch on FER-2013 · PyTorch · OpenCV · Grad-CAM · MoodLens v1.0
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ── Run inference ─────────────────────────────────────────────────────────────

image = Image.open(uploaded).convert("RGB")

with st.spinner("Detecting faces and reading emotions..."):
    result = pipeline.run(image)

if result.num_faces == 0:
    st.warning("No faces detected. Try a clearer photo with a visible face.")
    st.stop()

# ── Results layout ────────────────────────────────────────────────────────────

# Face selector if multiple faces
face_idx = 0
if result.num_faces > 1:
    st.markdown(f"**{result.num_faces} faces detected** — select one to inspect:")
    tabs = st.tabs([
        f"{EMOTION_EMOJI.get(f.emotion, '🙂')} Face {i+1}"
        for i, f in enumerate(result.faces)
    ])
else:
    tabs = [st.container()]

for tab_idx, tab in enumerate(tabs):
    with tab:
        face = result.faces[tab_idx]
        emoji = EMOTION_EMOJI.get(face.emotion, "🙂")
        color = EMOTION_COLOR.get(face.emotion, "#7c6af7")

        # Emotion badge
        st.markdown(f"""
        <div class="emotion-badge">
            <div class="emotion-emoji">{emoji}</div>
            <div class="emotion-text">
                <div class="emotion-name">{face.emotion}</div>
                <div class="emotion-conf">{face.confidence:.1f}% confidence</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Three columns: original annotated | face crop | Grad-CAM
        col1, col2, col3 = st.columns([2, 1, 1])

        with col1:
            st.markdown('<div class="section-label">Detected face</div>', unsafe_allow_html=True)
            st.image(result.annotated, use_container_width=True)

        with col2:
            st.markdown('<div class="section-label">Face crop</div>', unsafe_allow_html=True)
            st.image(face.face_crop, use_container_width=True)

        with col3:
            st.markdown('<div class="section-label">Grad-CAM · what the CNN saw</div>', unsafe_allow_html=True)
            st.image(face.overlay, use_container_width=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Confidence chart
        st.markdown('<div class="section-label">Confidence across all emotions</div>', unsafe_allow_html=True)
        fig = prob_chart(face.probabilities, face.emotion)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        # Per-class breakdown (expandable)
        with st.expander("Raw probabilities"):
            for em, prob in sorted(face.probabilities.items(), key=lambda x: -x[1]):
                em_emoji = EMOTION_EMOJI.get(em, "")
                st.markdown(
                    f"`{em_emoji} {em:<10}` **{prob:.2f}%**"
                )

# Training curves sidebar
with st.sidebar:
    st.markdown("### Training history")
    fig = training_curves_chart()
    if fig:
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        st.caption("63.92% val accuracy · 50 epochs · FER-2013")
    st.markdown("---")
    st.markdown("**Model**")
    st.caption("EmotionCNN · 10.8M params · trained from scratch")
    st.markdown("**Stack**")
    st.caption("PyTorch · OpenCV Haar Cascade · Grad-CAM · Streamlit")

# Footer
st.markdown("""
<div class="ml-footer">
    Built from scratch on FER-2013 · PyTorch · OpenCV · Grad-CAM · MoodLens v1.0
</div>
""", unsafe_allow_html=True)
