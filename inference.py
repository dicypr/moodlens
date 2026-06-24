"""
MoodLens - Inference Pipeline

Single entry point that ties everything together:
  PIL Image → detect faces → predict emotion → generate Grad-CAM → return results

Used by the Streamlit app (Day 3) and can be tested standalone.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image

from model import EmotionCNN
from dataset import get_val_transform
from face_detector import FaceDetector
from gradcam import GradCAM


# ── Result dataclass ───────────────────────────────────────────────────────────

@dataclass
class FaceResult:
    """Result for a single detected face."""
    face_crop: Image.Image          # cropped face PIL image
    emotion: str                    # top predicted emotion
    confidence: float               # confidence % for top emotion
    probabilities: dict             # {emotion: float %} for all 7
    heatmap: np.ndarray             # Grad-CAM heatmap (48x48, float [0,1])
    overlay: Image.Image            # face_crop + heatmap blended
    box: Optional[tuple] = None     # (x, y, w, h) in original image, or None


@dataclass
class InferenceResult:
    """Full result for one image (may have multiple faces)."""
    original: Image.Image
    annotated: Image.Image          # original with bounding boxes drawn
    faces: list[FaceResult] = field(default_factory=list)
    num_faces: int = 0


# ── Pipeline ───────────────────────────────────────────────────────────────────

class MoodLensPipeline:
    EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]
    EMOTION_EMOJI = {
        "Angry": "😠", "Disgust": "🤢", "Fear": "😨",
        "Happy": "😊", "Neutral": "😐", "Sad": "😢", "Surprise": "😲"
    }

    def __init__(self, checkpoint_path: str, device: str = None):
        if device is None:
            device = (
                "cuda" if torch.cuda.is_available()
                else "mps" if torch.backends.mps.is_available()
                else "cpu"
            )
        self.device = torch.device(device)
        print(f"[MoodLens] Loading model on {self.device}...")

        # Model
        self.model = EmotionCNN(num_classes=7)
        ckpt = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state"])
        self.model.to(self.device)
        self.model.eval()
        print(f"[MoodLens] Model loaded (val_acc={ckpt.get('val_acc', '?'):.2f}%)")

        # Components
        self.transform = get_val_transform()
        self.detector = FaceDetector()
        self.gradcam = GradCAM(self.model)

    def _preprocess(self, face_pil: Image.Image) -> torch.Tensor:
        """PIL face crop → (1,1,48,48) tensor on device."""
        return self.transform(face_pil).unsqueeze(0).to(self.device)

    def predict_face(self, face_pil: Image.Image, box=None) -> FaceResult:
        """Run full prediction + Grad-CAM on a single face crop."""
        tensor = self._preprocess(face_pil)

        # Grad-CAM (includes forward + backward pass)
        heatmap, pred_idx, probs = self.gradcam.generate(tensor)
        emotion = self.EMOTIONS[pred_idx]
        confidence = probs[emotion]
        overlay = self.gradcam.overlay(face_pil, heatmap, alpha=0.45)

        return FaceResult(
            face_crop=face_pil,
            emotion=emotion,
            confidence=confidence,
            probabilities=probs,
            heatmap=heatmap,
            overlay=overlay,
            box=box,
        )

    def run(self, image: Image.Image, max_faces: int = 5) -> InferenceResult:
        """
        Full pipeline: image → detect faces → predict → Grad-CAM.

        Args:
            image:     PIL Image (any size, any mode)
            max_faces: max number of faces to process

        Returns:
            InferenceResult with all face predictions
        """
        image = image.convert("RGB")

        # Detect faces
        face_crops, boxes = self.detector.detect_and_crop_all(image)
        face_crops = face_crops[:max_faces]
        boxes = boxes[:max_faces]

        # Predict each face
        face_results = []
        predictions_for_drawing = []

        for crop, box in zip(face_crops, boxes):
            result = self.predict_face(crop, box)
            face_results.append(result)
            predictions_for_drawing.append((result.emotion, result.confidence))

        # Draw bounding boxes on original
        valid_boxes = [b for b in boxes if b is not None]
        if valid_boxes:
            annotated = self.detector.draw_boxes(image, valid_boxes, predictions_for_drawing)
        else:
            annotated = image.copy()

        return InferenceResult(
            original=image,
            annotated=annotated,
            faces=face_results,
            num_faces=len(face_results),
        )

    def run_from_path(self, image_path: str) -> InferenceResult:
        return self.run(Image.open(image_path))


# ── CLI test ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python inference.py <checkpoint.pth> <image.jpg>")
        exit()

    ckpt, img_path = sys.argv[1], sys.argv[2]
    pipeline = MoodLensPipeline(ckpt)
    result = pipeline.run_from_path(img_path)

    print(f"\nDetected {result.num_faces} face(s):")
    for i, face in enumerate(result.faces):
        emoji = MoodLensPipeline.EMOTION_EMOJI[face.emotion]
        print(f"\n  Face {i+1}: {emoji} {face.emotion} ({face.confidence:.1f}%)")
        print("  All probabilities:")
        for emotion, prob in sorted(face.probabilities.items(), key=lambda x: -x[1]):
            bar = "█" * int(prob / 3)
            print(f"    {emotion:<10} {prob:5.1f}%  {bar}")

        # Save outputs
        face.face_crop.save(f"output_face_{i}_crop.jpg")
        face.overlay.save(f"output_face_{i}_gradcam.jpg")
        print(f"  Saved: output_face_{i}_crop.jpg + output_face_{i}_gradcam.jpg")

    result.annotated.save("output_annotated.jpg")
    print(f"\nAnnotated image saved: output_annotated.jpg")
