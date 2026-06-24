"""
MoodLens - Face Detector
Uses OpenCV Haar Cascade to detect and crop faces from any image.
"""

import cv2
import numpy as np
from PIL import Image
from pathlib import Path
import urllib.request
import os


# ── Download cascade if not present ──────────────────────────────────────────

CASCADE_PATH = Path(__file__).parent / "haarcascade_frontalface_default.xml"
CASCADE_URL = (
    "https://raw.githubusercontent.com/opencv/opencv/master/"
    "data/haarcascades/haarcascade_frontalface_default.xml"
)

def ensure_cascade():
    if not CASCADE_PATH.exists():
        print("Downloading Haar cascade...")
        urllib.request.urlretrieve(CASCADE_URL, CASCADE_PATH)
        print("Done.")

ensure_cascade()


# ── Face Detector ─────────────────────────────────────────────────────────────

class FaceDetector:
    def __init__(self, scale_factor=1.1, min_neighbors=5, min_size=(30, 30)):
        self.detector = cv2.CascadeClassifier(str(CASCADE_PATH))
        self.scale_factor = scale_factor
        self.min_neighbors = min_neighbors
        self.min_size = min_size

    def detect(self, image: np.ndarray):
        """
        Args:
            image: BGR numpy array (from cv2.imread) or RGB numpy array
        Returns:
            list of (x, y, w, h) bounding boxes, sorted by area descending
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        faces = self.detector.detectMultiScale(
            gray,
            scaleFactor=self.scale_factor,
            minNeighbors=self.min_neighbors,
            minSize=self.min_size,
        )
        if len(faces) == 0:
            return []
        # Sort by area (largest face first)
        return sorted(faces, key=lambda f: f[2] * f[3], reverse=True)

    def crop_face(self, image: np.ndarray, box, padding: float = 0.2):
        """
        Crops a single face with optional padding.
        Args:
            image: RGB numpy array
            box:   (x, y, w, h)
            padding: fraction of face size to pad on each side
        Returns:
            PIL Image of the cropped face (RGB)
        """
        x, y, w, h = box
        pad_x = int(w * padding)
        pad_y = int(h * padding)
        ih, iw = image.shape[:2]

        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(iw, x + w + pad_x)
        y2 = min(ih, y + h + pad_y)

        crop = image[y1:y2, x1:x2]
        return Image.fromarray(crop)

    def detect_and_crop_all(self, pil_image: Image.Image, padding: float = 0.2):
        """
        Full pipeline: PIL image → list of cropped face PIL images + boxes.
        Returns:
            faces: list of PIL Images (one per detected face)
            boxes: list of (x, y, w, h)
        """
        rgb = np.array(pil_image.convert("RGB"))
        boxes = self.detect(rgb)

        if not boxes:
            # Fallback: use the whole image as the face
            return [pil_image], [None]

        faces = [self.crop_face(rgb, box, padding) for box in boxes]
        return faces, list(boxes)

    def draw_boxes(self, pil_image: Image.Image, boxes, predictions=None):
        """
        Draw bounding boxes on the image.
        predictions: optional list of (emotion_str, confidence) per box
        Returns: PIL Image with boxes drawn
        """
        img = np.array(pil_image.convert("RGB")).copy()
        for i, box in enumerate(boxes):
            if box is None:
                continue
            x, y, w, h = box
            cv2.rectangle(img, (x, y), (x + w, y + h), (99, 102, 241), 2)

            if predictions and i < len(predictions):
                emotion, conf = predictions[i]
                label = f"{emotion} {conf:.0f}%"
                cv2.putText(
                    img, label,
                    (x, max(y - 8, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (99, 102, 241), 2,
                )
        return Image.fromarray(img)


# ── Quick test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else None
    if not path:
        print("Usage: python face_detector.py <image_path>")
        exit()

    img = Image.open(path)
    detector = FaceDetector()
    faces, boxes = detector.detect_and_crop_all(img)
    print(f"Detected {len(faces)} face(s)")
    for i, (face, box) in enumerate(zip(faces, boxes)):
        out = f"face_{i}.jpg"
        face.save(out)
        print(f"  Face {i}: box={box} → saved {out}")
