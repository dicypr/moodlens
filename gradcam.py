"""
MoodLens - Grad-CAM
Generates heatmaps showing WHICH part of the face the CNN focused on.

Grad-CAM: Gradient-weighted Class Activation Mapping
Paper: https://arxiv.org/abs/1610.02391

How it works:
  1. Forward pass → get prediction
  2. Backprop gradient of predicted class score w.r.t. last conv layer activations
  3. Global-average-pool the gradients → per-channel importance weights
  4. Weighted sum of activation maps → heatmap
  5. Upsample to input size + overlay on original image
"""

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import cv2


class GradCAM:
    def __init__(self, model, target_layer=None):
        """
        Args:
            model:        EmotionCNN instance
            target_layer: the conv layer to hook (default: last conv in block3)
        """
        self.model = model
        self.model.eval()

        self._activations = None
        self._gradients = None

        # Default: last conv layer in block3 (index 3 = second Conv2d)
        if target_layer is None:
            target_layer = model.block3[3]

        # Forward hook — capture activations
        target_layer.register_forward_hook(self._save_activation)
        # Backward hook — capture gradients
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self._activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self._gradients = grad_output[0].detach()

    def generate(self, input_tensor: torch.Tensor, class_idx: int = None):
        """
        Args:
            input_tensor: (1, 1, 48, 48) preprocessed image tensor
            class_idx:    which class to explain (None = predicted class)
        Returns:
            heatmap: numpy array (48, 48) in [0, 1]
            pred_idx: predicted class index
            probs: dict {emotion: probability}
        """
        self.model.zero_grad()

        # Forward
        logits = self.model(input_tensor)
        probs = F.softmax(logits, dim=1)

        if class_idx is None:
            class_idx = logits.argmax(dim=1).item()

        # Backprop for the target class
        score = logits[0, class_idx]
        score.backward()

        # Grad-CAM computation
        gradients = self._gradients[0]    # (C, H, W)
        activations = self._activations[0] # (C, H, W)

        # Global average pool gradients → weights
        weights = gradients.mean(dim=(1, 2))  # (C,)

        # Weighted combination of activation maps
        cam = torch.zeros(activations.shape[1:], dtype=torch.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i]

        # ReLU (only positive influence matters)
        cam = F.relu(cam)

        # Normalize to [0, 1]
        cam = cam.numpy()
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()

        # Upsample to 48×48
        cam = cv2.resize(cam, (48, 48))

        return cam, class_idx, {
            emotion: round(probs[0][i].item() * 100, 2)
            for i, emotion in enumerate(["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"])
        }

    def overlay(self, original_pil: Image.Image, heatmap: np.ndarray, alpha: float = 0.45):
        """
        Overlays the Grad-CAM heatmap on the original face image.
        Args:
            original_pil: PIL Image (the face crop, any size)
            heatmap:      numpy (48,48) in [0,1]
            alpha:        blend strength of heatmap
        Returns:
            PIL Image with heatmap overlay
        """
        # Resize heatmap to match original image
        w, h = original_pil.size
        heatmap_resized = cv2.resize(heatmap, (w, h))

        # Convert to color heatmap (JET colormap)
        heatmap_uint8 = np.uint8(255 * heatmap_resized)
        heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        heatmap_rgb = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

        # Blend with original
        original_np = np.array(original_pil.convert("RGB")).astype(np.float32)
        heatmap_np = heatmap_rgb.astype(np.float32)

        blended = (1 - alpha) * original_np + alpha * heatmap_np
        blended = np.clip(blended, 0, 255).astype(np.uint8)

        return Image.fromarray(blended)


# ── Quick test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    from model import EmotionCNN
    from dataset import get_val_transform

    if len(sys.argv) < 3:
        print("Usage: python gradcam.py <checkpoint.pth> <face_image.jpg>")
        exit()

    ckpt_path, img_path = sys.argv[1], sys.argv[2]

    # Load model
    model = EmotionCNN()
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model_state"])

    # Load image
    pil = Image.open(img_path).convert("RGB")
    transform = get_val_transform()
    tensor = transform(pil).unsqueeze(0)

    # Grad-CAM
    cam_gen = GradCAM(model)
    heatmap, pred_idx, probs = cam_gen.generate(tensor)
    overlay = cam_gen.overlay(pil, heatmap)

    overlay.save("gradcam_output.jpg")
    emotion = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"][pred_idx]
    print(f"Prediction: {emotion}")
    print(f"Saved: gradcam_output.jpg")
