"""
================================================================================
FACIAL EMOTION RECOGNITION â€” MODEL ARCHITECTURE & INFERENCE LAYER
Author: Corey Leath (Trojan3877)

Purpose:
    Defines model architectures and provides a production-safe inference wrapper.

Architectures:
    - EmotionCNN: Lightweight CNN for fast inference.
    - ResNetEmotion: Transfer-learning backbone for higher accuracy.

Design Decisions:
    - Two architecture options for deployment flexibility.
    - Softmax applied at inference time only (not inside forward pass).
    - Wrapper enforces eval() mode and device placement.

Tradeoffs:
    - EmotionCNN is lightweight but lower capacity.
    - ResNetEmotion improves accuracy but increases latency.
    - No preprocessing inside model (kept separate for clarity).

Future Improvements:
    - Add ONNX export support.
    - Add quantization support.
    - Add batched inference optimization.
================================================================================
"""

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from src.src.config.settings import settings


# ==============================================================================
# LOGGING
# ==============================================================================

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


# ==============================================================================
# CONSTANTS
# ==============================================================================

EMOTION_LABELS = {
    0: "Angry",
    1: "Disgust",
    2: "Fear",
    3: "Happy",
    4: "Sad",
    5: "Surprise",
    6: "Neutral"
}


# ==============================================================================
# LIGHTWEIGHT CNN
# ==============================================================================

class EmotionCNN(nn.Module):
    """
    Lightweight CNN for low-latency deployment.

    Input:  (B, 1, 48, 48)
    Output: (B, 7) logits
    """

    def __init__(self, num_classes=7):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(64 * 12 * 12, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        """
        Forward pass.

        Complexity:
            O(n * k^2 * c) per conv layer.

        Design Note:
            No softmax here â€” applied during inference only.
        """
        x = self.pool(F.relu(self.conv1(x)))  # (B,1,48,48) â†’ (B,32,24,24)
        x = self.pool(F.relu(self.conv2(x)))  # â†’ (B,64,12,12)
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# ==============================================================================
# RESNET18 BACKBONE
# ==============================================================================

class ResNetEmotion(nn.Module):
    """
    Transfer-learning backbone using ResNet18.

    Input:  (B, 1, 224, 224)
    Output: (B, 7) logits
    """

    def __init__(self, num_classes=7):
        super().__init__()

        self.model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        # Modify input layer to accept grayscale
        self.model.conv1 = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False
        )

        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)


# ==============================================================================
# INFERENCE WRAPPER
# ==============================================================================

class EmotionModel:
    """
    Production-safe inference wrapper.

    Responsibilities:
        - Load model weights
        - Manage device placement
        - Enforce eval() mode
        - Return structured prediction output
    """

    def __init__(self, model_path=settings.MODEL_PATH, use_resnet=False):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_resnet = use_resnet

        logger.info(f"Using device: {self.device}")

        # Select architecture
        if self.use_resnet:
            self.model = ResNetEmotion()
        else:
            self.model = EmotionCNN()

        self._load_weights(model_path)

    def _load_weights(self, model_path: str):
        """
        Safely loads model weights.
        Raises explicit error if file missing.
        """
        if not torch.cuda.is_available():
            logger.info("Running on CPU.")

        try:
            state_dict = torch.load(model_path, map_location=self.device, weights_only=True)
            self.model.load_state_dict(state_dict)
        except Exception as e:
            raise RuntimeError(f"Failed to load model weights: {e}")

        self.model.to(self.device)
        self.model.eval()

        logger.info("Model loaded successfully.")

    def predict(self, face_tensor):
        """
        Args:
            face_tensor: numpy array or tensor
                Shape: (1, 1, 48, 48) or (B, 1, H, W)

        Returns:
            dict:
                {
                    "emotion": str,
                    "confidence": float
                }
        """

        if not isinstance(face_tensor, torch.Tensor):
            face_tensor = torch.tensor(face_tensor, dtype=torch.float32)

        if face_tensor.dim() != 4:
            raise ValueError("Input tensor must be 4D (B, C, H, W)")

        face_tensor = face_tensor.to(self.device)

        with torch.no_grad():
            logits = self.model(face_tensor)
            probabilities = F.softmax(logits, dim=1)

        predicted_idx = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0, predicted_idx].item()

        return {
            "emotion": EMOTION_LABELS[predicted_idx],
            "confidence": round(float(confidence), 4)
        }
