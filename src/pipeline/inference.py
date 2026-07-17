import logging
from typing import Optional

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

logger = logging.getLogger(__name__)


class EmotionPipeline:
    """
    Handles preprocessing + model inference for the CNN emotion classifier.
    """

    def __init__(self, model_path: str = "models/emotion_cnn.pth"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._load_error: Optional[str] = None
        try:
            self.model = self._load_model(model_path)
            self.model.to(self.device)
            self.model.eval()
        except Exception as e:
            self.model = None
            self._load_error = str(e)
            logger.warning(f"EmotionPipeline: model not loaded â€” {e}")

        # Standard CNN preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((48, 48)),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

        # Emotion classes â€” update to match your model
        self.labels = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]

    def _load_model(self, path):
        """
        Loads trained PyTorch model.
        """
        try:
            model = torch.load(path, map_location=self.device, weights_only=True)
            return model
        except Exception as e:
            raise RuntimeError(f"Error loading model: {str(e)}")

    def preprocess(self, image: Image.Image):
        """
        Apply all transforms to prepare the image for the CNN.
        """
        return self.transform(image).unsqueeze(0).to(self.device)

    def predict_image(self, image: Image.Image) -> list[str]:
        """
        Full inference pipeline:
        1. Load image
        2. Preprocess
        3. Forward pass through CNN
        4. Convert logits â†’ probabilities
        5. Return top predicted labels
        """
        if self.model is None:
            raise RuntimeError(
                f"Model weights not loaded. {self._load_error}"
            )

        # Preprocess
        inputs = self.preprocess(image)

        # Model inference
        with torch.no_grad():
            logits = self.model(inputs)
            probs = F.softmax(logits, dim=1).cpu().numpy()[0]

        # Sort probabilities high â†’ low
        ranked = sorted(
            list(zip(self.labels, probs)),
            key=lambda x: x[1],
            reverse=True
        )

        # Return only labels (top 1â€“2 predictions)
        top_predictions = [ranked[0][0]]
        if ranked[1][1] > 0.20:  # include second emotion if confident
            top_predictions.append(ranked[1][0])

        return top_predictions

    def predict(self, image_file) -> list[str]:
        """Backward-compatible adapter for file-like callers."""
        image = Image.open(image_file.file).convert("RGB")
        return self.predict_image(image)
