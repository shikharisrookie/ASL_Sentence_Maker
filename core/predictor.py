"""
predictor.py — Sign Language Prediction Engine.

Responsible for:
  • Loading the trained Keras CNN model.
  - Preprocessing hand crops (resize -> normalise -> batch dimension).
  • Running inference and returning (class_name, confidence).
"""

import cv2
import numpy as np
from tensorflow.keras.models import load_model


# Default 29-class ASL label map: A-Z (0–25), del (26), nothing (27), space (28)
DEFAULT_LABELS = {i: chr(65 + i) for i in range(26)}
DEFAULT_LABELS.update({26: "del", 27: "nothing", 28: "space"})


class SignPredictor:
    """Encapsulates CNN model loading and single-frame inference."""

    def __init__(self, model_path: str = "SLR_final.h5",
                 labels_map: dict = None,
                 img_size: int = 64):
        """
        Args:
            model_path: Path to the Keras .h5 model file.
            labels_map: Dict mapping class index -> label string.
                        Defaults to A-Z + del/nothing/space (29 classes).
            img_size: Input resolution the CNN expects (square).
        """
        self.img_size = img_size
        self.labels_map = labels_map or DEFAULT_LABELS
        self.model = None

        try:
            print(f"[SignPredictor] Loading model from '{model_path}'...")
            self.model = load_model(model_path)
            print("[SignPredictor] Model loaded successfully.")
        except Exception as e:
            print(f"[SignPredictor] ERROR loading model: {e}")
            print("[SignPredictor] WARNING: Running in NO-MODEL mode (no predictions).")

    @property
    def is_ready(self) -> bool:
        """True if the model was loaded successfully."""
        return self.model is not None

    def preprocess(self, bgr_frame: np.ndarray, bbox: tuple) -> np.ndarray | None:
        """
        Crop, resize, and normalise the hand region for inference.

        Args:
            bgr_frame: Full BGR webcam frame.
            bbox: (x_min, y_min, x_max, y_max) bounding box around the hand.

        Returns:
            Preprocessed tensor of shape (1, img_size, img_size, 3), or None
            if the crop is too small / empty.
        """
        x_min, y_min, x_max, y_max = bbox

        # Reject crops that are too small to be meaningful
        if (x_max - x_min) < 20 or (y_max - y_min) < 20:
            return None

        roi = bgr_frame[y_min:y_max, x_min:x_max]
        if roi.size == 0:
            return None

        # Resize first in BGR (matches training pipeline order), then convert to RGB
        roi_resized = cv2.resize(roi, (self.img_size, self.img_size))
        roi_rgb = cv2.cvtColor(roi_resized, cv2.COLOR_BGR2RGB)

        # Normalise pixel values to [0, 1]
        roi_normalised = roi_rgb.astype("float32") / 255.0

        # Add batch dimension -> (1, 64, 64, 3)
        return np.expand_dims(roi_normalised, axis=0)

    def predict(self, preprocessed_input: np.ndarray) -> tuple[str, float]:
        """
        Run inference on a preprocessed input tensor.

        Returns:
            (class_name, confidence) — e.g. ("A", 0.94)
        """
        if self.model is None or preprocessed_input is None:
            return "nothing", 0.0

        prediction = self.model.predict(preprocessed_input, verbose=0)
        class_id = int(np.argmax(prediction, axis=1)[0])
        confidence = float(np.max(prediction))
        class_name = self.labels_map.get(class_id, "Unknown")

        return class_name, confidence
