"""
hand_tracker.py — Hand Detection & Landmark Extraction via MediaPipe Tasks API.

Responsible for:
  • Detecting hands in each frame using MediaPipe HandLandmarker.
  • Extracting a square bounding box around the detected hand.

Note: All skeleton/keypoint drawing has been intentionally removed.
The CNN model must receive clean, overlay-free frames for accurate inference.
"""

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision





class HandTracker:
    """Wraps MediaPipe HandLandmarker for single-hand detection & tracking."""

    def __init__(self, model_asset_path: str = "hand_landmarker.task",
                 min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5):
        """
        Args:
            model_asset_path: Path to the MediaPipe HandLandmarker .task model.
            min_detection_confidence: Minimum confidence for initial detection.
            min_tracking_confidence: Minimum confidence for frame-to-frame tracking.
        """
        base_options = mp_python.BaseOptions(model_asset_path=model_asset_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=1,
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self.detector = vision.HandLandmarker.create_from_options(options)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(self, bgr_frame: np.ndarray):
        """
        Detect hand landmarks in a BGR frame.

        Returns:
            landmarks (list | None): List of MediaPipe NormalizedLandmark objects,
                                     or None if no hand was found.
        """
        rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        result = self.detector.detect(mp_image)

        if result.hand_landmarks:
            return result.hand_landmarks[0]
        return None

    def get_bounding_box(self, landmarks, frame_shape, padding: int = 80):
        """
        Compute a square bounding box around the detected hand.

        The box is padded and clamped to frame edges to avoid out-of-bounds
        access, then forced to be square so the CNN doesn't receive distorted
        aspect ratios.

        Args:
            landmarks: List of NormalizedLandmark from MediaPipe.
            frame_shape: (height, width, channels) of the frame.
            padding: Extra pixels to add around the tight bounding box.

        Returns:
            (x_min, y_min, x_max, y_max) — integer pixel coordinates.
        """
        h, w = frame_shape[:2]

        # Convert normalised landmarks to pixel coordinates
        xs = [int(lm.x * w) for lm in landmarks]
        ys = [int(lm.y * h) for lm in landmarks]

        center_x = (min(xs) + max(xs)) // 2
        center_y = (min(ys) + max(ys)) // 2

        # Take the larger dimension so the box is square, then add padding
        box_size = max(max(xs) - min(xs), max(ys) - min(ys)) + padding
        half = box_size // 2

        x_min = max(0, center_x - half)
        y_min = max(0, center_y - half)
        x_max = min(w, center_x + half)
        y_max = min(h, center_y + half)

        # Enforce strict square after clamping
        crop_size = min(x_max - x_min, y_max - y_min)
        x_max = x_min + crop_size
        y_max = y_min + crop_size

        return x_min, y_min, x_max, y_max


    def close(self):
        """Release MediaPipe detector resources."""
        if self.detector:
            self.detector.close()
