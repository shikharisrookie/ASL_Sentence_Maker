"""
recognizer.py — Main Orchestrator for Real-Time ASL Recognition.

Responsible for:
  • Connecting the webcam capture loop to all subsystems.
  - Coordinating HandTracker -> SignPredictor -> PredictionStabiliser -> HUD -> TTS.
  • Handling keyboard input for manual overrides.
"""

import cv2
import time
import numpy as np

from core.hand_tracker import HandTracker
from core.predictor import SignPredictor
from core.stabiliser import PredictionStabiliser
from core.hud import HUD
from core.tts_engine import TTSEngine


class ASLRecognizer:
    """
    Top-level controller that runs the webcam loop and delegates work
    to specialised subsystem modules.

    Architecture:
        ┌──────────┐     ┌──────────────┐     ┌──────────────┐
        │ Webcam   │ ──► │ HandTracker  │ ──► │ SignPredictor │
        └──────────┘     └──────────────┘     └──────┬───────┘
                                                     │
                                                     ▼
        ┌──────────┐     ┌──────────────┐     ┌──────────────┐
        │   HUD    │ ◄── │ Stabiliser   │ ◄── │  (sign, conf)│
        └──────────┘     └──────────────┘     └──────────────┘
                                │
                                ▼
                          ┌──────────┐
                          │   TTS    │
                          └──────────┘
    """

    def __init__(self, *,
                 model_path: str = "SLR_final.h5",
                 confidence_threshold: float = 0.65,
                 frame_threshold: int = 15,
                 cooldown_seconds: float = 1.2,
                 enable_tts: bool = False):
        """
        Initialise all subsystems.

        Args:
            model_path: Path to the trained Keras .h5 model.
            confidence_threshold: Min confidence to accept a prediction.
            frame_threshold: Consecutive identical frames required.
            cooldown_seconds: Minimum gap between repeated same-letter appends.
            enable_tts: Whether to enable text-to-speech output.
        """
        # ── Subsystem initialisation ──
        self.tracker = HandTracker()
        self.predictor = SignPredictor(model_path=model_path)
        self.stabiliser = PredictionStabiliser(
            confidence_threshold=confidence_threshold,
            frame_threshold=frame_threshold,
            cooldown_seconds=cooldown_seconds,
        )
        self.tts = TTSEngine(enabled=enable_tts)

        # ── Per-frame state (for HUD rendering) ──
        self._current_sign = "nothing"
        self._current_confidence = 0.0

        # ── FPS tracking ──
        self._fps_time = time.time()
        self._fps = 0.0

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self):
        """
        Open the webcam and enter the real-time recognition loop.

        Keyboard controls:
            s         -> Insert space
            Backspace -> Delete last character
            c         -> Clear sentence
            t         -> Speak the current sentence (TTS)
            q / ESC   -> Quit
        """
        print("=" * 60)
        print("  ASL Real-Time Recogniser")
        print("=" * 60)
        print("Controls:")
        print("  s         -> Space")
        print("  Backspace -> Delete last char")
        print("  c         -> Clear sentence")
        print("  t         -> Speak sentence (TTS)")
        print("  q / ESC   -> Quit")
        print("=" * 60)

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("[ERROR] Could not open webcam (index 0).")
            return

        # Attempt to set a higher resolution for better hand detection
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    print("[WARN] Failed to grab frame. Retrying...")
                    continue

                # Mirror the frame so it feels like a mirror for the user
                frame = cv2.flip(frame, 1)

                # ══ CRITICAL ══════════════════════════════════════════
                # Save a clean copy BEFORE any drawing is done on frame.
                # The CNN must receive a skeleton-free image — the model
                # was trained on plain hand photos, not annotated frames.
                # ══════════════════════════════════════════════════════
                clean_frame = frame.copy()

                # ── Step 1: Detect hand ───────────────────────────────
                landmarks = self.tracker.detect(clean_frame)

                predicted_sign = "nothing"
                confidence = 0.0

                if landmarks is not None:
                    # Get bounding box — draws ONLY a thin rectangle,
                    # no red keypoint dots and no green skeleton lines.
                    bbox = self.tracker.get_bounding_box(
                        landmarks, frame.shape, padding=40
                    )
                    # Draw the clean bounding box on the display frame only
                    x1, y1, x2, y2 = bbox
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 220, 220), 2)

                    # ── Step 2: Run CNN inference ─────────────────────
                    # Use clean_frame so skeleton overlay never reaches the CNN
                    if self.predictor.is_ready:
                        preprocessed = self.predictor.preprocess(clean_frame, bbox)
                        if preprocessed is not None:
                            predicted_sign, confidence = self.predictor.predict(preprocessed)
                else:
                    # No hand detected -- reset the duplicate-letter block
                    # so the user can re-sign the same letter after lowering hand
                    if self._current_sign == "nothing":
                        self.stabiliser.last_added_sign = None

                # ── Step 3: Stabilise prediction ──────────────────────
                accepted = self.stabiliser.update(predicted_sign, confidence)

                # Update per-frame display state
                self._current_sign = predicted_sign
                self._current_confidence = confidence

                # ── Step 4: Render HUD overlay ────────────────────────
                HUD.draw(
                    frame,
                    sign=self._current_sign,
                    confidence=self._current_confidence,
                    sentence=self.stabiliser.sentence,
                    progress=self.stabiliser.progress,
                    model_loaded=self.predictor.is_ready,
                    tts_enabled=self.tts.enabled,
                )

                # ── Step 5: Calculate & show FPS ──────────────────────
                self._update_fps(frame)

                # ── Display ───────────────────────────────────────────
                cv2.imshow("ASL Real-Time Recogniser", frame)

                # ── Step 6: Handle keyboard input ─────────────────────
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:  # q or ESC
                    break
                self._handle_key(key)

        except KeyboardInterrupt:
            print("\n[INFO] Interrupted by user.")
        finally:
            # ── Cleanup ───────────────────────────────────────────────
            cap.release()
            cv2.destroyAllWindows()
            self.tracker.close()
            print("[INFO] Recogniser shut down cleanly.")
            if self.stabiliser.sentence:
                print(f"[INFO] Final sentence: \"{self.stabiliser.sentence}\"")

    # ------------------------------------------------------------------
    # Keyboard handler
    # ------------------------------------------------------------------

    def _handle_key(self, key: int):
        """Process keyboard shortcuts for manual sentence editing."""
        if key == ord('s'):
            self.stabiliser.manual_space()
        elif key == 8 or key == 127:   # Backspace / Delete
            self.stabiliser.manual_delete()
        elif key == ord('c'):
            self.stabiliser.manual_clear()
        elif key == ord('t'):
            # Speak the current sentence
            self.tts.speak(self.stabiliser.sentence)

    # ------------------------------------------------------------------
    # FPS display
    # ------------------------------------------------------------------

    def _update_fps(self, frame: np.ndarray):
        """Calculate and render FPS counter on the frame."""
        now = time.time()
        elapsed = now - self._fps_time
        if elapsed > 0:
            self._fps = 1.0 / elapsed
        self._fps_time = now

        fps_text = f"FPS: {self._fps:.0f}"
        h = frame.shape[0]
        cv2.putText(frame, fps_text, (frame.shape[1] - 130, 115),
                    cv2.FONT_HERSHEY_PLAIN, 1.2, (120, 120, 120), 1, cv2.LINE_AA)
