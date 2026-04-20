"""
ASL Real-Time Recognition Application
======================================
A modular, production-quality American Sign Language recognition system
using OpenCV, MediaPipe (hand tracking), and a pre-trained Keras CNN model.

Features:
  - Live webcam inference with MediaPipe hand tracking
  - Prediction stabilization via confidence threshold (≥0.8) + frame consistency
  - Sentence buffer: append only on new letter + confidence + 1.5s debounce
  - SPACE gesture + Space bar / 'S' key → add space between words
  - Cooldown mechanism to prevent repeated letter triggering
  - Polished on-screen UI with detected letter, confidence bar, and sentence
  - Text-to-speech (pyttsx3) speaks each letter/word as it's added
  - Keyboard shortcuts: Space/S=space, Backspace=delete, C=clear, T=speak, Q=quit

Author: Shikhar (assisted by Antigravity AI)
"""

import cv2
import numpy as np
import time
import threading
import queue
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from tensorflow.keras.models import load_model


# ──────────────────────────────────────────────────────────────────────────────
# GEOMETRIC SIGN OVERRIDE — Landmark-based classifier for signs the CNN
# struggles with due to training-data mismatch (e.g. ASL 'A').
# ──────────────────────────────────────────────────────────────────────────────
def geometric_sign_override(landmarks):
    """
    Uses MediaPipe hand landmark geometry to detect ASL 'A' which the CNN
    consistently mis-classifies (returns 'nothing', 'S', 'T', 'Z', 'Y').

    Four geometric checks must ALL pass to trigger 'A':
        1. All 4 fingers CURLED (tips below PIPs).
        2. Thumb SPREAD to the side (ratio > 0.35).
        3. Thumb tip ABOVE or at the knuckle line.
        4. KEY — Thumb tip x is OUTSIDE the knuckle horizontal band:
             'S': thumb wraps OVER fist → tip is INSIDE the band.
             'A': thumb sticks OUT laterally → tip is OUTSIDE the band.
    """
    lm = landmarks
    # Landmark indices (MediaPipe):
    #   Tips:  4=thumb 8=index 12=middle 16=ring 20=pinky
    #   PIPs:  6=index 10=middle 14=ring 18=pinky
    #   MCPs:  5=index  9=middle 13=ring 17=pinky
    #   Wrist: 0

    # ── Check 1: All 4 fingers curled ────────────────────────────────────
    if not (lm[8].y  > lm[6].y  and
            lm[12].y > lm[10].y and
            lm[16].y > lm[14].y and
            lm[20].y > lm[18].y):
        return None  # Hand is open

    # ── Check 2: Thumb has lateral spread ────────────────────────────────
    hand_size = max(
        ((lm[9].x - lm[0].x)**2 + (lm[9].y - lm[0].y)**2) ** 0.5, 0.01
    )
    thumb_spread = (
        (lm[4].x - lm[5].x)**2 + (lm[4].y - lm[5].y)**2
    ) ** 0.5 / hand_size
    if thumb_spread <= 0.35:
        return None  # Thumb tucked (S, M, E)

    # ── Check 3: Thumb tip not below knuckle line ────────────────────────
    avg_knuckle_y = (lm[5].y + lm[9].y + lm[13].y) / 3.0
    if lm[4].y > avg_knuckle_y + 0.05:
        return None  # Thumb pointing downward

    # ── Check 4: Thumb tip is OUTSIDE the knuckle x-band ────────────────
    # 'S' → thumb crosses OVER the fist → tip x is inside the knuckle band
    # 'A' → thumb sticks OUT to the side → tip x is outside the knuckle band
    kxs = [lm[5].x, lm[9].x, lm[13].x, lm[17].x]
    kx_min, kx_max = min(kxs), max(kxs)
    margin = 0.015  # small tolerance
    if kx_min - margin <= lm[4].x <= kx_max + margin:
        return None  # Thumb is over the fist → 'S', 'M', 'E', not 'A'

    # All 4 checks passed → ASL 'A'
    return ('A', 0.95)



# ──────────────────────────────────────────────────────────────────────────────
# 1. PREDICTION STABILIZER — Smooths noisy frame-by-frame predictions
# ──────────────────────────────────────────────────────────────────────────────
class PredictionStabilizer:
    """
    Implements temporal stability logic to prevent flickering predictions.
    
    A prediction is only considered "stable" when the SAME sign is detected
    for `frame_threshold` consecutive frames AND the confidence exceeds
    `confidence_threshold`.
    
    Uses a rolling average of recent predictions for additional smoothing.
    """

    def __init__(self, confidence_threshold=0.8, frame_threshold=10, history_size=3):
        self.confidence_threshold = confidence_threshold  # Minimum confidence to accept
        self.frame_threshold = frame_threshold            # Consecutive frames required
        self.history_size = history_size                   # Rolling average window size (smaller = faster response)
        
        # Internal state
        self.current_sign = "nothing"      # Currently tracked sign
        self.frame_counter = 0             # How many consecutive frames match 
        self.prediction_history = []       # Rolling buffer of raw prediction arrays
        self.stable_sign = "nothing"       # The last fully-stabilized sign
        self.stable_confidence = 0.0       # Confidence of the stable sign

    def update(self, raw_prediction_array, labels_map):
        """
        Feed a new frame's raw prediction array into the stabilizer.
        
        Args:
            raw_prediction_array: np.ndarray of shape (num_classes,) from model.predict()
            labels_map: dict mapping class index -> label string
            
        Returns:
            tuple: (stabilized_sign: str, confidence: float, is_newly_stable: bool)
        """
        # Step 1: Append to rolling history buffer
        self.prediction_history.append(raw_prediction_array)
        if len(self.prediction_history) > self.history_size:
            self.prediction_history.pop(0)
        
        # Step 2: Compute smoothed prediction via rolling average
        avg_prediction = np.mean(self.prediction_history, axis=0)
        predicted_class = int(np.argmax(avg_prediction))
        confidence = float(np.max(avg_prediction))
        predicted_sign = labels_map.get(predicted_class, "nothing")
        
        # Step 3: Apply confidence gate
        if confidence < self.confidence_threshold:
            predicted_sign = "nothing"  # Below threshold = treat as no gesture
        
        # Step 4: Frame consistency counter
        is_newly_stable = False
        if predicted_sign == self.current_sign:
            self.frame_counter += 1
        else:
            self.current_sign = predicted_sign
            self.frame_counter = 1  # Reset on sign change
        
        # Step 5: Check if stability threshold was reached this exact frame
        if self.frame_counter == self.frame_threshold:
            self.stable_sign = self.current_sign
            self.stable_confidence = confidence
            is_newly_stable = True  # Signal to sentence buffer: "add this letter now"
        
        return self.current_sign, confidence, is_newly_stable

    def reset(self):
        """Clear all stabilizer state (used when clearing the sentence)."""
        self.current_sign = "nothing"
        self.frame_counter = 0
        self.prediction_history.clear()
        self.stable_sign = "nothing"
        self.stable_confidence = 0.0


# ──────────────────────────────────────────────────────────────────────────────
# 2. SENTENCE BUFFER — Manages the sentence being built from gestures
# ──────────────────────────────────────────────────────────────────────────────
class SentenceBuffer:
    """
    Maintains the running sentence string and handles gesture-based editing.
    
    Includes a cooldown mechanism: after a letter is added, the same letter
    cannot be added again until either:
      (a) a different sign is stabilized, OR
      (b) the hand returns to "nothing" (neutral state)
      
    This prevents holding up "A" and getting "AAAAAA" in your sentence.
    """

    def __init__(self, cooldown_seconds=1.5):
        self.sentence = ""                     # The sentence being built
        self.last_added_sign = None            # Last sign that was appended
        self.last_add_time = 0                 # Timestamp of last addition
        self.cooldown_seconds = cooldown_seconds  # Minimum gap between same letter
        self.letter_just_added = False         # Flag for UI flash effect

    def process_stable_sign(self, sign, is_newly_stable):
        """
        Called every frame with the current stabilized sign.
        Only appends to the sentence when `is_newly_stable` is True
        and the cooldown has elapsed.
        
        Args:
            sign: str — the stabilized sign label
            is_newly_stable: bool — True only on the exact frame stability was reached
        """
        self.letter_just_added = False
        now = time.time()
        
        # Reset cooldown lock when hand goes neutral
        if sign == "nothing":
            self.last_added_sign = None
            return
            
        if not is_newly_stable:
            return  # Not yet stable enough — do nothing
        
        # Cooldown check: prevent rapid re-triggering of the same sign
        if sign == self.last_added_sign and (now - self.last_add_time) < self.cooldown_seconds:
            return  # Still in cooldown for this letter
        
        # Handle special gesture classes
        if sign == "space":
            self.sentence += " "
            self.letter_just_added = True
        elif sign == "del":
            if len(self.sentence) > 0:
                self.sentence = self.sentence[:-1]
            self.letter_just_added = True
        elif sign != "nothing":
            # Regular letter A-Z
            self.sentence += sign
            self.letter_just_added = True
        
        # Update cooldown tracking
        if self.letter_just_added:
            self.last_added_sign = sign
            self.last_add_time = now

    def add_space(self):
        """Manually add a space (keyboard shortcut)."""
        self.sentence += " "
        self.last_added_sign = None

    def backspace(self):
        """Remove last character (keyboard shortcut)."""
        if len(self.sentence) > 0:
            self.sentence = self.sentence[:-1]

    def clear(self):
        """Clear entire sentence (keyboard shortcut)."""
        self.sentence = ""
        self.last_added_sign = None

    def get_sentence(self):
        return self.sentence


# ──────────────────────────────────────────────────────────────────────────────
# 3. HAND TRACKER — Wraps MediaPipe hand detection and landmark extraction
# ──────────────────────────────────────────────────────────────────────────────
class HandTracker:
    """
    Uses MediaPipe Tasks API for robust hand detection and landmark tracking.
    Provides bounding-box extraction and skeleton drawing on each frame.
    """

    def __init__(self, model_asset_path="hand_landmarker.task"):
        base_options = mp_python.BaseOptions(model_asset_path=model_asset_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=1,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.detector = vision.HandLandmarker.create_from_options(options)

    def detect(self, rgb_frame):
        """
        Run hand detection on an RGB frame.
        
        Returns:
            landmarks list or None if no hand detected
        """
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        result = self.detector.detect(mp_image)
        
        if result.hand_landmarks and len(result.hand_landmarks) > 0:
            return result.hand_landmarks[0]
        return None

    def get_bounding_box(self, frame, landmarks):
        """
        Compute a square bounding box around the detected hand landmarks.
        Draws ONLY a clean bounding-box rectangle on the display frame
        (no red dots, no green skeleton lines) to keep the overlay minimal
        and avoid visual confusion.

        Returns:
            (x_min, y_min, x_max, y_max) — pixel coordinates of the square crop.
        """
        h, w, _ = frame.shape
        x_min, y_min = w, h
        x_max, y_max = 0, 0

        # Compute bounding box from raw landmarks (no drawing)
        for lm in landmarks:
            cx, cy = int(lm.x * w), int(lm.y * h)
            x_min, y_min = min(x_min, cx), min(y_min, cy)
            x_max, y_max = max(x_max, cx), max(y_max, cy)

        # Force the bounding box to be a perfect square (prevents CNN distortion).
        # Use tight padding (40px) so the hand fills most of the crop — this
        # closely matches training images where the hand fills ~80% of the frame.
        # Too much padding causes the hand to appear smaller than training data,
        # which confuses visually similar signs like A vs M.
        center_x = (x_min + x_max) // 2
        center_y = (y_min + y_max) // 2
        box_size = max(x_max - x_min, y_max - y_min) + 40  # tight, training-matched padding
        half = box_size // 2

        x_min = max(0, center_x - half)
        y_min = max(0, center_y - half)
        x_max = min(w, center_x + half)
        y_max = min(h, center_y + half)

        # Enforce strict square if we hit the frame edge
        crop_size = min(x_max - x_min, y_max - y_min)
        x_max = x_min + crop_size
        y_max = y_min + crop_size

        # Draw a clean cyan bounding box — no skeleton, no red dots
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 220, 220), 2)
        return x_min, y_min, x_max, y_max

    def close(self):
        """Explicitly release MediaPipe resources."""
        if self.detector is not None:
            self.detector.close()


# ──────────────────────────────────────────────────────────────────────────────
# 4. IMAGE PREPROCESSOR — Prepares hand crops for CNN inference
# ──────────────────────────────────────────────────────────────────────────────
class ImagePreprocessor:
    """
    Crops the hand region from a frame and prepares it for the CNN model.
    The model expects a 64x64 RGB image normalized to [0, 1].
    """

    @staticmethod
    def prepare_for_model(frame, bbox, target_size=(64, 64)):
        """
        Crop, resize, and normalize the hand region for CNN inference.

        Preprocessing pipeline matches the training data exactly:
          1. Crop the BGR ROI from the clean (no-overlay) frame
          2. Resize to 64×64
          3. Convert BGR → RGB   (Keras ImageDataGenerator loads as RGB)
          4. Normalize to [0, 1]
          5. Add batch dimension

        Args:
            frame: BGR frame from OpenCV (must be the clean, overlay-free copy)
            bbox: (x_min, y_min, x_max, y_max) bounding box in pixel coords
            target_size: tuple, the CNN's expected input size (default 64×64)

        Returns:
            np.ndarray of shape (1, 64, 64, 3), or None if the crop is invalid
        """
        x_min, y_min, x_max, y_max = bbox

        # Safety: reject crops that are too small to be useful
        if (x_max - x_min) < 20 or (y_max - y_min) < 20:
            return None

        roi = frame[y_min:y_max, x_min:x_max]
        if roi.size == 0:
            return None

        # Resize first (in BGR), then convert colour — matches cv2 flow used at training
        roi_resized = cv2.resize(roi, target_size)
        # BGR -> RGB  (training used cv2.COLOR_BGR2RGB before feeding to Keras)
        roi_rgb = cv2.cvtColor(roi_resized, cv2.COLOR_BGR2RGB)
        # Normalize pixel values to [0, 1]
        roi_normalized = roi_rgb.astype("float32") / 255.0
        # Add batch dimension: (64, 64, 3) -> (1, 64, 64, 3)
        return np.expand_dims(roi_normalized, axis=0)


# ──────────────────────────────────────────────────────────────────────────────
# 5. UI RENDERER — Draws the polished on-screen overlay
# ──────────────────────────────────────────────────────────────────────────────
class UIRenderer:
    """
    Renders a clean, semi-transparent UI overlay onto each frame showing:
      - Current detected sign + confidence level
      - Sentence being formed  
      - Frame stability progress bar
      - Keyboard shortcut hints
      - Visual flash when a letter is added
    """

    # Color palette
    BG_COLOR = (20, 20, 20)         # Dark panel background
    TEXT_WHITE = (255, 255, 255)     # Primary text
    TEXT_GRAY = (160, 160, 160)      # Secondary/hint text
    ACCENT_GREEN = (0, 230, 118)    # Active/positive indicators
    ACCENT_YELLOW = (0, 210, 255)   # Warning / in-progress
    ACCENT_RED = (60, 60, 255)      # Error / missing model
    FLASH_COLOR = (0, 255, 200)     # Letter-added flash

    @staticmethod
    def draw(frame, current_sign, confidence, frame_counter, frame_threshold,
             sentence, model_loaded, letter_just_added):
        """Render the full UI overlay onto the frame."""
        h, w, _ = frame.shape
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_small = cv2.FONT_HERSHEY_PLAIN

        # ── Top bar: current detection info ──────────────────────────────
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 80), UIRenderer.BG_COLOR, -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        if not model_loaded:
            cv2.putText(frame, "[!] MODEL NOT LOADED", (20, 50),
                        font, 1, UIRenderer.ACCENT_RED, 2, cv2.LINE_AA)
        elif current_sign and current_sign != "nothing":
            # Detected sign label
            sign_text = f"Detected: {current_sign.upper()}"
            cv2.putText(frame, sign_text, (20, 35),
                        font, 0.9, UIRenderer.ACCENT_GREEN, 2, cv2.LINE_AA)

            # Confidence percentage
            conf_text = f"Confidence: {confidence * 100:.1f}%"
            conf_color = UIRenderer.ACCENT_GREEN if confidence > 0.85 else UIRenderer.ACCENT_YELLOW
            cv2.putText(frame, conf_text, (20, 65),
                        font_small, 1.3, conf_color, 1, cv2.LINE_AA)

            # Frame stability progress bar (top right)
            bar_x, bar_y, bar_w, bar_h = w - 220, 20, 200, 20
            progress = min(frame_counter / frame_threshold, 1.0)
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h),
                          UIRenderer.TEXT_GRAY, 1)
            fill_w = int(bar_w * progress)
            bar_color = UIRenderer.ACCENT_GREEN if progress >= 1.0 else UIRenderer.ACCENT_YELLOW
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h),
                          bar_color, -1)
            cv2.putText(frame, f"{frame_counter}/{frame_threshold}", (bar_x + 5, bar_y + 15),
                        font_small, 1.0, UIRenderer.BG_COLOR, 1, cv2.LINE_AA)
        else:
            cv2.putText(frame, "Show a sign to start...", (20, 50),
                        font, 0.8, UIRenderer.TEXT_GRAY, 1, cv2.LINE_AA)

        # ── Flash effect when a letter is added ──────────────────────────
        if letter_just_added:
            flash_overlay = frame.copy()
            cv2.rectangle(flash_overlay, (0, 0), (w, h), UIRenderer.FLASH_COLOR, -1)
            cv2.addWeighted(flash_overlay, 0.1, frame, 0.9, 0, frame)

        # ── Bottom panel: sentence display ───────────────────────────────
        panel_h = 90
        panel_y = h - panel_h
        overlay2 = frame.copy()
        cv2.rectangle(overlay2, (0, panel_y), (w, h), UIRenderer.BG_COLOR, -1)
        cv2.addWeighted(overlay2, 0.75, frame, 0.25, 0, frame)

        # Sentence text
        display_sentence = sentence if sentence else "(empty)"
        cv2.putText(frame, "Sentence:", (20, panel_y + 25),
                    font_small, 1.3, UIRenderer.TEXT_GRAY, 1, cv2.LINE_AA)
        cv2.putText(frame, display_sentence, (20, panel_y + 55),
                    font, 0.8, UIRenderer.TEXT_WHITE, 2, cv2.LINE_AA)

        # Keyboard shortcuts hint
        hint = "'Space' Space | 'Backspace' Del | 'C' Clear | 'T' Speak | 'Q' Quit"
        cv2.putText(frame, hint, (20, panel_y + 80),
                    font_small, 1.0, UIRenderer.TEXT_GRAY, 1, cv2.LINE_AA)

        return frame


# ──────────────────────────────────────────────────────────────────────────────
# 6. TEXT-TO-SPEECH ENGINE — Background Threaded Wrapper
# ──────────────────────────────────────────────────────────────────────────────
class TextToSpeech:
    """
    Asynchronous text-to-speech wrapper using a background thread and queue.
    This ensures that pyttsx3.runAndWait() does not block the webcam feed.
    """

    def __init__(self):
        self.tts_queue = queue.Queue()
        self.enabled = False
        
        try:
            import pyttsx3
            self.enabled = True
            # Start background thread to process speech requests
            self.worker_thread = threading.Thread(target=self._tts_worker, daemon=True)
            self.worker_thread.start()
            print("[TTS] Asynchronous Text-to-Speech engine initialized.")
        except ImportError:
            print("[TTS] pyttsx3 not installed. TTS disabled.")
            print("      Install with: pip install pyttsx3")

    def _tts_worker(self):
        """Background worker that continuously reads from the queue and speaks."""
        import pyttsx3
        # pyttsx3 MUST be initialized inside the thread it runs on
        engine = pyttsx3.init()
        engine.setProperty('rate', 150)   # Speaking speed
        engine.setProperty('volume', 0.9)
        
        while True:
            text = self.tts_queue.get()
            if text is None:
                break
            try:
                engine.say(text)
                engine.runAndWait()
            except Exception as e:
                print(f"[TTS] Speech error: {e}")
            self.tts_queue.task_done()

    def speak(self, text):
        """Add text to the speech queue to be spoken asynchronously."""
        if self.enabled and text and text.strip():
            self.tts_queue.put(text)


# ──────────────────────────────────────────────────────────────────────────────
# 7. MAIN APPLICATION — Orchestrates all components
# ──────────────────────────────────────────────────────────────────────────────
class ASLRecognizer:
    """
    Main application class that ties together all modules:
      - HandTracker for detection
      - ImagePreprocessor for CNN input preparation
      - PredictionStabilizer for temporal smoothing
      - SentenceBuffer for text accumulation
      - UIRenderer for on-screen display
      - TextToSpeech for spoken output
    """

    def __init__(self, model_path="SLR_final.h5",
                 confidence_threshold=0.80,
                 frame_threshold=20,
                 cooldown_seconds=1.5):
        """
        Initialize the ASL recognizer application.
        
        Args:
            model_path: path to the pre-trained .h5 Keras model
            confidence_threshold: minimum prediction confidence to accept (0.0 - 1.0)
            frame_threshold: consecutive frames needed before a sign is "locked in"
            cooldown_seconds: minimum seconds between adding the same letter twice
        """
        print("=" * 60)
        print("  ASL Real-Time Recognition System")
        print("=" * 60)

        # Load the pre-trained CNN model
        print(f"\n[MODEL] Loading from {model_path}...")
        try:
            self.model = load_model(model_path)
            self.model_loaded = True
            print("[MODEL] Loaded successfully!")
        except Exception as e:
            print(f"[MODEL] Error: {e}")
            print("[MODEL] Running in demo mode (no predictions)")
            self.model = None
            self.model_loaded = False

        # Build class label map: 0-25 -> A-Z, 26 -> del, 27 -> nothing, 28 -> space
        classes = [chr(i) for i in range(65, 91)] + ["del", "nothing", "space"]
        self.labels_map = {i: cls for i, cls in enumerate(classes)}

        # Initialize all sub-components
        self.hand_tracker = HandTracker()
        self.stabilizer = PredictionStabilizer(
            confidence_threshold=confidence_threshold,
            frame_threshold=frame_threshold
        )
        self.sentence_buffer = SentenceBuffer(cooldown_seconds=cooldown_seconds)
        self.tts = TextToSpeech()

        print(f"\n[CONFIG] Confidence threshold: {confidence_threshold}")
        print(f"[CONFIG] Frame threshold: {frame_threshold}")
        print(f"[CONFIG] Cooldown: {cooldown_seconds}s")
        print("\n" + "=" * 60)

    def _handle_keyboard(self, key):
        """
        Process keyboard input for manual controls.
        
        Shortcuts:
            Space     -> Add a space between words  (main shortcut)
            S         -> Add a space between words  (alias)
            Backspace -> Delete last character
            C         -> Clear entire sentence
            T         -> Speak the current sentence aloud
            Q         -> Quit the application
            
        Returns:
            True if should quit, False otherwise
        """
        if key == ord('q'):
            return True
        elif key == 32 or key == ord('s'):  # Space bar (32) or 'S' key
            self.sentence_buffer.add_space()
            print("[INPUT] Space added")
        elif key == 8 or key == 127:  # Backspace / Delete
            self.sentence_buffer.backspace()
            print("[INPUT] Backspace")
        elif key == ord('c'):
            self.sentence_buffer.clear()
            self.stabilizer.reset()
            print("[INPUT] Sentence cleared")
        elif key == ord('t'):
            sentence = self.sentence_buffer.get_sentence()
            if sentence.strip():
                print(f"\n[TTS] Speaking: \"{sentence}\"")
                self.tts.speak(sentence)
            else:
                print("[TTS] Nothing to speak yet.")
        return False

    def run(self):
        """
        Main application loop.
        Captures webcam frames, runs inference, updates the sentence,
        and renders the UI until the user quits.
        """
        print("\nStarting camera... Press 'Q' to quit.\n")
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            print("[ERROR] Cannot open webcam. Check camera permissions.")
            return

        _frame_idx = 0          # Frame counter for CNN skip logic
        _last_raw_pred = None   # Cache last CNN output for skipped frames
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[WARNING] Frame capture failed. Retrying...")
                continue

            # Mirror the frame (more natural for the user)
            frame = cv2.flip(frame, 1)

            # ── IMPORTANT: Save a clean copy BEFORE drawing anything ─────
            # The CNN must receive a clean image (no skeleton overlays).
            # Drawing skeleton/keypoints on 'frame' before cropping was
            # corrupting the CNN input, causing it to always predict 'Q'.
            clean_frame = frame.copy()
            rgb_frame = cv2.cvtColor(clean_frame, cv2.COLOR_BGR2RGB)

            # ── Step 1: Detect hand ──────────────────────────────────────
            landmarks = self.hand_tracker.detect(rgb_frame)

            current_sign = "nothing"
            confidence = 0.0
            is_newly_stable = False

            if landmarks is not None:
                # Get bounding box; draws only a clean rectangle (no red dots/skeleton)
                bbox = self.hand_tracker.get_bounding_box(frame, landmarks)

                # ── Step 2: Crop and preprocess for CNN ──────────────────
                # Run CNN every 2nd frame; re-use last result on skipped frames.
                # The rolling average in the stabilizer keeps predictions smooth.
                if self.model_loaded:
                    input_tensor = ImagePreprocessor.prepare_for_model(clean_frame, bbox)

                    if input_tensor is not None:
                        if _frame_idx % 2 == 0 or _last_raw_pred is None:
                            # ── Step 3: Run CNN inference ─────────────────
                            raw_prediction = self.model.predict(input_tensor, verbose=0)[0]
                            _last_raw_pred = raw_prediction
                        else:
                            # Use cached prediction from previous frame
                            raw_prediction = _last_raw_pred

                         # ── Step 3b: Conditional geometric override for 'A' ──
                        # The diagnostic showed this user's 'A' → CNN returns:
                        #   'nothing', 'S', 'T', 'Z', 'Y', 'M', 'N', 'H', 'K'
                        # We ONLY apply the geo override when CNN is already in
                        # that confusion zone. If CNN confidently predicts any
                        # other sign (B, C, D, ...) we leave it completely alone.
                        # This prevents false A/M detections on all other letters.
                        A_CONFUSION_SIGNS = {
                            "nothing", "S", "T", "Z", "Y", "M", "N", "H", "K"
                        }
                        cnn_top_idx  = int(np.argmax(raw_prediction))
                        cnn_top_sign = self.labels_map.get(cnn_top_idx, "nothing")
                        cnn_top_conf = float(raw_prediction[cnn_top_idx])

                        # Gate: only attempt geo when CNN is confused
                        if cnn_top_conf < 0.70 or cnn_top_sign in A_CONFUSION_SIGNS:
                            geo_result = geometric_sign_override(landmarks)
                            if geo_result is not None:
                                geo_sign, geo_conf = geo_result
                                geo_idx = next(
                                    (k for k, v in self.labels_map.items()
                                     if v == geo_sign), None
                                )
                                if geo_idx is not None:
                                    # Heavily weight geometric result (85/15 geo/CNN)
                                    synthetic_pred = raw_prediction * 0.15
                                    synthetic_pred[geo_idx] = geo_conf
                                    raw_prediction = synthetic_pred / synthetic_pred.sum()

                        # ── Step 4: Stabilize prediction ─────────────────
                        current_sign, confidence, is_newly_stable = \
                            self.stabilizer.update(raw_prediction, self.labels_map)
            else:
                _last_raw_pred = None   # Reset cache when hand leaves frame
                # No hand detected — feed "nothing" to stabilizer
                nothing_pred = np.zeros(len(self.labels_map))
                nothing_pred[27] = 1.0  # Index 27 = "nothing"
                current_sign, confidence, is_newly_stable = \
                    self.stabilizer.update(nothing_pred, self.labels_map)

            # ── Step 5: Update sentence buffer ───────────────────────────
            self.sentence_buffer.process_stable_sign(current_sign, is_newly_stable)

            # Auto-Speak the letter/word when it's newly added to the buffer
            if self.sentence_buffer.letter_just_added:
                sign_to_speak = self.sentence_buffer.last_added_sign
                # If they gesture 'space', speak the whole last word!
                if sign_to_speak == "space":
                    words = self.sentence_buffer.sentence.strip().split()
                    if words:
                        self.tts.speak(words[-1]) # Speak the last completed word
                elif sign_to_speak == "del":
                    self.tts.speak("delete")
                elif sign_to_speak and sign_to_speak != "nothing":
                    self.tts.speak(sign_to_speak) # Speak the single character

            # ── Step 6: Draw UI ──────────────────────────────────────────
            frame = UIRenderer.draw(
                frame,
                current_sign=current_sign,
                confidence=confidence,
                frame_counter=self.stabilizer.frame_counter,
                frame_threshold=self.stabilizer.frame_threshold,
                sentence=self.sentence_buffer.get_sentence(),
                model_loaded=self.model_loaded,
                letter_just_added=self.sentence_buffer.letter_just_added
            )

            cv2.imshow("ASL Real-Time Recognizer", frame)
            _frame_idx += 1

            # ── Step 7: Handle keyboard input ────────────────────────────
            key = cv2.waitKey(1) & 0xFF
            if self._handle_keyboard(key):
                break

        # ── Cleanup ──────────────────────────────────────────────────────
        print("\nShutting down...")
        cap.release()
        cv2.destroyAllWindows()
        self.hand_tracker.close()
        print("Goodbye!")


# ──────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # ┌──────────────────────────────────────────────────────────────────┐
    # │ CONFIGURATION — Adjust these values to tune behavior            │
    # │                                                                  │
    # │  confidence_threshold: Higher = stricter (fewer false positives) │
    # │  frame_threshold:      Higher = slower but more stable           │
    # │  cooldown_seconds:     Higher = longer gap between same letter   │
    # └──────────────────────────────────────────────────────────────────┘
    recognizer = ASLRecognizer(
        model_path="SLR_final.h5",
        confidence_threshold=0.80,   # Balanced threshold — geometric override handles A
        frame_threshold=10,          # 10 frames ≈ 0.6s at ~15 CNN fps — fast response
        cooldown_seconds=1.2         # 1.2s debounce between same letter
    )
    recognizer.run()
