"""
stabiliser.py — Prediction Stabilisation & Sentence Buffer Management.

Responsible for:
  • Filtering noisy per-frame predictions via confidence thresholding.
  • Requiring N consecutive identical predictions before accepting a sign.
  • Enforcing a cooldown period so the same letter isn't appended repeatedly.
  • Managing the running sentence buffer (append, space, delete, clear).
"""

import time


class PredictionStabiliser:
    """
    Smooths raw frame-by-frame predictions into stable, actionable sign events.

    Pipeline:
        raw prediction -> confidence gate -> frame consistency check -> cooldown gate -> buffer action
    """

    def __init__(self,
                 confidence_threshold: float = 0.65,
                 frame_threshold: int = 15,
                 cooldown_seconds: float = 1.2):
        """
        Args:
            confidence_threshold: Minimum softmax confidence to consider a prediction valid.
            frame_threshold: How many consecutive frames of the *same* sign are
                             needed before it is accepted and appended.
            cooldown_seconds: Minimum time (s) between two consecutive appends of
                              the *same* character.  Prevents repeated letters when
                              the user holds a pose too long.
        """
        self.confidence_threshold = confidence_threshold
        self.frame_threshold = frame_threshold
        self.cooldown_seconds = cooldown_seconds

        # --- Internal state ---
        self.current_sign: str | None = None     # Sign being tracked for consistency
        self.frame_counter: int = 0              # Consecutive matching frames so far
        self.last_added_sign: str | None = None  # Last character added to the sentence
        self.last_add_time: float = 0.0          # Timestamp of the last buffer action
        self.sentence: str = ""                  # Running sentence buffer

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, predicted_sign: str, confidence: float) -> str | None:
        """
        Feed one frame's prediction into the stabiliser.

        Args:
            predicted_sign: Raw class label from the predictor.
            confidence: Softmax confidence score for that class.

        Returns:
            The accepted sign string if a stable action was triggered, else None.
        """
        # Step 1: Confidence gate — reject low-confidence predictions
        if confidence < self.confidence_threshold:
            predicted_sign = "nothing"

        # Step 2: Frame consistency — count identical consecutive frames
        if predicted_sign == self.current_sign:
            self.frame_counter += 1
        else:
            self.current_sign = predicted_sign
            self.frame_counter = 1

        # Step 3: Once "nothing" is seen, reset the duplicate-block so the
        #          user can sign the same letter again after lowering their hand.
        if predicted_sign == "nothing":
            self.last_added_sign = None
            return None

        # Step 4: Check if we've reached the stability threshold
        if self.frame_counter < self.frame_threshold:
            return None

        # Step 5: Cooldown gate — prevent the same sign from firing too rapidly
        now = time.time()
        if (predicted_sign == self.last_added_sign
                and (now - self.last_add_time) < self.cooldown_seconds):
            return None

        # ---- Sign accepted — perform buffer action ----
        accepted = self._apply_to_buffer(predicted_sign)
        if accepted:
            self.last_added_sign = predicted_sign
            self.last_add_time = now
            self.frame_counter = 0   # Reset so the user must re-stabilise

        return accepted

    def manual_space(self):
        """Insert a space via keyboard shortcut."""
        self.sentence += " "
        self.last_added_sign = None

    def manual_delete(self):
        """Delete the last character via keyboard shortcut."""
        if self.sentence:
            self.sentence = self.sentence[:-1]

    def manual_clear(self):
        """Clear the entire sentence via keyboard shortcut."""
        self.sentence = ""
        self.last_added_sign = None

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _apply_to_buffer(self, sign: str) -> str | None:
        """
        Apply the accepted sign to the sentence buffer.

        Returns the sign string on success, None if it was a no-op.
        """
        if sign == "space":
            self.sentence += " "
            return sign
        elif sign == "del":
            if self.sentence:
                self.sentence = self.sentence[:-1]
                return sign
            return None
        elif sign not in ("nothing",):
            self.sentence += sign
            return sign
        return None

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    @property
    def progress(self) -> float:
        """
        Fraction of frames accumulated toward the threshold (0.0 – 1.0).
        Useful for displaying a progress bar on the HUD.
        """
        if self.frame_threshold <= 0:
            return 1.0
        return min(self.frame_counter / self.frame_threshold, 1.0)
