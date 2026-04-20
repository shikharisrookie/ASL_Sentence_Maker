"""
hud.py — Heads-Up Display Renderer for the ASL Recogniser.

Responsible for:
  • Rendering a polished overlay on each webcam frame.
  • Showing the detected sign, confidence, stability progress bar,
    sentence buffer, and status indicators.
"""

import cv2
import numpy as np


# ── Colour Palette (BGR) ──
_BG_DARK    = (20, 20, 20)
_WHITE      = (255, 255, 255)
_GREY       = (160, 160, 160)
_GREEN      = (0, 220, 100)
_YELLOW     = (0, 220, 255)
_RED        = (60, 60, 255)
_ACCENT     = (255, 180, 0)
_BAR_BG     = (60, 60, 60)
_BAR_FILL   = (0, 230, 120)


class HUD:
    """Stateless HUD renderer — call `draw()` each frame."""

    FONT = cv2.FONT_HERSHEY_SIMPLEX
    FONT_SMALL = cv2.FONT_HERSHEY_PLAIN

    @staticmethod
    def draw(frame: np.ndarray, *,
             sign: str = "—",
             confidence: float = 0.0,
             sentence: str = "",
             progress: float = 0.0,
             model_loaded: bool = True,
             tts_enabled: bool = False) -> np.ndarray:
        """
        Render the full HUD overlay onto the frame (in-place).

        Args:
            frame: BGR image from the webcam.
            sign: Currently detected sign label.
            confidence: Softmax confidence [0, 1].
            sentence: The sentence buffer to display.
            progress: Stability progress (0–1) toward frame_threshold.
            model_loaded: Whether the CNN model is available.
            tts_enabled: Whether TTS mode is active.

        Returns:
            The same frame (modified in-place).
        """
        h, w = frame.shape[:2]

        # ── Top bar: current sign + confidence ──────────────────────────
        HUD._draw_top_bar(frame, sign, confidence, progress, model_loaded, w)

        # ── Bottom bar: sentence display ────────────────────────────────
        HUD._draw_bottom_bar(frame, sentence, h, w, tts_enabled)

        return frame

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _draw_top_bar(frame, sign, confidence, progress, model_loaded, w):
        """Draw the dark top panel with sign info and progress bar."""
        bar_h = 90
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, bar_h), _BG_DARK, -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

        if not model_loaded:
            cv2.putText(frame, "MODEL NOT LOADED", (20, 55),
                        HUD.FONT, 1.0, _RED, 2, cv2.LINE_AA)
            return

        # Sign label
        sign_display = sign if sign and sign != "nothing" else "—"
        color = _GREEN if confidence > 0.8 else _YELLOW if confidence > 0.5 else _GREY
        cv2.putText(frame, f"Sign: {sign_display}", (20, 40),
                    HUD.FONT, 1.0, color, 2, cv2.LINE_AA)

        # Confidence percentage
        conf_text = f"{confidence * 100:.0f}%"
        cv2.putText(frame, conf_text, (20, 75),
                    HUD.FONT, 0.7, _GREY, 1, cv2.LINE_AA)

        # Stability progress bar (right-aligned)
        bar_x = w - 320
        bar_y = 25
        bar_w = 280
        bar_h_inner = 18
        cv2.rectangle(frame, (bar_x, bar_y),
                      (bar_x + bar_w, bar_y + bar_h_inner), _BAR_BG, -1)

        fill_w = int(bar_w * progress)
        fill_color = _BAR_FILL if progress < 1.0 else _GREEN
        if fill_w > 0:
            cv2.rectangle(frame, (bar_x, bar_y),
                          (bar_x + fill_w, bar_y + bar_h_inner), fill_color, -1)

        # Progress label
        cv2.putText(frame, "Stability", (bar_x, bar_y - 5),
                    HUD.FONT_SMALL, 1.0, _GREY, 1, cv2.LINE_AA)

        # Border
        cv2.rectangle(frame, (bar_x, bar_y),
                      (bar_x + bar_w, bar_y + bar_h_inner), _ACCENT, 1)

        # Lock icon when fully stable
        if progress >= 1.0:
            cv2.putText(frame, "LOCKED", (bar_x + bar_w + 5, bar_y + 14),
                        HUD.FONT_SMALL, 1.0, _GREEN, 1, cv2.LINE_AA)

    @staticmethod
    def _draw_bottom_bar(frame, sentence, h, w, tts_enabled):
        """Draw the dark bottom panel with the sentence buffer."""
        bar_h = 70
        y_start = h - bar_h

        overlay = frame.copy()
        cv2.rectangle(overlay, (0, y_start), (w, h), _BG_DARK, -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

        # Sentence text (truncate if too long)
        max_chars = w // 16  # rough estimate of how many chars fit
        display = sentence if len(sentence) <= max_chars else "…" + sentence[-(max_chars - 1):]
        cv2.putText(frame, f">> {display}_", (20, h - 25),
                    HUD.FONT, 0.85, _WHITE, 2, cv2.LINE_AA)

        # TTS indicator
        if tts_enabled:
            cv2.putText(frame, "TTS ON", (w - 100, h - 25),
                        HUD.FONT_SMALL, 1.0, _GREEN, 1, cv2.LINE_AA)

        # Keyboard shortcut hints (small text above the sentence bar)
        hints = "s=Space  Bksp=Delete  c=Clear  t=Speak  q=Quit"
        cv2.putText(frame, hints, (20, y_start - 8),
                    HUD.FONT_PLAIN if hasattr(HUD, 'FONT_PLAIN') else HUD.FONT_SMALL,
                    0.9, (120, 120, 120), 1, cv2.LINE_AA)
