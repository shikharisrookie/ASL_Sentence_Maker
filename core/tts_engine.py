"""
tts_engine.py — Optional Text-to-Speech Engine.

Responsible for:
  • Initialising pyttsx3 if available.
  • Speaking the current sentence asynchronously (non-blocking).
  • Graceful degradation if pyttsx3 is not installed.
"""

import threading


class TTSEngine:
    """
    Lightweight TTS wrapper using pyttsx3.

    Falls back to a no-op if pyttsx3 is unavailable so the rest of the
    application continues to work without TTS.
    """

    def __init__(self, enabled: bool = False):
        """
        Args:
            enabled: Whether to attempt loading the TTS engine.
        """
        self.enabled = enabled
        self.engine = None

        if not enabled:
            return

        try:
            import pyttsx3
            self.engine = pyttsx3.init()
            # Sensible defaults for clarity
            self.engine.setProperty("rate", 150)
            self.engine.setProperty("volume", 0.9)
            print("[TTSEngine] pyttsx3 initialised successfully.")
        except ImportError:
            print("[TTSEngine] WARNING: pyttsx3 not installed. TTS disabled.")
            print("[TTSEngine]   Install with: pip install pyttsx3")
            self.enabled = False
        except Exception as e:
            print(f"[TTSEngine] WARNING: Failed to initialise TTS: {e}")
            self.enabled = False

    def speak(self, text: str):
        """
        Speak the given text in a background thread so the main loop
        is not blocked.

        Args:
            text: The sentence string to speak aloud.
        """
        if not self.enabled or not self.engine or not text.strip():
            return

        # Run pyttsx3 in a daemon thread to avoid blocking the webcam loop
        thread = threading.Thread(target=self._speak_blocking, args=(text,), daemon=True)
        thread.start()

    def _speak_blocking(self, text: str):
        """Internal blocking speech call (runs in a thread)."""
        try:
            self.engine.say(text)
            self.engine.runAndWait()
        except Exception as e:
            print(f"[TTSEngine] Speech error: {e}")
