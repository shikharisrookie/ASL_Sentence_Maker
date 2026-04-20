"""
asl_realtime.py — Main entry point for the Real-Time ASL Recognition System.

Usage:
    python asl_realtime.py [--model SLR_final.h5] [--confidence 0.65]
                           [--frames 15] [--cooldown 1.2] [--tts]

Controls:
    s          → Insert space into sentence
    Backspace  → Delete last character
    c          → Clear entire sentence
    t          → Speak the current sentence (if --tts enabled)
    q / ESC    → Quit
"""

import argparse
from core.recognizer import ASLRecognizer


def parse_args():
    """Parse command-line arguments for runtime configuration."""
    parser = argparse.ArgumentParser(
        description="Real-Time ASL Fingerspelling Recognition"
    )
    parser.add_argument(
        "--model", type=str, default="SLR_final.h5",
        help="Path to the trained Keras .h5 model file (default: SLR_final.h5)"
    )
    parser.add_argument(
        "--confidence", type=float, default=0.65,
        help="Minimum prediction confidence to accept a sign (default: 0.65)"
    )
    parser.add_argument(
        "--frames", type=int, default=15,
        help="Number of consecutive identical frames required for stabilisation (default: 15)"
    )
    parser.add_argument(
        "--cooldown", type=float, default=1.2,
        help="Seconds to wait before the same letter can be appended again (default: 1.2)"
    )
    parser.add_argument(
        "--tts", action="store_true",
        help="Enable text-to-speech output (requires pyttsx3)"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    recognizer = ASLRecognizer(
        model_path=args.model,
        confidence_threshold=args.confidence,
        frame_threshold=args.frames,
        cooldown_seconds=args.cooldown,
        enable_tts=args.tts,
    )
    recognizer.run()


if __name__ == "__main__":
    main()
