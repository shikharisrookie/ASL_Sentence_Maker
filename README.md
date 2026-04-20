# 🤟 ASL Real-Time Sign Language Recognition

A real-time **American Sign Language (ASL) fingerspelling recognition** system built with Python, OpenCV, MediaPipe, and a custom-trained TensorFlow/Keras CNN.

---

## 🎯 Features

- **Live webcam inference** — detects and classifies ASL hand signs in real-time
- **MediaPipe hand tracking** — precise 21-point hand landmark detection
- **CNN model** — custom-trained on the ASL Alphabet dataset (29 classes: A–Z + space, del, nothing)
- **Geometric override layer** — landmark-based correction for signs the CNN struggles with
- **Prediction stabilization** — rolling average + frame consistency to avoid flickering
- **Sentence builder** — accumulates letters into a full sentence with cooldown logic
- **Text-to-Speech** — speaks letters/words as they're formed (optional, requires `pyttsx3`)
- **Keyboard shortcuts** — Space, Backspace, C (clear), T (speak), Q (quit)

---

## 🗂️ Project Structure

```
Sign-Language-Recognition-main/
│
├── asl_recognizer.py          # 🚀 Main entry point (run this)
├── SLR_final.h5               # Trained Keras CNN model (not on GitHub – download separately)
├── hand_landmarker.task       # MediaPipe hand detection model
│
├── core/                      # Modular version (used by asl_realtime.py)
│   ├── hand_tracker.py
│   ├── predictor.py
│   ├── stabiliser.py
│   ├── hud.py
│   ├── tts_engine.py
│   └── recognizer.py
│
├── SignLanguageRecognition.ipynb  # Training notebook
├── requirments.txt                # Python dependencies
└── README.md
```

---

## ⚙️ Setup & Installation

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/Sign-Language-Recognition.git
cd Sign-Language-Recognition
```

### 2. Create a virtual environment
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Mac/Linux:
source .venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirments.txt
```

### 4. Download the model files
The trained model (`SLR_final.h5`) and MediaPipe model (`hand_landmarker.task`) are too large for GitHub.

- **`SLR_final.h5`** — train your own using `SignLanguageRecognition.ipynb`, or download from the project owner.
- **`hand_landmarker.task`** — download from [MediaPipe Models](https://developers.google.com/mediapipe/solutions/vision/hand_landmarker#models) and place in the project root.

---

## 🚀 Running the App

```bash
# Activate venv first (if not already active)
.venv\Scripts\activate     # Windows
source .venv/bin/activate  # Mac/Linux

# Run
python asl_recognizer.py
```

---

## 🎮 Keyboard Controls

| Key | Action |
|-----|--------|
| `Space` or `S` | Add space between words |
| `Backspace` | Delete last character |
| `C` | Clear entire sentence |
| `T` | Speak current sentence (TTS) |
| `Q` or `ESC` | Quit |

---

## 📦 Dependencies

```
tensorflow
keras
opencv-python
mediapipe
numpy
pyttsx3  (optional, for text-to-speech)
```

---

## 🧠 Model Details

- **Architecture**: Custom CNN (Conv2D × 6, BatchNorm, MaxPool, Dropout, Dense)
- **Input**: 64×64 RGB images
- **Classes**: 29 (A–Z + `del`, `nothing`, `space`)
- **Training data**: [ASL Alphabet Dataset – Kaggle](https://www.kaggle.com/datasets/grassknoted/asl-alphabet)

---

## 👨‍💻 Author

Built by **Shikhar** with AI assistance from **Antigravity (Google DeepMind)**.
