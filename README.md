# ✍️ AirWrite — Hand Gesture Drawing in the Air

> Draw, erase, save, and undo — entirely with your hand. No mouse. No stylus. Just gestures.

AirWrite is a real-time, computer vision–powered drawing application that tracks your hand through a webcam using **MediaPipe** and lets you paint on a virtual canvas overlaid on your live camera feed. Switch brushes, cycle colors, undo strokes, and save your artwork — all without touching a keyboard (except when you want to).

---

## 📸 Features

- **6 brush types** — Pen, Marker, Spray, Glow, Neon, Eraser
- **9 color palette** with gesture-based cycling
- **Undo / Clear / Save** via intuitive hand gestures
- **Live hand skeleton overlay** rendered on camera feed
- **Sidebar UI** showing brush type, color palette, size slider, gesture guide, and keyboard shortcuts
- **Toast notifications** for every action
- Works at **720p, 540p, 480p, or 240p** — auto-detects the best resolution your webcam supports

---

## ⚠️ Python Version Compatibility

> **This is the most important thing to read before installing.**

| Python Version | Supported |
|---|---|
| 3.10 | ✅ Fully supported |
| 3.11 | ✅ Fully supported |
| 3.12 | ✅ Fully supported |
| 3.13 | ❌ **Not supported** — MediaPipe has no 3.13 build yet |

MediaPipe wheels are not yet published for Python 3.13. If you run `pip install mediapipe` on Python 3.13 you will get:

```
ERROR: Could not find a version that satisfies the requirement mediapipe
```

**Solution:** Use Python 3.12 or lower. The recommended approach is to create a dedicated virtual environment:

```bash
# Check your version first
python --version

# If 3.13, install 3.12 from https://www.python.org/downloads/
# Then create a virtual environment with it explicitly:
py -3.11 -m venv venv          # Windows
python3.11 -m venv venv        # macOS / Linux
```

---

## 🛠️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/airwrite.git
cd airwrite
```

### 2. Create and activate a virtual environment

**Windows**
```bash
py -3.11 -m venv venv
venv\Scripts\activate
```

**macOS / Linux**
```bash
python3.11 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run

```bash
python airwrite.py
```

A window titled **AirWrite | Hand Gesture Drawing** will open. Hold your hand in front of the webcam and you're ready.

---

## 🖐️ Gesture Reference

All gestures are detected in real time. The sidebar on the left highlights the currently active gesture.

| Gesture | Hand Shape | Action |
|---|---|---|
| ☝️ **Draw** | Index finger only | Paint on canvas |
| ✌️ **Pause** | Index + Middle fingers | Lift the pen (stop drawing) |
| 🖐️ **Erase** | Open palm (4 fingers) | Erase at fingertip |
| 👍 **Next Color** | Thumbs up | Cycle to next color |
| 🤘 **Save** | Index + Pinky (Rock On) | Save canvas as PNG |
| 🤙 **Clear** | Thumb + Pinky (Shaka) | Clear entire canvas |
| 🖖 **Undo** | Index + Middle + Ring (3 fingers) | Undo last stroke |

### Tips for reliable gesture detection

- Keep your hand **30–60 cm** from the camera
- Make sure your hand is **well-lit** — avoid backlighting
- **Hold gestures steady** for ~0.8–2 seconds (debounce is intentional to prevent accidental triggers)
- **Clear (Shaka)** has a 2-second debounce — hold 🤙 for a full 2 seconds to confirm

---

## ⌨️ Keyboard Shortcuts

| Key | Action |
|---|---|
| `B` | Cycle to next brush type |
| `C` | Cycle to next color |
| `+` / `=` | Increase brush size |
| `-` | Decrease brush size |
| `U` | Undo last stroke |
| `S` | Save canvas as PNG |
| `Q` / `Esc` | Quit |

---

## 🎨 Brush Types

| Brush | Effect |
|---|---|
| **Pen** | Clean anti-aliased line with opacity |
| **Marker** | Wide semi-transparent stroke |
| **Spray** | Scattered particle spray effect |
| **Glow** | Soft blurred halo around the stroke |
| **Neon** | Multi-layered bright neon light effect |
| **Eraser** | Clears pixels back to black |

---

## 💾 Saved Files

Images are saved in the **same directory as `airwrite.py`** with the filename format:

```
airwrite_YYYYMMDD_HHMMSS.png
```

Example: `airwrite_20241215_143022.png`

---

## 🐛 Known Issues & Fixes

### ❌ `mediapipe` fails to install

**Cause:** You are on Python 3.13 (unsupported) or an ARM-based system without a compatible wheel.

**Fix:**
```bash
# Confirm your Python version
python --version

# Downgrade to Python 3.12 and recreate your virtual environment
py -3.12 -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

---

### ❌ `numpy` compatibility error with MediaPipe

**Cause:** NumPy 2.x introduced breaking API changes. MediaPipe 0.10.x requires NumPy < 2.0.

**Error message looks like:**
```
A module that was compiled using NumPy 1.x cannot be run in NumPy 2.x
```

**Fix:** The `requirements.txt` pins `numpy<2.0.0` to prevent this. If you installed numpy separately, downgrade it:
```bash
pip install "numpy<2.0.0"
```

---

### ❌ No camera found / Black window

**Cause:** OpenCV cannot access your webcam. This can be a permission issue or an index issue.

**Fix:**
- **macOS:** Go to System Settings → Privacy & Security → Camera → enable for Terminal / your IDE
- **Windows:** Go to Settings → Privacy → Camera → allow desktop apps
- **Linux:** Add your user to the `video` group:
  ```bash
  sudo usermod -aG video $USER
  # Then log out and back in
  ```
- If you have multiple cameras (e.g., laptop + external), try changing `cv2.VideoCapture(0)` to `cv2.VideoCapture(1)` in `airwrite.py`

---

### ❌ Gestures not detecting / Wrong gesture triggered

**Cause:** Poor lighting, hand too far or too close, or background clutter confusing the model.

**Fixes:**
- Ensure your hand is **evenly lit from the front** — no strong backlight behind you
- Keep hand **within 30–70 cm** of the camera
- Use a **plain, uncluttered background** if detection is inconsistent
- Avoid wearing **gloves or rings** that obscure landmark positions
- If **Draw** triggers when you don't want it to, slow down and form gestures more deliberately

---

### ❌ High CPU usage / Low frame rate

**Cause:** MediaPipe's hand model is CPU-intensive, especially at `model_complexity=1`.

**Fix:** In `airwrite.py`, change `model_complexity` from `1` to `0`:
```python
hands = mp_hands.Hands(
    model_complexity=0,   # ← change from 1 to 0 for faster performance
    ...
)
```
This trades a small amount of accuracy for significantly better frame rate on slower machines.

---

### ❌ `cv2` module not found

**Cause:** OpenCV is not installed, or installed outside your virtual environment.

**Fix:**
```bash
# Make sure your venv is activated, then:
pip install opencv-python
```

---

## 📁 Project Structure

```
airwrite/
│
├── airwrite.py          # Main application
├── requirements.txt     # Python dependencies
└── README.md            # This file
```

Saved drawings appear in the same folder as `airwrite.py`.

---

## 🧰 Dependencies

| Package | Version | Purpose |
|---|---|---|
| `opencv-python` | >=4.8, <5.0 | Camera capture, drawing, display |
| `mediapipe` | >=0.10, <0.11 | Real-time hand landmark detection |
| `numpy` | >=1.24, <2.0 | Canvas array operations |

All other imports (`math`, `time`, `datetime`) are part of Python's standard library.

---

## 🤝 Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you'd like to change.

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).