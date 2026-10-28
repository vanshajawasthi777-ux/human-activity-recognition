# Human Sense AI

**Real-time human activity and gesture recognition using computer vision.**

[![Python syntax check](https://github.com/vanshajawasthi777-ux/human-activity-recognition/actions/workflows/python-check.yml/badge.svg)](https://github.com/vanshajawasthi777-ux/human-activity-recognition/actions/workflows/python-check.yml)

Human Sense AI is a Python desktop prototype developed during an AI vision hackathon. It processes a live camera feed and uses visual landmarks to explore hand gestures, human pose, and basic activity cues.

> **Status:** Hackathon prototype for experimentation and learning. It has not been validated as a production-ready activity recognition system.

## Features

- Live camera capture and frame processing with OpenCV.
- Hand gesture and pose landmark processing with MediaPipe.
- Geometric feature calculations from detected landmarks.
- Heuristic estimates for basic activity cues, including standing and sitting.
- Desktop interface built with Tkinter.

> **Model note:** The current activity estimates use hand-crafted geometric rules. They are not evidence of a separately trained or benchmarked activity-classification model.

## Technology

| Technology | Purpose |
| --- | --- |
| Python | Application logic |
| OpenCV | Camera capture and image processing |
| MediaPipe | Hand gesture and pose landmarks |
| NumPy | Numerical operations |
| Tkinter | Desktop interface (included with many Python installations) |

## Getting started (Windows)

### 1. Clone the repository

```powershell
git clone https://github.com/vanshajawasthi777-ux/human-activity-recognition.git
cd human-activity-recognition
```

### 2. Create a virtual environment

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
```

### 3. Install dependencies

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### 4. Check model assets and launch

Make sure the model files expected by the application are present in the repository's `models` directory. Then run:

```powershell
.\Launch Human Sense AI.bat
```

The launcher expects the virtual environment at `.venv`. If installation fails, check the Python version supported by the installed MediaPipe release and review the error output.

## Repository structure

```text
human-activity-recognition/
├── .github/
│   ├── ISSUE_TEMPLATE/bug_report.md
│   ├── PULL_REQUEST_TEMPLATE.md
│   └── workflows/python-check.yml
├── assets/                     # Application assets
├── models/                     # Model assets expected by the application
├── src/
│   ├── gesture_demo.py         # Main application
│   └── geture_demo.py          # Compatibility entry point
├── camera_test.py              # Basic camera test
├── requirements.txt            # Runtime Python dependencies
├── CONTRIBUTING.md             # Contribution and validation guidance
├── SECURITY.md                 # Security and privacy reporting guidance
├── Launch Human Sense AI.bat   # Windows launcher
└── .gitignore
```

## Validation

GitHub Actions runs a Python syntax/compilation check for the source files on pushes and pull requests. This check does **not** install dependencies, launch the GUI, access a webcam, load model assets, or measure recognition quality. A passing check is not proof that the full application runs on every machine.

To run the same syntax check locally:

```powershell
.\.venv\Scripts\python.exe -m compileall -q src camera_test.py
```

## Limitations

- Results depend on camera quality, lighting, framing, and landmark visibility.
- Activity estimates are heuristic and have not been benchmarked against a labelled evaluation dataset.
- Performance and dependency compatibility can vary across machines.
- No accuracy or latency figures are claimed without reproducible testing.

## Planned improvements

- Verify and document supported Python and dependency versions.
- Add screenshots and a short demo recording from a real run.
- Document the source and license of every model asset and dataset.
- Add runtime tests and clearer handling for unavailable cameras or missing model files.
- Evaluate recognition quality using a documented dataset and reproducible metrics.

## Third-party assets

This project depends on third-party computer-vision packages and model assets. Check their respective licenses and redistribution conditions before reuse or deployment.

## Contributing

Please read [CONTRIBUTING.md](CONTRIBUTING.md) before opening a pull request. Report reproducible problems using the repository's bug report template.

## Author

**Vanshaj Awasthi**
