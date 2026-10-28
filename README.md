# Human Sense AI

**Real-time human activity and gesture recognition using computer vision.**

Human Sense AI is a Python desktop prototype developed during an AI vision hackathon. It uses a live camera feed and landmark-based visual processing to explore human pose, hand gestures, and basic activity cues.

> **Project status:** Hackathon prototype — intended for experimentation and learning, not as a validated or production-ready activity recognition system.

## What it does

- Captures live video through OpenCV.
- Uses MediaPipe perception components to process hand and pose landmarks.
- Calculates geometric features from detected landmarks.
- Estimates basic activity cues, such as standing or sitting, using heuristic rules.
- Provides a desktop interface built with Tkinter.

**Important:** The current activity logic includes hand-crafted geometric rules. It should not be described as a separately trained or validated activity-classification model unless that implementation is added and evaluated.

## Technology

| Component | Role |
| --- | --- |
| Python | Application logic |
| OpenCV | Camera capture and image processing |
| MediaPipe | Hand and pose landmark detection |
| NumPy | Numerical operations |
| Tkinter | Desktop interface |

## Run on Windows

1. Clone or download this repository.
2. Open the project folder in VS Code.
3. Set up a Python environment and install the dependencies required by the code.
4. Confirm that the required model assets are present in the locations expected by the application.
5. Run **Launch Human Sense AI.bat**.

The launcher expects a project-local virtual environment at **.venv**. If your environment uses a different folder or the required packages are missing, update the launcher or run the application from your configured environment.

The project entry point is **src/gesture_demo.py**.

## Repository layout

- **src/gesture_demo.py** — main camera and landmark-processing application
- **src/geture_demo.py** — compatibility entry point
- **camera_test.py** — basic camera test
- **Launch Human Sense AI.bat** — Windows launcher

## Limitations

- Detection quality depends on camera quality, lighting, framing, and landmark visibility.
- The current activity estimates are heuristic and have not been presented as benchmarked model results.
- Hardware and dependency compatibility may vary across machines.
- No accuracy, latency, or dataset-performance claims are made without a reproducible evaluation.

## Roadmap

- Document exact installation steps and pinned dependencies.
- Add a short demo and screenshots.
- Record the source and license of each model asset and any dataset used.
- Add tests and clearer handling for missing models or unavailable cameras.
- Evaluate activity recognition against a documented dataset and report reproducible metrics.

## Credits and asset licensing

This project uses third-party computer-vision libraries and may require model assets. Check the applicable licenses and document the exact asset sources before redistributing or deploying the project.

## Author

**Vanshaj Awasthi**
