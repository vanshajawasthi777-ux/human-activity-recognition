# Human Sense AI

A Python-based computer-vision prototype for human activity and gesture recognition, built for a time-limited AI vision hackathon.

## Overview

Human Sense AI explores how a camera feed can be processed to recognize human movement and gestures in real time. The project brings together a desktop interface and computer-vision components in one runnable prototype.

## Tech Stack

- **Python** — application logic
- **OpenCV (cv2)** — camera input and image/video processing
- **MediaPipe** — real-time perception and landmark-processing components
- **NumPy** — numerical operations
- **Tkinter** — desktop user interface

## Run the Project

1. Clone or download this repository.
2. Open the project folder in VS Code.
3. Make sure Python and the project's required packages are installed.
4. Check that any required model assets are present in the expected locations.
5. Run the project's launch script, `Launch Human Sense AI.bat`, on Windows, or launch the main Python entry point from your configured environment.

> **Note:** Exact setup commands depend on the dependency and model files included in the current project. Verify these files before running on a new machine.

## Project Status

This is a hackathon prototype. The code and setup may evolve as the project is refactored, documented, and tested on additional machines.

## Data and Model Transparency

Before using this project for research or redistribution, document the exact origin, version, and license of each dataset and pretrained model used. Do not assume that a library's pretrained model was trained on a dataset collected by this project. Dataset and model provenance should be verified against the actual assets and implementation.

## Future Improvements

- Split the application into focused modules
- Add a verified dependency list and reproducible setup instructions
- Document the model pipeline and dataset/model sources
- Add screenshots and a short demo
- Add tests and graceful handling for unavailable cameras or missing model assets

## Author

**Vanshaj Awasthi**

Built as a learning project to explore practical computer vision and AI application development.
