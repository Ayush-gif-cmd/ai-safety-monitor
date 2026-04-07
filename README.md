# AI-Based Safety Monitoring System (Gloves Detection)

A real-time Computer Vision system designed to monitor safety compliance by detecting whether individuals are wearing safety gloves.

## Features
- **Real-Time Detection:** Uses OpenCV and MediaPipe for high-speed, lightweight hand bounding-box detection.
- **Glove Verification:** Applies an HSV skin-color segmentation heuristic inside the hand region. If bare skin is detected, it flags a violation.
- **Audio Alerts:** Automatically triggers a warning sound when gloves are missing.
- **Event Logging:** Captures the video frame and logs the timestamp in a `.csv` file upon violation.
- **Interactive UI:** Built with Streamlit for a clean, accessible control panel.

## Prerequisites
- Python 3.8+
- Webcam access

## Installation

1. Clone or download this repository.
2. Open a terminal in the project directory.
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. (Optional) Re-generate the alert sound if needed:
   ```bash
   python create_sound.py
   ```

## Usage

1. Start the Streamlit application:
   ```bash
   streamlit run app.py
   ```
2. A browser window will open automatically.
3. In the right panel, check the "Turn On Camera" box to begin monitoring.
4. The system will outline detected hands. Green indicates safe (gloves detected), and red indicates a violation (bare hand detected).

## Architecture Details

This repository uses **MediaPipe** for hand detection rather than YOLOv8. 
*Why?* Standard pre-trained YOLOv8 models do not reliably detect hands natively without a custom dataset. MediaPipe is the industry standard for lightweight, zero-shot hand tracking. 
However, the codebase is completely modular. The `HandGloveDetector` class in `core/detector.py` can be easily swapped out for an `Ultralytics` custom YOLOv8 model once a PPE dataset is available.

## Author
AI Assistant
