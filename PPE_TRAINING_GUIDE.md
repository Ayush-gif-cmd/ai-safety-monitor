# YOLOv8 Training & Implementation Guide for PPE Detection

This guide will explain how to train a custom YOLOv8 model for your AI Safety Monitoring System to detect Helmets/Hairnets and Gloves.

## Model Suggestions
For real-time camera detection on standard hardware, use the YOLOv8 nano (`yolov8n.pt`) or small (`yolov8s.pt`) models. They provide the best balance between speed (FPS) and accuracy.

### Dataset Suggestions
To train your model, you need a dataset that is annotated with bounding boxes for the following classes:
`['helmet', 'head', 'gloves', 'bare_hand', 'mask', 'bare_face', 'person']`
(You can customize these classes, but having representations of both protected and unprotected states yields best results).

**Where to find datasets:**
1. **Roboflow Universe (Recommended):** Search for "PPE Detection", "Hardhat and Vest", "Face Mask Detection", or "Gloves Detection". You can merge datasets to cover both head, face, and hand protection.
2. **Kaggle:** Search for "Safety Helmet and Face Mask Detection Dataset".
3. **Make your own:** Collect images from your webcam or workplace and annotate them manually using tools like [CVAT](https://www.cvat.ai/) or [LabelImg](https://github.com/HumanSignal/labelImg).

## Step-by-Step Training Process

### 1. Prepare Environment
Open your terminal and install Ultralytics (YOLOv8 framework):
```bash
pip install ultralytics
```

### 2. Format Your Dataset (YOLO format)
Ensure your dataset has a `data.yaml` file that looks something like this:
```yaml
train: ../train/images
val: ../valid/images

nc: 6
names: ['helmet', 'bare_head', 'gloves', 'bare_hand', 'mask', 'bare_face']
```

### 3. Start Training
Create a python script (`train.py`) or run this in a Jupyter Notebook:
```python
from ultralytics import YOLO

# Load a pre-trained model (starting with nano for fast training)
model = YOLO("yolov8n.pt") 

# Train the model
# Adjust epochs based on how quickly your model converges (typically 50-100)
results = model.train(
    data="path/to/your/data.yaml",
    epochs=50,
    imgsz=640,
    device="cpu" # Change to "mps" (Mac) or "0" (Nvidia GPU) if available
)
```

Run the script: `python train.py`

### 4. Locate Your Trained Model
Once training completes, your best weights will be saved in `runs/detect/train/weights/best.pt`.

## Integration with the Application

The `core/detector.py` has been updated to support custom YOLOv8 models gracefully. 
By default, the application uses MediaPipe + Color Segmentation heuristics (your previous working system).

**To activate YOLOv8:**
1. Rename your trained model to `ppe_model.pt`.
2. Move `ppe_model.pt` into the root directory of this project.
3. The app will automatically detect it upon startup and switch from MediaPipe to YOLOv8 for more accurate detection!

**How the adapted YOLOv8 logic works in our code:**
- It uses `ultralytics` to predict bounding boxes.
- We check the class (`cls`) inside each box.
- If it sees classes like "bare_head", "bare_face", or "bare_hand", it logs a violation ("Head Protection Missing", "Mask Missing", or "Gloves Missing").
- This fully integrates with the existing UI, Logging, and Alert systems!
