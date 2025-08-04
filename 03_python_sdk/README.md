
# YOLOv11 CoreML SDK

This SDK provides a simple way to use a YOLOv11 CoreML model for object detection.

## Installation

```bash
pip install .
```

## Usage

```python
from yolo_sdk import YOLOv11CoreML

# Initialize the model
model = YOLOv11CoreML()

# Run prediction
results = model.predict('https://ultralytics.com/images/bus.jpg', save=True)

print(results)
```
