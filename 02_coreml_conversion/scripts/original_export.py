
from ultralytics import YOLO

# Load a YOLOv11 model
model = YOLO('yolo11n.pt')  # load an official model

# Export the model
model.export(format='coreml')
