import torch
from pathlib import Path

def train_yolo(data_yaml, weights="yolov5s.pt", img_size=416, batch_size=1, epochs=10):
    from yolov5 import train

    # Start training
    train.run(
        data=data_yaml,
        weights=weights,
        imgsz=img_size,
        batch_size=batch_size,
        epochs=epochs,
        project=Path("runs/train"),
        name="idd_yolo",
        exist_ok=True
    )
