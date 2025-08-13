**IDD Object Detection using YOLOv5**
**Project Overview**

This project implements an object detection pipeline for the Indian Driving Dataset (IDD) using YOLOv5.
The main objectives are:

Loading and preprocessing the dataset

Converting annotations to YOLO format

Training and validating a YOLOv5 model

Note: Due to system constraints (CPU, limited RAM), we preprocess and train on a subset of 300 images (250 for training, 50 for validation).

**Running the Project**
python main.py

What happens:

Preprocess a subset of 300 images due to system constraints.

Convert annotations from VOC format to YOLO format.

Split images into training (250) and validation (50) sets.

Train YOLOv5 on the dataset.

Training Parameters (CPU-friendly):

Image size: 416 × 416

Batch size: 1

Epochs: 10
