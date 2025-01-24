<h1 align=center>Object Detection and Inference using Yolo and Detectron</h1>

<p align=center><img width="840" alt="image" src="https://github.com/user-attachments/assets/7d955f22-defb-4f95-a2b0-760086b0d751" /></p>


This repository demonstrates the process of object detection using the YOLOv8 model(You Only Look Once, Version 8) leveraging Ultralytics' tools and Detectron2, for training, validation, and inference. The primary objective of this project was to train a YOLOv8 model and Detectron2 to recognize tools such as hammers, pliers, ropes, screwdrivers, toolboxes, and wrenches.

## YOLOv8 Object Detection 

`Object_Detection_and_inference_on_tools_dataset_using_Yolo.ipynb`

## Features
- **Custom Dataset Training**: Prepares and trains YOLOv8 on datasets with specific classes and bounding boxes.
- **Real-time Inference**: Supports fast and efficient detection on test images and videos.

## Prerequisites
### Libraries and Dependencies
Ensure the following Python libraries are installed:
- `ultralytics`: Core library for YOLOv8.
- `opencv-python`: For image and video processing.
- `matplotlib`: For visualizing results.

Install YOLOv8 directly from PyPI:
```bash
pip install ultralytics
```

### Dataset Preparation
Your dataset should be in YOLO format, containing:
- Images: Stored in subdirectories (`train`, `val`, `test`).
- Labels: Text files with class IDs and bounding box coordinates.

---

# Detectron2 Object Detection 

## Overview
This 'Object_Detection_and_inference_on_tools_dataset_using_Detectron.ipynb' demonstrates object detection using **Detectron2**, a Facebook AI framework for dense prediction tasks. The implementation involves training a Faster R-CNN model on a custom COCO-format dataset, evaluating its performance, and visualizing predictions.

## Features
- **Detectron2 Integration**: Leverages the powerful Detectron2 library for efficient and accurate object detection using the Faster R-CNN architecture.
- **Custom Dataset Usage**: Trains the model on a dataset formatted in COCO-style, including annotations for object bounding boxes and labels.
- **Custom Training and Evaluation**: Includes validation during training to ensure robust performance.
- **Visualization Tools**: Tools to visualize training datasets, annotations, and inference results.

## Prerequisites
### Libraries and Dependencies
Ensure the following Python libraries are installed:
- `torchvision`: For handling vision-related operations.
- `torch`: The core deep learning framework.
- `pycocotools`: For working with COCO-format datasets.
- `roboflow`: For downloading datasets directly from Roboflow.

Install Detectron2 directly from GitHub:
```bash
pip install 'git+https://github.com/facebookresearch/detectron2.git'
```

### Dataset
Dataset Link - `https://universe.roboflow.com/pxrksuhn/aihub-aizqc/dataset/1`

The dataset must be in COCO format. 
```python
from roboflow import Roboflow
rf = Roboflow(api_key="your_api_key")
project = rf.workspace("workspace_name").project("project_name")
version = project.version(1)
dataset = version.download("coco")
```

---

# Setup and Execution

## YOLOv8 Setup
### 1. Training
Train the YOLOv8 model on your dataset:
```python
from ultralytics import YOLO
model = YOLO('yolov8n.pt')
model.train(data='data.yaml', epochs=50, imgsz=640)
```

### 2. Inference
Run inference on test images or videos:
```python
model.predict(source='test.jpg', save=True, conf=0.5)
```

## Detectron2 Setup
### 1. Dataset Registration
Register the dataset directories for Detectron2:
```python
from detectron2.data.datasets import register_coco_instances
register_coco_instances("train_dataset", {}, "train_annotations.json", "train_images_path")
```

### 2. Training
Train the Faster R-CNN model:
```python
from detectron2.engine import DefaultTrainer
trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()
```

---

# Results and Visualization

## YOLOv8
- **Inference Results**: Fast and efficient object detection with YOLOv8 on test images.
- **Visualization**: Bounding boxes and labels displayed on test data.

|Validation batch results|Inferencing on Test set|
|---|---|
|<img width="432" alt="image" src="https://github.com/user-attachments/assets/dd17b2d7-8575-463a-9a40-7378515939c3" />|<img width="373" alt="image" src="https://github.com/user-attachments/assets/81c6f78d-60b3-46ae-baa3-03894268e890" />| 



## Detectron2
- **Metrics**: Evaluate precision, recall, and mean average precision (mAP).
- **Visualization**: Display annotated training datasets and inference results.
- <img width="579" alt="image" src="https://github.com/user-attachments/assets/de67b0bc-82c8-457e-9921-98b161c2f078" />

---

# Additional Notes

- **Hardware Support**: Both frameworks are optimized for GPU acceleration but support CPU usage with reduced performance.
- **Dataset Preparation**: Ensure datasets adhere to the respective formats for seamless integration.
- **Performance Tuning**: Experiment with learning rates, batch sizes, and augmentations for optimal results.


