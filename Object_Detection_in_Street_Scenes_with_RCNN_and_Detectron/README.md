# Object Detection in Street Scenes Video with Faster R-CNN and Detectron

<img width="857" alt="image" src="https://github.com/user-attachments/assets/33dacb34-937e-45cd-8438-98fa948f465b" />


This repository provides code and instructions to evaluate object detection for detecting vehicles in a **short street scene video** using two pre-trained models: Torchvision’s Faster R-CNN and Facebook AI’s Detectron2. The workflow includes using pre-extracted frames from videos to generate object-detected outputs (identify (classify) and draw bounding boxes) and creating new annotated videos based on the predictions.

## Requirements
- Python 3.x
- Torchvision library
- Detectron2 library
- OpenCV (cv2)
- NumPy

**Video For OD** - **LA_street_test.mp4**

## Setup
Ensure you have the required libraries installed:

1. **Install Detectron2:**
   ```bash
   pip install 'git+https://github.com/facebookresearch/detectron2.git'
   ```

2. **Additional Dependencies:**
   Install any missing Python libraries (e.g., OpenCV, NumPy) using:
   ```bash
   pip install opencv-python-headless numpy
   ```

## Evaluation Workflow
The project processes video frames using the following workflow:

1. Extract video frames and store them in a specified input folder.
2. Perform object detection using both Torchvision’s and Detectron2’s pre-trained models.
3. The Model outputs bounding box coordinates, class labels, and confidence scores.
4. Generate annotated output frames and merge them back into a video.

### 1. Pre-Trained Model: Torchvision’s Faster R-CNN

**Key Steps:**
- Load a pre-trained Faster R-CNN model from Torchvision.
- Adjust category labels for COCO dataset compatibility.
- Process frames and draw bounding boxes with category labels.

**Code Overview:**
- Load the model using `torchvision.models.detection.fasterrcnn_resnet50_fpn`.
- Define category names and color schemes.
- Iterate through input frames, apply predictions, and annotate.
- Save the annotated video.

**Usage:**
Set the following parameters:
```python
input_folder = 'path_of_the_raw_video_folder'
output_video_path = 'output_torchvision.mp4'
```
Run the function:
```python
generate_video_from_frames_torchvision(input_folder, output_video_path, model)
```

### 2. Pre-Trained Model: Detectron2

**Key Steps:**
- Install and configure Detectron2.
- Use a pre-trained Detectron2 Faster R-CNN model for inference.
- Redefine COCO category labels to ensure index alignment with Torchvision’s.
- Process frames, apply predictions, and annotate.

**Code Overview:**
- Load the model configuration using `model_zoo.get_config_file()`.
- Set model weights and thresholds for predictions.
- Adjust the `MODEL_DEVICE` to `cuda` or `cpu` based on availability.
- Process frames to detect and annotate objects.
- Merge annotated frames into a new video.

**Usage:**
Set the following parameters:
```python
input_folder = 'path_of_the_raw_video_folder'
output_video_path = 'output_detectron2.mp4'
```
Run the function:
```python
generate_video_from_frames_detectron(input_folder, output_video_path, predictor)
```

## Results
For both models, the results will include:
- Annotated frames saved in the specified output folder.
- A final video combining these frames.The generated video is saved at a specified path (e.g., `LA_street_output.mp4`) with a consistent frame rate for smooth playback.
  
|LA_Street_Output_from_Fast R-CNN_TorchVision|LA_Street_Output_from_Detectron2|
|---|---|
|<img width="417" alt="image" src="https://github.com/user-attachments/assets/b1da9c29-cea9-4ae8-bf9d-f8c4729b0bb0" />|<img width="417" alt="image" src="https://github.com/user-attachments/assets/a02dd430-5b5d-4d60-a466-fdbc3ae5e136" />| 

## Notes
- **Detectron2 Configuration:** By default, Detectron2 expects a GPU for processing. If unavailable, set `cfg.MODEL.DEVICE` to `'cpu'`.
- **Frame Input/Output:** Ensure the input folder contains extracted frames in `.jpg` format and specify a valid path for the output video.
- **COCO Dataset Alignment:** Both models are configured to use COCO dataset categories but handle index alignment differently. Ensure proper configuration for accurate label mapping.

## Conclusion

This repository offers a user-friendly and practical solution for object detection, leveraging pre-trained models to simplify tasks like video annotation and enabling users to seamlessly integrate these tools into real-world applications.




