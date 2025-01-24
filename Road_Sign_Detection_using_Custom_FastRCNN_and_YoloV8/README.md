<h1 align=center>Road Sign detection using Custom FastRCNN and Yolo V8</h1>

<p align=center><img width="650" alt="image" src="https://github.com/user-attachments/assets/00125c44-7e6e-4dda-981a-d6133686de10" /></p>


## Introduction
This project focuses on object detection for road signs using a combination of custom-built and pre-trained models, including YOLOv8. The Road Sign Dataset, was used for training, testing, and validation. Necessary data transformation and augmentation steps were performed to improve model generalization, and random samples from the train, test, and validation sets were plotted to ensure data integrity.

We implemented a custom object detection model from scratch, computed the Intersection over Union (IoU) of the predicted results with the test set, and visualized a few predictions. Additionally, the YOLOv8n pre-trained model was fine-tuned on the dataset, experimenting with various hyperparameters and data augmentations to optimize performance. IoU values, performance metrics, and results from different configurations were tabulated and visualized to evaluate the models’ efficiency.

The project also includes a comprehensive report detailing the choice of models, hyperparameters, and observations, along with the IoU results. This repository serves as a robust starting point for further exploration of object detection tasks using advanced models and custom implementations.

## Dataset

Dataset is a repository of road signs collected from various sources that is split among training and test datasets. Dataset has ground truth labels and bounding boxes for both test and train datasets. The dataset has the following folders:
- Labels_test – AnnotaCons in XML format
- Labels_train – AnnotaCons in XML format
- Train/images – Images for training
- Train/Labels – Labels of train set in Yolo Format (txt File with 5 entries, class, and bounding box co-ordinates)
- Test/images - Images for training
- Test/labels – Labels of test set in Yolo Format (txt File with 5 entries, class, and bounding box co-ordinates)
- Data.yaml – which has paths for the test and train folders and it has class labels.

From “data.yaml” we extracted the unique classes with their names. The unique classes are as follows: **Number of classes: 25**

**Road Sign Dataset:** `https://drive.google.com/drive/folders/1LIceJIn69vzmn40eAqz5kYJrsrC6m5lc?usp=sharing`

**Sample Images:**

<img width="757" alt="image" src="https://github.com/user-attachments/assets/6a60ab6b-12ce-4bf0-ab07-9ff3fd7dc9c6" />


## Data Pre-processing

To prepare the images for input to the custom Fast R-CNN model, the following transformations were applied to the training dataset. Each transformation was carefully selected to enhance the model’s generalization and adaptability.

**Training Transformations:**
- Resize: Each image was resized to a fixed resolution of 224x224 pixels to ensure uniformity across the dataset, which is crucial for the model’s convolutional layers.
- RandomPerspective (distortion scale=0.2, p=0.5): This introduces random perspective distortions, with a probability of 0.5. It Mimics perspective variations caused by different camera angles and distances. This allows the model to learn from slight distortions in shape.
- RandomRotation (±15 degrees): Images were randomly rotated by up to (+)(-)15 degrees, helping the model generalize to signs that may appear at slight angles. Here it is crucial to limit the rotation to 10 to 15 degrees only, so that it should keep the core orientation recognizable.
- RandomGrayscale (p=0.1): This transformation converts a random portion of images to grayscale with a probability of 0.1, enhancing the model’s resilience to color changes or poorly colored images. This simulates conditions where color information is limited (e.g., low lighting or monochrome imaging). This transformation encourages the model to rely on shape and structure rather than color alone.
- ColorJitter (brightness=0.2, contrast=0.2): Randomly adjusts brightness and contrast by a factor of ±20%. Helps the model adapt to varying lighting conditions, such as shadows, glare, or dim lighting, by introducing a range of brightness and contrast variations. This prevents the model from overfitting to specific lighting conditions present in the training data.
- GaussianBlur (kernel size=3, sigma=0.1-1.5): Applies a Gaussian blur with a random kernel size of 3 and sigma range of 0.1 to 1.5. Reduces high-frequency noise and simulates slight camera blurring, making the model more resilient to blurry images. This simulates conditions where images may not be sharply focused, improving robustness in suboptimal conditions.
- Normalization: Normalizes pixel values with mean = [0.485, 0.456, 0.406] and standard deviation = [0.229, 0.224, 0.225]. This normalization step standardizes the input data, centering the values and helping the model converge faster during training.
- ToTensor: The images were converted into PyTorch-compatible tensors using ToTensor, ensuring compatibility with the Fast R-CNN model.
- 
**Validation and Test Transformations:** For the test data, minimal transformations were applied to ensure that model evaluation reflects real-world performance.
- Resize: Similar to training, standardizes image dimensions to 224x224 for compatibility with the model’s input layer.
- Normalization: Normalized with the same mean and standard deviation as in training, ensuring consistency between training and test data.
- ToTensor: Converted to tensors for model compatibility.


## Modeling

### Custom Fast-RCNN
We implemented a Fast-RCNN model with Residual Blocks from scratch. The architecture involves passing an input image through a backbone CNN to extract features, generating a feature map, and using selective search to create regional proposals. These proposals are resized using an ROI Pooling layer and passed to Fully Connected layers to output classes and bounding box coordinates.

Residual Blocks in the backbone CNN improve object detection by adding shortcut connections, enabling gradients to flow efficiently and addressing the vanishing gradient problem. This accelerates convergence and reduces training time. A strong backbone with residual blocks enhances feature extraction, capturing detailed and abstract features for identifying objects in complex backgrounds.

<img width="731" alt="image" src="https://github.com/user-attachments/assets/8104e1e5-ad78-485c-b490-896bf9f4ebf7" />

The graphs above show the Training and Validation Loss (left) and Accuracy (right) over 100 epochs for the object detection model. Losses decreased sharply early on and plateaued, indicating the model generalized well with the current architecture. Accuracy steadily rose and stabilized around 81–82%, suggesting the model reached its learning capacity.

While the model performed stably, further improvements may require architectural changes, more complex layers, or pre-trained backbones for better results.

### Yolo V8

- Baseline Pass

  - For the baseline pass, we utilized the pre-trained YOLOv8n model, chosen for its speed, efficiency, and accuracy, making it particularly well-suited for detecting small objects and real-time applications. The training was conducted with default parameters over 10 epochs on the road sign detection dataset. The dataset was split into training, validation, and test sets, with the validation set containing the same number of images as the test set (~12% of the training set). The model was run in training mode, utilizing the YOLOv8n.pt weights, and verbose output was disabled to display only essential information. This initial setup aimed to evaluate the model’s performance without any customizations.

- Final Pass

  - In the final pass, the training configuration was significantly enhanced to improve model performance. The number of epochs was increased to 50 to allow the model to learn more extensively. The image size remained at 832 pixels, and the optimizer was fine-tuned with an initial learning rate of 0.0005 and a final learning rate of 0.00001 using cosine annealing scheduling. Weight decay (0.001) was added for regularization, and a warmup period over the first three epochs helped stabilize the training process. Advanced data augmentation techniques were employed, including label smoothing (0.05), dropout (0.1), mosaic augmentation (0.4), and mixup augmentation (0.2). Additional transformations included adjustments to hue, saturation, and brightness, along with rotation (±10°), translation (20%), scaling (30%), shearing (20%), and minor perspective distortions (0.0005). The IoU threshold for non-max suppression was set to 0.6, and a confidence threshold of 0.5 ensured the model considered only reliable detections. Early stopping with a patience of 20 epochs was implemented to halt training if validation metrics did not improve. The number of workers for data loading was increased to eight for faster processing, and plots of training metrics were generated for visualization. These enhancements aimed to boost the model’s accuracy and generalization capabilities.
 
| Baseline Pass | Final Pass |
| --- | --- |
|<img width="400" alt="image" src="https://github.com/user-attachments/assets/3d5deb6e-42f2-4146-9c7c-df9daf1fd19d" />|<img width="400" alt="image" src="https://github.com/user-attachments/assets/1434b2ce-ee67-4f35-9c2f-3296a7ef3a7e" />|


## Conclusion

### Custom Fast-RCNN

<img width="547" alt="image" src="https://github.com/user-attachments/assets/f0085cf0-ae48-4ecf-8ded-650af02a3f46" />


The custom Fast RCNN model achieved a test accuracy of 55.18%, showing moderate competence in recognizing road signs but significant room for improvement. High-performing classes like “Attention Please” and “Uneven Road” achieved precision and recall above 0.85, while low-performing classes like “Speed limit” and “Crosswalk” had precision and recall below 0.30.

The macro F1-score of 0.59 and weighted F1-score of 0.55 highlight class imbalance, while the low Mean IoU of 0.1948 indicates limitations in localization accuracy. These results suggest potential for improvement through architectural changes, better handling of class imbalance, or enhanced training strategies.

### Yolo V8
The final YOLOv8n model, optimized in Pass 3, demonstrated the best performance for road sign detection, striking a balance between F1 score and recall. The IoU for the test set was 0.8759, indicating the model’s strong ability to predict bounding boxes accurately. Incorporating data augmentations, weight decay, higher epochs, and other hyperparameters significantly improved performance, achieving a class accuracy of 95.79% and an F1 score of 0.9572. False positives were reduced by 36.4% compared to the baseline, while recall remained high at 0.9824, and precision reached 0.9333. Despite minor trade-offs in IoU and confidence scores, the configuration proved robust and reliable for road sign detection. These model weights were used for predictions on the test dataset for the Kaggle competition.

<img width="226" alt="image" src="https://github.com/user-attachments/assets/fcb94b9f-d5eb-42c9-a8e3-275909737304" />

## License
MIT License

Copyright (c) 2024 Eshita Gupta

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
