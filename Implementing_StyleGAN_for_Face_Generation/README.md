# Implementing StyleGAN from scratch for Face Generation

This project implements StyleGAN, NVIDIA's groundbreaking architecture for generating high-quality, photorealistic images. StyleGAN's unique, style-based design allows for control over image features like face shape, expressions, and details, offering incredible flexibility in generative modeling.

## Key Features

**Generator & Discriminator:** Style-based architecture for realistic image synthesis.

**Progressive Growing:** Gradual resolution increases during training for stability.

**Latent Space Exploration:** Manipulate features like age, gender, and expressions to understand and control the generative process.

**Evaluation Metrics:** Assess image quality and diversity using metrics like FID.

## Dataset

[Dataset link](https://github.com/eshita-ds/Deep-Learning-Projects/tree/main/Implementing_StyleGAN_for_Face_Generation/STYLEGAN)

![image](https://github.com/user-attachments/assets/f6520bc9-0f6f-45de-b5e0-a58c6a32c85a)



## Requirements

- Python 3.9 or higher
- Libraries
  - Pandas
  - Numpy
  - Scikit-Learn
  - Pytorch
  - PIL (Python Imaging Library) - Facilitates image loading, manipulation, and visualization
  - OS - Manages file operations and organizes resources effectively.
  - TQDM: Provides progress bars for training and preprocessing tasks.

## Implementation

  ### Data Loading
  
  - A custom data loader (FolderDataset) loads images from a specified directory.
  - Preprocessing includes resizing, normalization, and augmentations like random horizontal flipping.
  - Images are dynamically batched based on resolution during progressive training

  ### Model Definitions
  
  - Mapping Network: Maps latent vector z to intermediate latent space w.
  - Generator: Implements noise injection, AdaIN, and progressive growing for stable, high-quality synthesis.
  - Discriminator: Uses weighted-scaled convolution, minibatch standard deviation, and progressive layers for robust feature extraction.

  ### Training Process
  
  - Implements WGAN-GP for stable training:
    - Generator Loss: Fool the discriminator into classifying fake images as real.
    - Discriminator Loss: Differentiate between real and fake images while enforcing the Lipschitz constraint with a gradient penalty.
  - Tracks losses for both generator and discriminator.
    
   ![image](https://github.com/user-attachments/assets/568260aa-9b24-4b92-94b2-aa8ba5907a77)



  ### Evaluation
  
  - FID Score: Evaluates image quality by comparing distributions of real and generated images.

 ![image](https://github.com/user-attachments/assets/7da770bd-25a0-494b-ba60-fd1c4b52809c)

    
  - PPL: Assesses the smoothness of latent space by measuring perceptual similarity between interpolated images.

  ### Results and Testing
  
  - Final FID Score: 382.7496337890625at step 5 with 45 epochs after tuning hyperparameters.
  - ![image](https://github.com/user-attachments/assets/4717db4a-2dc2-46d8-b0c6-cc5857bd980a)

  - PPL evaluated across different batch sizes to ensure smooth latent space.
  - Saved models tested on unseen data to validate performance.
    
  ![image](https://github.com/user-attachments/assets/4b4d0811-1593-40d2-be91-290671b38230)

  ![image](https://github.com/user-attachments/assets/cf32ade9-7e51-41f3-aff0-c1f608fe3a78)

  ### Model Saving and Outputs
  - Models, configurations, and metrics are saved for reproducibility.
  - Sample images generated and stored for each resolution step.

### Reference

**Paper: "A Style-Based Generator Architecture for Generative Adversarial Networks".**
