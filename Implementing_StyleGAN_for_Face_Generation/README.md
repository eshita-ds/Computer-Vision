# Implementing StyleGAN from scratch for Face Generation

This project implements StyleGAN, NVIDIA's groundbreaking architecture for generating high-quality, photorealistic images. StyleGAN's unique, style-based design allows for control over image features like face shape, expressions, and details, offering incredible flexibility in generative modeling.

## Key Features

**Generator & Discriminator:** Style-based architecture for realistic image synthesis.

**Progressive Growing:** Gradual resolution increases during training for stability.

**Latent Space Exploration:** Manipulate features like age, gender, and expressions to understand and control the generative process.

**Evaluation Metrics:** Assess image quality and diversity using metrics like FID.

## Dataset

[Dataset link](https://github.com/eshita-ds/Deep-Learning-Projects/tree/main/Implementing_StyleGAN_for_Face_Generation/STYLEGAN)

<img width="443" alt="image" src="https://github.com/user-attachments/assets/9233d610-68d9-4732-90ae-962b3c42474c" />


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
    <img width="275" alt="image" src="https://github.com/user-attachments/assets/04667c0d-0a98-4ea4-81f9-6f3db1324d8a" />


  ### Evaluation
  
  - FID Score: Evaluates image quality by comparing distributions of real and generated images.

  <img width="591" alt="image" src="https://github.com/user-attachments/assets/76e5cac1-20cb-447d-a84f-e256d3da4bc5" />
    
  - PPL: Assesses the smoothness of latent space by measuring perceptual similarity between interpolated images.

  ### Results and Testing
  
  - Final FID Score: 382.7496337890625at step 5 with 45 epochs after tuning hyperparameters.
  - <img width="317" alt="image" src="https://github.com/user-attachments/assets/1ac91cc9-d0dd-4494-be5a-bfb0c594c428" />
  - PPL evaluated across different batch sizes to ensure smooth latent space.
  - Saved models tested on unseen data to validate performance.
    
  <img width="257" alt="image" src="https://github.com/user-attachments/assets/a30f764d-7bb0-4324-8c3e-f4626eafb163" />
  <img width="250" alt="image" src="https://github.com/user-attachments/assets/6c72fcf9-2f2b-4cb3-9a2c-a31dfb09b0d6" />

  ### Model Saving and Outputs
  - Models, configurations, and metrics are saved for reproducibility.
  - Sample images generated and stored for each resolution step.

### Reference

**Paper: "A Style-Based Generator Architecture for Generative Adversarial Networks".**
