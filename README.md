# CNN-Based Automatic Image Colorization

**Deep Learning Project | Image Processing | Convolutional Neural Networks**

## 1. Project Overview

This project focuses on developing a Deep Learning-based image colorization system that automatically converts grayscale facial images into color images using Convolutional Neural Networks (CNNs).

The project uses the **CelebA (CelebFaces Attributes Dataset)**, which contains more than 200,000 celebrity facial images.

The main objective is to train a CNN model to learn the relationship between grayscale image features and their corresponding color information.

The trained model will take a grayscale image as input and generate a predicted color image. The generated output will then be compared against the original RGB image to evaluate the model's performance.

### Project Objectives

- Develop an automated grayscale-to-color image conversion system.
- Understand and implement CNN-based image-to-image learning.
- Implement a basic CNN as a baseline model.
- Develop a U-Net-based encoder-decoder CNN as the primary model.
- Use LAB color space for more focused color prediction.
- Evaluate the generated images using quantitative and qualitative metrics.
- Compare the performance of different CNN architectures.

---

## 2. Problem Statement

Grayscale images contain brightness and structural information but lack explicit color information.

The objective is to develop a supervised Deep Learning model that can predict plausible color information from grayscale images.

Given a grayscale image, the CNN learns to predict the corresponding color channels using paired grayscale and original RGB images from the CelebA dataset.

Mathematically:

**Input:**

X = Grayscale image (L channel)

**Target:**

Y = Original color information (a and b channels)

**Prediction:**

Ŷ = CNN(X)

The predicted color channels are combined with the original lightness channel to reconstruct the final color image.

---

## 3. Dataset

### CelebA Dataset

The project uses the CelebA dataset, which contains more than 200,000 celebrity face images with facial attributes and variations.

Dataset characteristics:

- Diverse facial structures.
- Different skin tones.
- Different hairstyles and hair colors.
- Different facial expressions.
- Different lighting conditions.
- Variations in image backgrounds.

### Dataset Preparation

Instead of maintaining independent and potentially mismatched datasets, the original RGB images will be processed into corresponding LAB representations.

Each image will produce:

| Component | Description | Purpose |
|---|---|---|
| Original RGB | Original color image | Ground truth and final comparison |
| L channel | Lightness information | CNN input |
| a channel | Green-to-red information | CNN target |
| b channel | Blue-to-yellow information | CNN target |

Proposed image resolution: 128 × 128 pixels.

### Dataset Split

| Dataset | Percentage | Purpose |
|---|---:|---|
| Training | 80% | Model learning |
| Validation | 10% | Hyperparameter tuning |
| Testing | 10% | Final model evaluation |

The split will be performed at the image-filename level to prevent data leakage.

---

## 4. Color Space: CIELAB

The project uses the CIELAB color space instead of directly predicting RGB values.

LAB consists of three channels:

- L: Lightness.
- a: Green-to-red color information.
- b: Blue-to-yellow color information.

### Why LAB?

In RGB, all three channels contain a mixture of brightness and color information.

LAB separates lightness from chromatic information.

This allows the CNN to focus on predicting the missing color channels rather than reconstructing brightness and color simultaneously.

### Colorization Process

1. Load the original RGB image.
2. Convert RGB into LAB.
3. Extract the L channel.
4. Pass L into the CNN.
5. Predict a and b.
6. Combine the original L with predicted a and b.
7. Convert the reconstructed LAB image into RGB.

Mathematically:

LAB_predicted = [L_original, a_predicted, b_predicted]

Final_RGB = LAB_to_RGB(LAB_predicted)

The L channel is retained from the grayscale input to preserve the original brightness and structural information.

---

## 5. CNN Models and Architecture

The project will implement two CNN architectures.

### Model 1: Basic CNN (Baseline Model)

The Basic CNN will serve as the baseline for evaluating the performance of the proposed U-Net architecture.

It will consist of:

1. Input layer.
2. Convolutional layers.
3. ReLU activation.
4. Batch Normalization.
5. Feature extraction layers.
6. Output convolution layer.

The model takes a grayscale image and predicts two LAB color channels.

**Input:** 128 × 128 × 1

**Output:** 128 × 128 × 2

Purpose:

- Establish baseline performance.
- Understand basic convolutional feature extraction.
- Compare a simpler architecture against U-Net.

### Model 2: U-Net-Based CNN (Primary Model)

The primary model will use a U-Net-inspired encoder-decoder architecture with skip connections.

U-Net was originally developed for biomedical image segmentation. Its encoder-decoder structure is also useful for image-to-image tasks such as image colorization.

The architecture consists of three major components:

**A. Encoder**

The encoder extracts image features and gradually reduces spatial dimensions.

It learns:

- Facial structures.
- Hair patterns.
- Skin regions.
- Edges and textures.
- High-level image features.

**B. Bottleneck**

The bottleneck captures deeper and more complex feature representations.

**C. Decoder**

The decoder gradually reconstructs spatial information and predicts the missing color channels.

**D. Skip Connections**

Skip connections transfer feature information from the encoder to corresponding decoder layers.

They help preserve spatial details that may otherwise be lost during downsampling.

### Proposed U-Net Architecture

| Stage | Operation | Output Dimensions |
|---|---|---|
| Input | Grayscale image | 128 × 128 × 1 |
| Encoder 1 | Double Conv, 32 filters | 128 × 128 × 32 |
| Pool 1 | Max Pooling | 64 × 64 × 32 |
| Encoder 2 | Double Conv, 64 filters | 64 × 64 × 64 |
| Pool 2 | Max Pooling | 32 × 32 × 64 |
| Encoder 3 | Double Conv, 128 filters | 32 × 32 × 128 |
| Pool 3 | Max Pooling | 16 × 16 × 128 |
| Encoder 4 | Double Conv, 256 filters | 16 × 16 × 256 |
| Pool 4 | Max Pooling | 8 × 8 × 256 |
| Bottleneck | Double Conv, 512 filters | 8 × 8 × 512 |
| Decoder 1 | Upsampling + Skip Connection | 16 × 16 × 256 |
| Decoder 2 | Upsampling + Skip Connection | 32 × 32 × 128 |
| Decoder 3 | Upsampling + Skip Connection | 64 × 64 × 64 |
| Decoder 4 | Upsampling + Skip Connection | 128 × 128 × 32 |
| Output | 1 × 1 Conv, 2 filters | 128 × 128 × 2 |

### Convolution Block

Each convolution block will generally contain:

1. Conv2D (3 × 3).
2. Batch Normalization.
3. ReLU activation.
4. Conv2D (3 × 3).
5. Batch Normalization.
6. ReLU activation.

### Architecture Diagram

```text
             GRAYSCALE IMAGE
              128 x 128 x 1
                     |
                     v
              ENCODER BLOCK 1
              128 x 128 x 32
                     |
                 Max Pool
                     |
                     v
              ENCODER BLOCK 2
              64 x 64 x 64
                     |
                 Max Pool
                     |
                     v
              ENCODER BLOCK 3
              32 x 32 x 128
                     |
                 Max Pool
                     |
                     v
              ENCODER BLOCK 4
              16 x 16 x 256
                     |
                 Max Pool
                     |
                     v
                BOTTLENECK
                8 x 8 x 512
                     |
                     v
              DECODER BLOCK 1
              16 x 16 x 256
                     |
                     v
              DECODER BLOCK 2
              32 x 32 x 128
                     |
                     v
              DECODER BLOCK 3
              64 x 64 x 64
                     |
                     v
              DECODER BLOCK 4
              128 x 128 x 32
                     |
                     v
                OUTPUT LAYER
               128 x 128 x 2
                     |
                     v
              PREDICTED a + b
                     |
                     v
           COMBINE WITH ORIGINAL L
                     |
                     v
              LAB TO RGB
                     |
                     v
             COLORIZED IMAGE
```

Skip connections will connect each encoder stage to its corresponding decoder stage.

---

## 6. Technologies and Libraries

| Technology | Application |
|---|---|
| Python | Main programming language |
| PyTorch | CNN architecture and model training |
| OpenCV | Image processing and color-space conversion |
| NumPy | Numerical operations and array processing |
| Matplotlib | Image visualization and training graphs |
| Scikit-image | Image quality evaluation |
| Pillow | Image loading and saving |
| Google Colab | GPU-based development and training |
| Git and GitHub | Version control and project collaboration |

### Why PyTorch?

PyTorch will be used because it provides flexibility for:

- Building custom CNN architectures.
- Implementing encoder-decoder networks.
- Managing image tensors.
- Performing automatic differentiation.
- Implementing custom training loops.
- Saving and loading trained models.

---

## 7. Training Configuration

The models will be trained using supervised learning.

### Initial Hyperparameters

| Parameter | Initial Configuration |
|---|---|
| Framework | PyTorch |
| Input Resolution | 128 × 128 |
| Input Channels | 1 |
| Output Channels | 2 |
| Optimizer | Adam |
| Initial Learning Rate | 0.001 |
| Loss Function | L1 Loss |
| Batch Size | 16 or 32 |
| Epochs | 30–50 |
| Hidden Activation | ReLU |
| Output Activation | Tanh |
| Normalization | Batch Normalization |
| Learning Rate Scheduler | ReduceLROnPlateau |

The final hyperparameters will be selected based on validation performance and available GPU memory.

### Loss Function: L1 Loss

L1 Loss measures the absolute difference between predicted and actual color values.

L1 = Mean(|Target - Prediction|)

It will be used to optimize the predicted a and b channels.

### Optimizer: Adam

Adam will update the network weights based on gradients calculated through backpropagation.

### Training Process

1. Load a batch of grayscale images.
2. Load their corresponding LAB color targets.
3. Pass grayscale images through the CNN.
4. Generate predicted a and b channels.
5. Calculate L1 Loss.
6. Perform backpropagation.
7. Update model parameters using Adam.
8. Calculate validation loss.
9. Save the best-performing model checkpoint.
10. Repeat for the specified number of epochs.

---

## 8. Complete Project Workflow

The development process will be divided into seven phases.

### Phase 1: Dataset Collection and Preprocessing

- Download the CelebA dataset.
- Extract the original images.
- Inspect sample images.
- Resize images to 128 × 128.
- Convert RGB images into LAB.
- Extract the L channel as input.
- Extract a and b channels as targets.
- Normalize the image channels.
- Split the dataset into training, validation and testing.

**Deliverable:** Prepared paired image dataset.

### Phase 2: Basic CNN Development

- Define convolutional layers.
- Implement ReLU activation.
- Add Batch Normalization.
- Implement feature extraction.
- Create the two-channel output layer.
- Test the model's input and output dimensions.

**Deliverable:** Basic CNN baseline model.

### Phase 3: U-Net Architecture Development

- Implement convolution blocks.
- Build the encoder.
- Implement Max Pooling.
- Build the bottleneck.
- Implement decoder blocks.
- Add skip connections.
- Create the final output layer.
- Verify architecture and tensor dimensions.

**Deliverable:** Complete U-Net model.

### Phase 4: Model Training

- Load the prepared dataset.
- Define the loss function.
- Configure the Adam optimizer.
- Implement the training loop.
- Perform backpropagation.
- Monitor validation loss.
- Save model checkpoints.
- Plot training and validation loss.

**Deliverable:** Trained CNN models.

### Phase 5: Image Colorization

- Load the best trained model.
- Select unseen test images.
- Convert the original images to LAB.
- Extract grayscale input.
- Predict missing color channels.
- Reconstruct LAB images.
- Convert reconstructed LAB images into RGB.

**Deliverable:** Predicted color images.

### Phase 6: Evaluation and Comparison

Compare:

- Original RGB image.
- Grayscale input.
- Basic CNN prediction.
- U-Net prediction.

Evaluate using:

- Mean Absolute Error (MAE).
- Peak Signal-to-Noise Ratio (PSNR).
- Structural Similarity Index (SSIM).

**Deliverable:** Quantitative and visual model comparison.

### Phase 7: Final Documentation and Presentation

- Prepare architecture diagrams.
- Document preprocessing.
- Explain CNN components.
- Include training graphs.
- Present image comparisons.
- Include evaluation metrics.
- Discuss limitations and possible improvements.

**Deliverable:** Final project report and presentation.

---

## 9. Evaluation Metrics

The performance of both CNN architectures will be evaluated using the same unseen test dataset.

| Metric | Purpose | Expected Direction |
|---|---|---|
| MAE | Measures average pixel error | Lower is better |
| PSNR | Measures reconstruction fidelity | Higher is better |
| SSIM | Measures structural similarity | Higher is better |

The metrics will be calculated on reconstructed RGB images against their corresponding original RGB images, using consistent pixel ranges.

Both models will be evaluated under the same testing conditions.

---

## 10. GitHub Repository Structure

```text
CNN-Image-Colorization/
│
├── README.md
│
├── requirements.txt
│
├── .gitignore
│
├── configs/
│   └── config.yaml
│
├── dataset/
│   ├── README.md
│   └── samples/
│
├── preprocessing/
│   ├── __init__.py
│   ├── prepare_dataset.py
│   ├── color_space.py
│   └── dataset_split.py
│
├── data/
│   ├── __init__.py
│   └── celebA_dataset.py
│
├── models/
│   ├── __init__.py
│   ├── basic_cnn.py
│   ├── unet.py
│   └── blocks.py
│
├── training/
│   ├── __init__.py
│   ├── train.py
│   ├── trainer.py
│   ├── loss.py
│   └── validate.py
│
├── testing/
│   ├── __init__.py
│   ├── predict.py
│   ├── evaluate.py
│   └── compare_models.py
│
├── utils/
│   ├── __init__.py
│   ├── visualization.py
│   ├── metrics.py
│   └── checkpoint.py
│
├── notebooks/
│   ├── 01_dataset_exploration.ipynb
│   ├── 02_preprocessing.ipynb
│   └── 03_model_experiments.ipynb
│
├── checkpoints/
│   └── README.md
│
├── outputs/
│   ├── comparisons/
│   ├── colorized_images/
│   └── training_graphs/
│
└── reports/
    ├── architecture.md
    └── results.md
```

Note: The actual CelebA dataset, large generated images and trained model weights should not be committed to GitHub. Keep those locally or in an appropriate cloud storage location.

---

## 11. Module and Function Documentation

The following functions define the intended responsibilities of the different modules.

### A. preprocessing/prepare_dataset.py

Responsible for loading and preparing original images.

Functions:

- `load_images()` – Loads image paths from the CelebA dataset.
- `resize_image()` – Resizes images to the required resolution.
- `convert_to_lab()` – Converts RGB images to LAB color space.
- `normalize_channels()` – Normalizes the L and ab channels.
- `save_processed_data()` – Saves processed data when using an offline preprocessing approach.

### B. preprocessing/color_space.py

Responsible for color-space conversion.

Functions:

- `rgb_to_lab()` – Converts RGB images to LAB.
- `lab_to_rgb()` – Reconstructs RGB images from LAB.
- `extract_l_channel()` – Extracts lightness information.
- `extract_ab_channels()` – Extracts color information.
- `denormalize_ab()` – Converts predicted normalized channels back to the expected LAB range.

### C. preprocessing/dataset_split.py

Responsible for dataset splitting.

Functions:

- `split_dataset()` – Divides image filenames into training, validation and testing.
- `save_split_files()` – Saves the split information.
- `verify_splits()` – Checks for duplicate filenames and data leakage.

### D. data/celebA_dataset.py

Responsible for creating the PyTorch Dataset and DataLoader pipeline.

Functions:

- `__init__()` – Initializes image paths and transformations.
- `__len__()` – Returns the number of samples.
- `__getitem__()` – Loads an image and returns its grayscale input and corresponding color target.
- `get_dataloaders()` – Creates training, validation and testing DataLoaders.

### E. models/blocks.py

Contains reusable CNN building blocks.

Functions and classes:

- `DoubleConv()` – Two convolutional layers with Batch Normalization and ReLU.
- `DownBlock()` – Performs downsampling and feature extraction.
- `UpBlock()` – Performs upsampling and combines encoder features through skip connections.

### F. models/basic_cnn.py

Implements the baseline CNN.

Functions:

- `__init__()` – Defines convolutional layers.
- `forward()` – Performs forward propagation and generates predicted color channels.

### G. models/unet.py

Implements the primary U-Net architecture.

Functions:

- `__init__()` – Defines encoder, bottleneck, decoder and output layers.
- `forward()` – Passes inputs through the network and combines skip connections.
- `get_model()` – Creates and returns the U-Net model.

### H. training/loss.py

Responsible for loss calculation.

Functions:

- `calculate_l1_loss()` – Calculates the L1 error between predicted and actual color channels.
- `calculate_total_loss()` – Returns the training loss.

### I. training/trainer.py

Responsible for model optimization.

Functions:

- `train_one_epoch()` – Trains the model for one epoch.
- `validate_one_epoch()` – Evaluates validation loss.
- `optimizer_step()` – Performs gradient updates.
- `save_checkpoint()` – Saves model weights and optimizer state.

### J. training/train.py

Main training entry point.

Functions:

- `load_config()` – Loads training configurations.
- `initialize_model()` – Initializes the selected CNN.
- `initialize_optimizer()` – Configures Adam.
- `train_model()` – Executes the complete training loop.
- `main()` – Runs the training pipeline.

### K. testing/predict.py

Responsible for generating colorized images.

Functions:

- `load_trained_model()` – Loads the best model checkpoint.
- `preprocess_input()` – Converts input images into the required model format.
- `predict_color_channels()` – Predicts the a and b channels.
- `reconstruct_image()` – Combines L and predicted ab.
- `colorize_image()` – Executes the complete image colorization process.

### L. utils/metrics.py

Responsible for image quality evaluation.

Functions:

- `calculate_mae()` – Calculates Mean Absolute Error.
- `calculate_psnr()` – Calculates Peak Signal-to-Noise Ratio.
- `calculate_ssim()` – Calculates Structural Similarity.
- `evaluate_model()` – Generates evaluation results for the test dataset.

### M. utils/visualization.py

Responsible for generating visual results.

Functions:

- `display_comparison()` – Displays original, grayscale and predicted images.
- `plot_training_loss()` – Plots training and validation loss.
- `save_comparison()` – Saves comparison images.
- `plot_metric_comparison()` – Visualizes evaluation metrics.

### N. testing/compare_models.py

Responsible for comparing the Basic CNN and U-Net.

Functions:

- `load_models()` – Loads both trained models.
- `compare_predictions()` – Compares their predictions.
- `compare_metrics()` – Compares MAE, PSNR and SSIM.
- `generate_comparison_report()` – Produces a consolidated comparison report.

---

## 12. Expected Project Outputs

The final project should produce the following outputs:

### Visual Results

For every selected test image:

1. Original RGB image.
2. Grayscale image.
3. Basic CNN colorized image.
4. U-Net colorized image.

### Training Results

- Training loss curves.
- Validation loss curves.
- Model checkpoints.
- Training configuration.
- Training logs.

### Evaluation Results

- MAE comparison.
- PSNR comparison.
- SSIM comparison.
- Basic CNN vs U-Net performance table.

### Final Deliverables

- Working preprocessing pipeline.
- Two implemented CNN architectures.
- Trained model weights.
- Image colorization script.
- Model comparison results.
- Project documentation.
- Final presentation.

---

## 13. Limitations

- The model is trained primarily on human facial images and may not generalize to unrelated image categories.
- Grayscale images do not contain enough information to determine every original color uniquely.
- The generated colors may differ from the original colors.
- Model performance depends on training data quality and diversity.
- Low image resolution may reduce fine-detail reconstruction.
- Pixel-level losses may favor averaged or less vibrant colors.

## 14. Future Improvements

Possible extensions include:

- Increasing image resolution to 256 × 256 or higher.
- Experimenting with perceptual loss.
- Exploring attention mechanisms.
- Implementing a GAN-based colorization model.
- Testing on images outside the CelebA dataset.
- Developing an interactive image colorization interface.

---

## 15. Final Project Summary

The project will implement a supervised CNN-based image colorization system using CelebA.

Two architectures will be developed:

**1. Basic CNN:** Baseline model for comparison.

**2. U-Net CNN:** Primary encoder-decoder architecture with skip connections.

Both models will use grayscale L-channel inputs and predict the a and b color channels in LAB color space.

The predicted channels will be combined with the original lightness channel to reconstruct the final RGB image.

The models will be trained using L1 Loss and the Adam optimizer. Their performance will be evaluated using MAE, PSNR and SSIM, along with visual comparison against the original images.

**The primary research objective is to investigate whether U-Net's encoder-decoder architecture and skip connections improve facial image colorization compared with a basic CNN.**
