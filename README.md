# CNN-Based Image Colorization Using Deep Learning

**A Deep Learning Project | Convolutional Neural Networks | CelebA Dataset**

## 1. Project Overview

This project aims to develop an automated image colorization system that converts grayscale images into color images using Convolutional Neural Networks (CNNs).

We will use the **CelebA (CelebFaces Attributes Dataset)**, which contains more than 200,000 celebrity facial images.

The model will learn to predict color information from grayscale images by studying the relationship between original color images and their grayscale counterparts.

The main objective is to develop and compare two CNN architectures:

1. Basic CNN (Baseline Model)
2. U-Net CNN (Primary Model)

The final generated images will be compared against their corresponding original color images to evaluate the performance of the models.

---

## 2. Problem Statement

Grayscale images contain brightness and structural information but do not contain explicit color information.

The objective is to develop a supervised Deep Learning model that can predict plausible colors from grayscale images.

The model will take a grayscale image as input, extract its features using CNN layers, and generate the missing color information.

**Input:** Grayscale image

**Output:** Predicted color image

**Goal:** Generate visually realistic color images while preserving the structure and details of the original image.

---

## 3. Dataset: CelebA

We will use the CelebA dataset because it contains a large collection of human facial images with variations in:

- Skin tones.
- Hair colors and hairstyles.
- Facial expressions.
- Facial structures.
- Backgrounds and lighting.

### Dataset Preparation

We will create paired training samples from the original RGB images.

Each original image will be converted into grayscale and LAB representations.

| Image component | Purpose |
|---|---|
| Original RGB | Ground truth |
| L channel | CNN input |
| a channel | First color target |
| b channel | Second color target |

### Dataset Split

| Dataset | Percentage |
|---|---:|
| Training | 80% |
| Validation | 10% |
| Testing | 10% |

The split will be performed before training to ensure that test images remain unseen.

---

## 4. Color Space: LAB

Instead of predicting all three RGB channels, we will use the CIELAB color space.

LAB consists of three channels:

- **L:** Lightness information.
- **a:** Green-to-red color information.
- **b:** Blue-to-yellow color information.

### Why LAB?

RGB combines brightness and color information across three channels.

LAB separates brightness from color, allowing the CNN to focus on predicting the missing color information.

### Our Approach

1. Convert the original RGB image into LAB.
2. Extract the L channel.
3. Provide the L channel to the CNN.
4. Predict the a and b channels.
5. Combine the original L channel with predicted a and b.
6. Convert the reconstructed LAB image into RGB.

Formula:

`Final LAB = [Original L, Predicted a, Predicted b]`

The reconstructed LAB image is then converted into RGB to generate the final colorized image.

---

## 5. CNN Models and Architecture

We will implement two CNN models to understand and compare their performance.

### Model 1: Basic CNN (Baseline)

The Basic CNN will act as our baseline model.

It will use multiple convolutional layers to extract image features and predict the missing color channels.

Main components:

- Convolutional layers (Conv2D).
- ReLU activation.
- Batch Normalization.
- Feature extraction layers.
- Output convolution layer.

**Input:** 128 × 128 × 1

**Output:** 128 × 128 × 2

Purpose: Establish a baseline against which the U-Net architecture can be evaluated.

### Model 2: U-Net CNN (Primary Model)

Our main architecture will be a custom U-Net-based encoder-decoder CNN with skip connections.

U-Net is suitable for image-to-image tasks because it extracts high-level features while preserving spatial information.

It consists of four major components:

**A. Encoder**

Extracts image features such as facial structures, edges, textures, hair and skin regions.

**B. Bottleneck**

Captures deeper and more complex image representations.

**C. Decoder**

Gradually reconstructs the image and generates the predicted color channels.

**D. Skip Connections**

Transfer features from encoder layers to corresponding decoder layers, helping preserve spatial details during image reconstruction.

### Proposed U-Net Architecture

| Stage | Operation | Output Dimensions |
|---|---|---|
| Input | Grayscale | 128 × 128 × 1 |
| Encoder 1 | Double Conv, 32 filters | 128 × 128 × 32 |
| Encoder 2 | Double Conv, 64 filters | 64 × 64 × 64 |
| Encoder 3 | Double Conv, 128 filters | 32 × 32 × 128 |
| Encoder 4 | Double Conv, 256 filters | 16 × 16 × 256 |
| Bottleneck | Double Conv, 512 filters | 8 × 8 × 512 |
| Decoder 1 | Upsampling + Skip | 16 × 16 × 256 |
| Decoder 2 | Upsampling + Skip | 32 × 32 × 128 |
| Decoder 3 | Upsampling + Skip | 64 × 64 × 64 |
| Decoder 4 | Upsampling + Skip | 128 × 128 × 32 |
| Output | 1 × 1 Conv, 2 filters | 128 × 128 × 2 |

Each convolution block contains two 3 × 3 convolutional layers, Batch Normalization and ReLU activation.

Max Pooling is used for downsampling, while upsampling and skip connections are used for image reconstruction.

---

## 6. Technologies Used

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| PyTorch | CNN architecture and model training |
| OpenCV | Image processing and LAB conversion |
| NumPy | Numerical operations |
| Matplotlib | Image visualization and training graphs |
| Scikit-image | Image evaluation metrics |
| Google Colab | GPU-based model training |
| GitHub | Version control and project documentation |

**Why PyTorch?**

PyTorch allows us to build custom CNN architectures, implement training loops, perform backpropagation and manage model parameters.

It also helps us understand the internal working of the neural network rather than relying entirely on pre-trained models.

---

## 7. Model Training

The project uses **Supervised Learning** because we have both grayscale input images and their corresponding original color targets.

### Training Process

1. Load a batch of grayscale images.
2. Load their corresponding original LAB color targets.
3. Pass the L channel through the CNN.
4. Predict the a and b channels.
5. Calculate the prediction error.
6. Perform backpropagation.
7. Update model weights using the Adam optimizer.
8. Repeat the process across multiple batches and epochs.
9. Monitor validation loss and save the best model.

### Training Configuration

| Parameter | Initial Value |
|---|---|
| Framework | PyTorch |
| Image Resolution | 128 × 128 |
| Input Channels | 1 |
| Output Channels | 2 |
| Optimizer | Adam |
| Learning Rate | 0.001 |
| Loss Function | L1 Loss |
| Batch Size | 16 or 32 |
| Epochs | 30–50 |
| Activation | ReLU |
| Normalization | Batch Normalization |

These are initial settings and may be adjusted based on validation performance and available GPU resources.

### Loss Function: L1 Loss

L1 Loss measures the absolute difference between the predicted and original color values.

`L1 Loss = Mean(|Actual Color - Predicted Color|)`

A lower loss indicates that the predicted color values are closer to the target values.

---

## 8. Complete Project Workflow

Our implementation will follow these stages:

**Step 1: Dataset Collection**
- Download CelebA.
- Inspect original RGB images.

**Step 2: Preprocessing**
- Resize images to 128 × 128.
- Convert RGB to LAB.
- Extract L as input and ab as target.
- Normalize the channels.

**Step 3: Dataset Splitting**
- 80% Training.
- 10% Validation.
- 10% Testing.

**Step 4: Model Development**
- Implement Basic CNN.
- Implement U-Net CNN.
- Verify input and output dimensions.

**Step 5: Model Training**
- Configure L1 Loss and Adam.
- Train both models.
- Monitor training and validation loss.
- Save model checkpoints.

**Step 6: Image Generation**
- Load an unseen grayscale image.
- Predict the missing color channels.
- Reconstruct LAB.
- Convert LAB into RGB.

**Step 7: Model Evaluation**
- Compare both CNN architectures.
- Compare predictions against original images.
- Calculate MAE, PSNR and SSIM.

**Step 8: Final Demonstration**
- Display original, grayscale and colorized images.
- Present training graphs and evaluation results.

---

## 9. Project Functions and Modules

The repository will be divided into modules for easier development and maintenance.

| File | Main Responsibility |
|---|---|
| `prepare_dataset.py` | Dataset preprocessing |
| `color_space.py` | RGB/LAB conversion |
| `celebA_dataset.py` | PyTorch Dataset and DataLoader |
| `blocks.py` | Reusable convolution blocks |
| `basic_cnn.py` | Basic CNN architecture |
| `unet.py` | U-Net architecture |
| `train.py` | Model training |
| `trainer.py` | Training and validation loops |
| `loss.py` | Loss calculation |
| `predict.py` | Image colorization |
| `evaluate.py` | Model evaluation |
| `metrics.py` | MAE, PSNR and SSIM |
| `visualization.py` | Image comparisons and graphs |
| `compare_models.py` | Basic CNN vs U-Net comparison |

### Important Functions

- `rgb_to_lab()` – Converts RGB images into LAB.
- `lab_to_rgb()` – Converts reconstructed LAB images into RGB.
- `__getitem__()` – Loads grayscale inputs and corresponding color targets.
- `DoubleConv()` – Implements two convolutional layers.
- `forward()` – Performs CNN forward propagation.
- `train_one_epoch()` – Trains the model for one epoch.
- `validate_one_epoch()` – Evaluates validation performance.
- `predict_color_channels()` – Predicts a and b channels.
- `colorize_image()` – Generates the final colorized image.
- `calculate_mae()` – Calculates pixel-level error.
- `calculate_psnr()` – Measures reconstruction fidelity.
- `calculate_ssim()` – Measures structural similarity.

---

## 10. Evaluation and Model Comparison

We will compare the original image with the generated color image using visual and numerical evaluation.

### Visual Comparison

For each selected test image, we will display:

1. Original RGB image.
2. Grayscale input image.
3. Basic CNN prediction.
4. U-Net prediction.

### Evaluation Metrics

| Metric | Purpose | Better Result |
|---|---|---|
| MAE | Measures average pixel error | Lower |
| PSNR | Measures reconstruction fidelity | Higher |
| SSIM | Measures structural similarity | Higher |

All metrics will be calculated on reconstructed RGB images using the same pixel range and the held-out test dataset.

### Experimental Comparison

The main experiment will investigate:

**Does a U-Net architecture with skip connections produce better colorization results than a basic CNN?**

Both models will use the same dataset split and evaluation conditions.

---

## 11. GitHub Repository Structure

```text
CNN-Image-Colorization/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── configs/
│   └── config.yaml
│
├── preprocessing/
│   ├── prepare_dataset.py
│   └── color_space.py
│
├── data/
│   └── celebA_dataset.py
│
├── models/
│   ├── basic_cnn.py
│   ├── unet.py
│   └── blocks.py
│
├── training/
│   ├── train.py
│   ├── trainer.py
│   ├── validate.py
│   └── loss.py
│
├── testing/
│   ├── predict.py
│   ├── evaluate.py
│   └── compare_models.py
│
├── utils/
│   ├── metrics.py
│   ├── visualization.py
│   └── checkpoint.py
│
├── notebooks/
│   ├── dataset_exploration.ipynb
│   └── model_experiments.ipynb
│
├── outputs/
│   ├── comparisons/
│   ├── colorized_images/
│   └── training_graphs/
│
└── checkpoints/
    └── README.md
```

The original CelebA dataset and large trained model files will be stored separately rather than committed directly to GitHub.

---

## 12. Limitations and Future Scope

### Limitations

- The model is primarily trained for facial image colorization.
- Exact original colors cannot always be recovered from grayscale images.
- Predicted colors may differ from the original colors.
- Results depend on dataset quality and training configuration.
- Low-resolution images may lose fine details.

### Future Improvements

- Increase image resolution.
- Experiment with perceptual loss.
- Explore attention mechanisms.
- Experiment with GAN-based colorization.
- Test on general-purpose images.
- Develop an interactive image colorization application.

---

## 13. Expected Final Results

The final project will deliver:

- A complete CelebA preprocessing pipeline.
- A custom Basic CNN model.
- A custom U-Net-based CNN model.
- Trained model checkpoints.
- A grayscale-to-color image prediction pipeline.
- Original vs grayscale vs predicted image comparisons.
- Training and validation graphs.
- MAE, PSNR and SSIM evaluation.
- A comparative analysis of both CNN architectures.

### Conclusion

This project demonstrates how Convolutional Neural Networks can learn image features and predict missing color information through supervised learning.

By comparing a Basic CNN with a U-Net-based encoder-decoder architecture, we aim to understand how different CNN designs affect image colorization performance.

The primary objective is to develop a working image colorization system and evaluate its ability to generate visually plausible color images from grayscale inputs.
