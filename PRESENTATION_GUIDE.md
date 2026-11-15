# Presentation & Screen Recording Master Guide: CNN Image Colorization

> **Purpose:** This document is designed for your **live presentation and screen recording in VS Code**. It breaks down the complete workflow, architectural arrow diagrams, every hyperparameter and layer configuration, evaluation metrics, and the exact commands to demonstrate.

---

## 1. High-Level Project Workflow (Arrow Diagram)

```
[1. CelebA Color Dataset] ──> [2. Grayscale Conversion Script] ──> [3. B&W Input Dataset (data/grayscale/)]
                                                                               │
┌──────────────────────────────────────────────────────────────────────────────┘
▼
[4. Data Preprocessing & LAB Transformation]
   ├── Resize to 128 x 128
   ├── Convert RGB ──> CIELAB
   ├── Extract L (Input) & ab (Targets)
   └── Normalize: L in [0, 1], ab in [-1, 1]
         │
         ▼
[5. Dataset Partitioning (80% Train | 10% Val | 10% Test)]
         │
         ▼
[6. Deep Learning Models] ──────────────────────┐
   ├── Baseline: Basic CNN (Sequential)        │
   └── Primary:  U-Net (With Skip Connections) │
         │                                      │
         ▼                                      ▼
[7. Training Engine]                     [8. Checkpointing]
   ├── Loss: L1 Loss                       ├── best_basic_cnn.pth
   ├── Optimizer: Adam (lr=0.001)          └── best_unet.pth
   └── Scheduler: ReduceLROnPlateau
         │
         ▼
[9. Color Reconstruction Pipeline]
   ├── Model predicts ab in [-1, 1]
   ├── Denormalize: L*100, ab*128
   └── Convert LAB ──> Reconstructed RGB [0, 255]
         │
         ▼
[10. Evaluation & Visual Verification]
   ├── Metrics: MAE (error), PSNR (fidelity dB), SSIM (texture similarity)
   └── 4-Panel Comparison: Original RGB vs Grayscale vs Basic CNN vs U-Net
```

---

## 2. Model Architecture Arrow Diagram (Inside the U-Net)

```
                 Input Lightness Tensor: L (1 x 128 x 128)
                                    │
                                    ▼
                ┌───────────────────────────────────────┐
                │ Encoder 1 (Inc): DoubleConv(1 -> 32)   │ ──(Skip Connection 1)──┐
                │ Shape: 32 x 128 x 128                 │                        │
                └───────────────────┬───────────────────┘                        │
                                    │ MaxPool2D(2, 2)                            │
                                    ▼                                            │
                ┌───────────────────────────────────────┐                        │
                │ Encoder 2 (Down1): DoubleConv(32 -> 64)│ ──(Skip Connection 2)──┼──┐
                │ Shape: 64 x 64 x 64                   │                        │  │
                └───────────────────┬───────────────────┘                        │  │
                                    │ MaxPool2D(2, 2)                            │  │
                                    ▼                                            │  │
                ┌───────────────────────────────────────┐                        │  │
                │ Encoder 3 (Down2): DoubleConv(64->128)│ ──(Skip Connection 3)──┼──┼──┐
                │ Shape: 128 x 32 x 32                  │                        │  │  │
                └───────────────────┬───────────────────┘                        │  │  │
                                    │ MaxPool2D(2, 2)                            │  │  │
                                    ▼                                            │  │  │
                ┌───────────────────────────────────────┐                        │  │  │
                │ Encoder 4 (Down3): DoubleConv(128->256│ ──(Skip Connection 4)──┼──┼──┼──┐
                │ Shape: 256 x 16 x 16                  │                        │  │  │  │
                └───────────────────┬───────────────────┘                        │  │  │  │
                                    │ MaxPool2D(2, 2)                            │  │  │  │
                                    ▼                                            │  │  │  │
                ┌───────────────────────────────────────┐                        │  │  │  │
                │ Bottleneck: DoubleConv(256 -> 512)    │                        │  │  │  │
                │ Shape: 512 x 8 x 8                    │                        │  │  │  │
                └───────────────────┬───────────────────┘                        │  │  │  │
                                    │ ConvTranspose2D(2, 2)                      │  │  │  │
                                    ▼                                            │  │  │  │
                ┌───────────────────────────────────────┐                        │  │  │  │
                │ Decoder 1 (Up1): Cat(512 + 256) -> 256│ <──────────────────────┘  │  │  │
                │ Shape: 256 x 16 x 16                  │                           │  │  │
                └───────────────────┬───────────────────┘                           │  │  │
                                    │ ConvTranspose2D(2, 2)                         │  │  │
                                    ▼                                               │  │  │
                ┌───────────────────────────────────────┐                           │  │  │
                │ Decoder 2 (Up2): Cat(256 + 128) -> 128│ <─────────────────────────┘  │  │
                │ Shape: 128 x 32 x 32                  │                              │  │
                └───────────────────┬───────────────────┘                              │  │
                                    │ ConvTranspose2D(2, 2)                            │  │
                                    ▼                                                  │  │
                ┌───────────────────────────────────────┐                              │  │
                │ Decoder 3 (Up3): Cat(128 + 64) -> 64  │ <────────────────────────────┘  │
                │ Shape: 64 x 64 x 64                   │                                 │
                └───────────────────┬───────────────────┘                                 │
                                    │ ConvTranspose2D(2, 2)                               │
                                    ▼                                                     │
                ┌───────────────────────────────────────┐                                 │
                │ Decoder 4 (Up4): Cat(64 + 32) -> 32   │ <───────────────────────────────┘
                │ Shape: 32 x 128 x 128                 │
                └───────────────────┬───────────────────┘
                                    │ 1 x 1 Conv2D(32 -> 2) + Tanh
                                    ▼
                Predicted ab Channels: (2 x 128 x 128) in [-1.0, 1.0]
```

---

## 3. Master Parameter Reference Table

| Category | Parameter Name | Value | Location in Code | Deep Learning Justification / Role |
|---|---|---|---|---|
| **Input** | Image Resolution | $128 \times 128$ | `configs/config.yaml` | Standard balance between spatial fidelity and CPU/GPU memory footprint |
| **Input** | Input Channels | $1$ ($L$) | `models/unet.py:27` | Lightness channel contains all edge and structural information |
| **Output** | Output Channels | $2$ ($a, b$) | `models/unet.py:27` | Color information is decoupled into green-red ($a$) and blue-yellow ($b$) |
| **Normalization** | $L$ Channel Scaling | $/ 100.0$ | `preprocessing/color_space.py:48` | Maps perceptual lightness $0-100$ into $[0.0, 1.0]$ for stable activations |
| **Normalization** | $a, b$ Scaling | $/ 128.0$ | `preprocessing/color_space.py:49` | Maps color space into $[-1.0, 1.0]$, exactly matching $\tanh$ bounds |
| **Convolution** | Kernel Size | $3 \times 3$ | `models/blocks.py:18` | Standard small receptive field capturing localized facial textures |
| **Convolution** | Stride & Padding | $s=1, p=1$ | `models/blocks.py:18` | Same-padding ensures spatial dimensions are preserved inside blocks |
| **Downsampling** | MaxPool Kernel & Stride | $k=2, s=2$ | `models/blocks.py:34` | Exact spatial halving ($128 \to 64 \to 32 \to 16 \to 8$) |
| **Upsampling** | ConvTranspose2d | $k=2, s=2$ | `models/blocks.py:49` | Learnable feature upsampling doubling resolution ($8 \to 16 \to 32 \to 64 \to 128$) |
| **Skip Fusion** | Concatenation Dim | $\text{dim}=1$ | `models/blocks.py:60` | Channel-wise stacking of encoder and decoder feature maps |
| **Activation** | Hidden Activation | `ReLU(inplace=True)` | `models/blocks.py:20` | Mitigates vanishing gradient and enables non-linear boundary learning |
| **Activation** | Final Activation | `nn.Tanh()` | `models/unet.py:48` | Guarantees bounded output in $[-1, 1]$ preventing out-of-gamut colors |
| **Normalization** | Batch Normalization | `BatchNorm2d` | `models/blocks.py:19` | Normalizes layer inputs across batch, stabilizing training convergence |
| **Training** | Loss Function | $L_1$ Loss | `training/loss.py:16` | $\text{Mean}(\|Actual - Predicted\|)$; avoids brownish/gray mean predictions |
| **Optimizer** | Optimizer | Adam | `training/train.py:35` | Adaptive moment estimation ($\beta_1=0.9, \beta_2=0.999$) |
| **Optimizer** | Learning Rate | $\eta = 0.001$ | `configs/config.yaml:23` | Optimal default for Adam on convolutional vision tasks |
| **Regularization**| Weight Decay | $10^{-5}$ | `configs/config.yaml:24` | $L_2$ penalty preventing exploding convolution weights |
| **Scheduling** | LR Scheduler | `ReduceLROnPlateau` | `training/train.py:38` | Factor $0.5$, patience $3$ epochs to fine-tune convergence |
| **Split** | Train / Val / Test | 80% / 10% / 10% | `configs/config.yaml:10` | Standard ML partition ensuring test set is strictly unseen |
| **Augmentation** | Random Horizontal Flip| $p=0.5$ | `data/celebA_dataset.py:41` | Teaches model facial symmetry and invariance to head orientation |

---

## 4. Step-by-Step Screen Recording Walkthrough in VS Code

Follow these exact steps when recording your project presentation in VS Code:

### Stage 1: Show the Folder Structure
1. Open the project root folder `DL_GreyScale` in VS Code.
2. In the Explorer pane, expand each folder and explain its responsibility:
   - `configs/` $\to$ Central hyperparameter configuration file.
   - `preprocessing/` $\to$ Scripts to clean data, convert to grayscale, and handle CIELAB conversions.
   - `data/` $\to$ The raw, processed, and grayscale datasets and PyTorch DataLoaders.
   - `models/` $\to$ Basic CNN (Baseline) and U-Net (Primary) architectures.
   - `training/` $\to$ Loss calculation, validation loops, and training engine.
   - `testing/` $\to$ Evaluation, prediction, and 4-panel comparison scripts.
   - `outputs/` $\to$ Visual comparisons, loss curves, and colorized outputs.
   - `checkpoints/` $\to$ Saved model weights (`best_unet.pth`, `best_basic_cnn.pth`).

### Stage 2: Demonstrate Grayscale Dataset Creation
Run the conversion script in the VS Code integrated terminal:
```powershell
python preprocessing/create_grayscale_dataset.py
```
**Talking Point:**
> *"Here we take original color CelebA images and convert them into our core Black & White dataset using perceptual luminance weighting: $Y = 0.299R + 0.587G + 0.114B$. The generated images are stored in `data/grayscale/`."*
Open `outputs/grayscale_samples/rgb_vs_grayscale.png` in VS Code to visually show the side-by-side color vs grayscale pairs.

### Stage 3: Explain the Model Architecture in VS Code
1. Open `models/unet.py` and show lines 25–50.
2. Point out:
   - 4-stage encoder (`inc`, `down1`, `down2`, `down3`)
   - 512-channel bottleneck
   - 4-stage decoder (`up1`, `up2`, `up3`, `up4`) with skip connections
   - Output $1 \times 1$ convolution with $\tanh$
3. Open `models/basic_cnn.py` and show that it lacks skip connections, establishing a baseline.

### Stage 4: Run Training Demonstration
Run the master training script:
```powershell
python training/train.py --model all --epochs 5 --batch-size 8
```
**Talking Point:**
> *"Notice the training log: in each epoch, the model computes the L1 color loss between predicted (a, b) channels and ground truth (a, b). When validation loss reaches a new low, the checkpoint is automatically updated."*

### Stage 5: Run Evaluation & Numerical Metrics
Run the evaluation script:
```powershell
python testing/evaluate.py --checkpoint checkpoints/best_unet.pth --model-type unet
```
**Talking Point:**
> *"We evaluate on the held-out test set across three standard metrics: MAE (average pixel error), PSNR (peak signal-to-noise ratio in decibels), and SSIM (structural similarity index)."*

### Stage 6: Show the 4-Panel Comparative Benchmark
Run the comparison benchmark:
```powershell
python testing/compare_models.py --num-samples 5
```
Open `outputs/comparisons/comparison_sample_1.png` in VS Code.
**Talking Point:**
> *"This 4-panel comparison demonstrates why U-Net is superior:*
> 1. *Panel 1: Ground truth color image.*
> 2. *Panel 2: Grayscale input.*
> 3. *Panel 3: Basic CNN output — notice the color bleeding across the face boundary and lower saturation.*
> 4. *Panel 4: U-Net output — the skip connections preserve high-resolution spatial boundaries, producing natural skin tones and crisp lip/hair borders."*

---

## 5. Deep Learning Evaluation Metrics Explained

### 1. Mean Absolute Error (MAE)
$$\text{MAE} = \frac{1}{3 \times H \times W} \sum_{c \in \{R,G,B\}} \sum_{i=1}^{H} \sum_{j=1}^{W} | I_{\text{original}}(c, i, j) - I_{\text{predicted}}(c, i, j) |$$
- **What it measures:** Average pixel-level difference in intensity values ($0-255$).
- **Our Result:** U-Net achieves **17.22**, an **18.66% reduction** over Basic CNN (21.17). Lower is better.

### 2. Peak Signal-to-Noise Ratio (PSNR)
$$\text{PSNR} = 10 \cdot \log_{10} \left( \frac{\text{MAX}_I^2}{\text{MSE}} \right)$$
where $\text{MAX}_I = 255.0$ and $\text{MSE} = \frac{1}{3HW} \sum (I_{\text{orig}} - I_{\text{pred}})^2$.
- **What it measures:** Peak reconstruction fidelity against error noise, expressed in decibels ($\text{dB}$).
- **Our Result:** U-Net reaches **19.93 dB**, outperforming Basic CNN (19.45 dB). Higher is better.

### 3. Structural Similarity Index (SSIM)
$$\text{SSIM}(x, y) = \frac{(2\mu_x\mu_y + c_1)(2\sigma_{xy} + c_2)}{(\mu_x^2 + \mu_y^2 + c_1)(\sigma_x^2 + \sigma_y^2 + c_2)}$$
- **What it measures:** Perceptual similarity considering luminance ($\mu$), contrast ($\sigma$), and structural correlation ($\sigma_{xy}$) across local windows.
- **Our Result:** U-Net achieves **0.8775** (where $1.0$ is perfect match), higher than Basic CNN (0.8655).

---

## 6. How to Run on Google Colab

If you want to run this in Google Colab:
1. Open Google Colab and set Runtime to **GPU** (`Runtime > Change runtime type > T4 GPU`).
2. Upload the repository or clone it:
   ```python
   !git clone https://github.com/tezendrax/Grayscale.git
   %cd Grayscale
   !pip install -r requirements.txt
   ```
3. Open `notebooks/colab_colorization_walkthrough.ipynb` directly in Colab and run each cell sequentially from **[STAGE 1]** through **[STAGE 9]**. Every cell has markdown explanations and self-contained logic!
