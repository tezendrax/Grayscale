"""
Visualization Utilities for Image Colorization
Generates comparison figures, image grids, and training loss curves.
"""

import os
from typing import List, Optional, Dict
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt


def plot_loss_curves(
    train_losses: List[float],
    val_losses: List[float],
    model_name: str = "model",
    save_path: Optional[str] = None
) -> None:
    """
    Plots and saves training and validation loss curves.
    """
    plt.figure(figsize=(8, 5))
    epochs = range(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, "b-o", label="Training Loss")
    plt.plot(epochs, val_losses, "r-s", label="Validation Loss")
    plt.title(f"{model_name.upper()} - Training & Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("L1 Loss")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
    plt.close()


def plot_side_by_side(
    original_rgb: np.ndarray,
    grayscale_l: np.ndarray,
    predicted_rgb: np.ndarray,
    model_name: str = "CNN",
    metrics: Optional[Dict[str, float]] = None,
    save_path: Optional[str] = None
) -> None:
    """
    Plots a 3-panel comparison: Original RGB | Grayscale (L) | Predicted RGB.
    """
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    # Original RGB
    axes[0].imshow(np.clip(original_rgb, 0, 255).astype(np.uint8))
    axes[0].set_title("Original RGB (Ground Truth)")
    axes[0].axis("off")

    # Grayscale L
    if grayscale_l.ndim == 3 and grayscale_l.shape[-1] == 1:
        grayscale_l = grayscale_l.squeeze(-1)
    axes[1].imshow(grayscale_l, cmap="gray")
    axes[1].set_title("Grayscale Input (L Channel)")
    axes[1].axis("off")

    # Predicted RGB
    axes[2].imshow(np.clip(predicted_rgb, 0, 255).astype(np.uint8))
    title = f"{model_name} Prediction"
    if metrics:
        metrics_str = f"PSNR: {metrics.get('PSNR', 0.0):.2f}dB | SSIM: {metrics.get('SSIM', 0.0):.3f}"
        title += f"\n{metrics_str}"
    axes[2].set_title(title)
    axes[2].axis("off")

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_comparison_grid(
    original_rgb: np.ndarray,
    grayscale_l: np.ndarray,
    basic_cnn_rgb: np.ndarray,
    unet_rgb: np.ndarray,
    basic_metrics: Optional[Dict[str, float]] = None,
    unet_metrics: Optional[Dict[str, float]] = None,
    save_path: Optional[str] = None
) -> None:
    """
    Plots the 4-panel comparison requested in README Section 10:
    1. Original RGB
    2. Grayscale input
    3. Basic CNN prediction
    4. U-Net prediction
    """
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    # 1. Original RGB
    axes[0].imshow(np.clip(original_rgb, 0, 255).astype(np.uint8))
    axes[0].set_title("1. Original RGB\n(Ground Truth)")
    axes[0].axis("off")

    # 2. Grayscale input
    if grayscale_l.ndim == 3 and grayscale_l.shape[-1] == 1:
        grayscale_l = grayscale_l.squeeze(-1)
    axes[1].imshow(grayscale_l, cmap="gray")
    axes[1].set_title("2. Grayscale Input\n(L Channel)")
    axes[1].axis("off")

    # 3. Basic CNN
    axes[2].imshow(np.clip(basic_cnn_rgb, 0, 255).astype(np.uint8))
    title_basic = "3. Basic CNN (Baseline)"
    if basic_metrics:
        title_basic += f"\nPSNR: {basic_metrics.get('PSNR', 0.0):.2f}dB | SSIM: {basic_metrics.get('SSIM', 0.0):.3f}"
    axes[2].set_title(title_basic)
    axes[2].axis("off")

    # 4. U-Net
    axes[3].imshow(np.clip(unet_rgb, 0, 255).astype(np.uint8))
    title_unet = "4. U-Net (Primary Model)"
    if unet_metrics:
        title_unet += f"\nPSNR: {unet_metrics.get('PSNR', 0.0):.2f}dB | SSIM: {unet_metrics.get('SSIM', 0.0):.3f}"
    axes[3].set_title(title_unet)
    axes[3].axis("off")

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
