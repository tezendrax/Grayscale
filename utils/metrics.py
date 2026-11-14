"""
Evaluation Metrics for Image Colorization
Includes MAE, PSNR, and SSIM calculations.
"""

from typing import Dict, Union, Tuple
import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def to_numpy_image(image: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
    """Helper to convert torch tensor or numpy array to uint8/float numpy array."""
    if isinstance(image, torch.Tensor):
        img_np = image.detach().cpu().numpy()
        if img_np.ndim == 4:
            # (B, C, H, W) -> (B, H, W, C)
            img_np = img_np.transpose(0, 2, 3, 1)
        elif img_np.ndim == 3 and img_np.shape[0] in [1, 3]:
            # (C, H, W) -> (H, W, C)
            img_np = img_np.transpose(1, 2, 0)
    else:
        img_np = np.asarray(image)
    return img_np


def calculate_mae(
    pred: Union[np.ndarray, torch.Tensor],
    target: Union[np.ndarray, torch.Tensor]
) -> float:
    """
    Computes Mean Absolute Error (MAE) between predicted and target images.
    
    Args:
        pred: Predicted RGB image [0, 255] or [0, 1]
        target: Ground truth RGB image in same scale
        
    Returns:
        MAE value (float)
    """
    p = to_numpy_image(pred).astype(np.float64)
    t = to_numpy_image(target).astype(np.float64)
    return float(np.mean(np.abs(p - t)))


def calculate_psnr(
    pred: Union[np.ndarray, torch.Tensor],
    target: Union[np.ndarray, torch.Tensor],
    data_range: float = 255.0
) -> float:
    """
    Computes Peak Signal-to-Noise Ratio (PSNR) between predicted and ground truth.
    
    Args:
        pred: Predicted RGB image
        target: Ground truth RGB image
        data_range: Maximum possible value (255 for uint8, 1.0 for float)
        
    Returns:
        PSNR value in decibels (dB)
    """
    p = to_numpy_image(pred)
    t = to_numpy_image(target)
    
    # Auto-detect data range if uint8
    if p.dtype == np.uint8 or t.dtype == np.uint8 or np.max(p) > 1.0:
        data_range = 255.0
        p = np.clip(p, 0, 255).astype(np.uint8)
        t = np.clip(t, 0, 255).astype(np.uint8)
    else:
        data_range = 1.0
        p = np.clip(p, 0.0, 1.0).astype(np.float64)
        t = np.clip(t, 0.0, 1.0).astype(np.float64)

    return float(peak_signal_noise_ratio(t, p, data_range=data_range))


def calculate_ssim(
    pred: Union[np.ndarray, torch.Tensor],
    target: Union[np.ndarray, torch.Tensor],
    data_range: float = 255.0
) -> float:
    """
    Computes Structural Similarity Index (SSIM) between predicted and ground truth.
    
    Args:
        pred: Predicted RGB image
        target: Ground truth RGB image
        data_range: Maximum possible value (255 for uint8, 1.0 for float)
        
    Returns:
        SSIM value between -1 and 1 (higher is better)
    """
    p = to_numpy_image(pred)
    t = to_numpy_image(target)

    if p.dtype == np.uint8 or t.dtype == np.uint8 or np.max(p) > 1.0:
        data_range = 255.0
        p = np.clip(p, 0, 255).astype(np.uint8)
        t = np.clip(t, 0, 255).astype(np.uint8)
    else:
        data_range = 1.0
        p = np.clip(p, 0.0, 1.0).astype(np.float64)
        t = np.clip(t, 0.0, 1.0).astype(np.float64)

    # Calculate multichannel SSIM
    if p.ndim == 3 and p.shape[2] == 3:
        return float(structural_similarity(t, p, channel_axis=2, data_range=data_range))
    elif p.ndim == 2:
        return float(structural_similarity(t, p, data_range=data_range))
    else:
        # Fallback for batch
        ssims = []
        for i in range(len(p)):
            ssims.append(structural_similarity(t[i], p[i], channel_axis=-1, data_range=data_range))
        return float(np.mean(ssims))


def calculate_all_metrics(
    pred: Union[np.ndarray, torch.Tensor],
    target: Union[np.ndarray, torch.Tensor]
) -> Dict[str, float]:
    """
    Computes MAE, PSNR, and SSIM between predicted and target images.
    """
    return {
        "MAE": calculate_mae(pred, target),
        "PSNR": calculate_psnr(pred, target),
        "SSIM": calculate_ssim(pred, target),
    }
