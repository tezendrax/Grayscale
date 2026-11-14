"""
Validation Functions for Model Evaluation during Training
"""

from typing import Dict, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from utils.metrics import calculate_all_metrics
from preprocessing.color_space import lab_to_rgb


def validate_one_epoch(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    compute_metrics: bool = False
) -> Tuple[float, Dict[str, float]]:
    """
    Evaluates the model over the validation dataset for one epoch.
    
    Args:
        model: PyTorch model
        val_loader: DataLoader for validation data
        criterion: Loss function
        device: torch.device
        compute_metrics: Whether to evaluate MAE, PSNR, and SSIM
        
    Returns:
        Tuple of (average_loss, metrics_dict)
    """
    model.eval()
    total_loss = 0.0
    metrics_accumulator = {"MAE": 0.0, "PSNR": 0.0, "SSIM": 0.0}
    num_samples = 0

    with torch.no_grad():
        for batch in val_loader:
            l_t = batch["L"].to(device)
            ab_t = batch["ab"].to(device)
            rgb_t = batch["rgb"].to(device)

            pred_ab = model(l_t)
            loss = criterion(pred_ab, ab_t)
            total_loss += loss.item() * l_t.size(0)

            if compute_metrics:
                # Convert batch to RGB
                pred_rgb = lab_to_rgb(l_t, pred_ab)
                target_rgb = (rgb_t.permute(0, 2, 3, 1).cpu().numpy() * 255.0).astype(np.uint8)
                
                for i in range(len(pred_rgb)):
                    sample_metrics = calculate_all_metrics(pred_rgb[i], target_rgb[i])
                    for k in metrics_accumulator:
                        metrics_accumulator[k] += sample_metrics[k]

            num_samples += l_t.size(0)

    avg_loss = total_loss / max(1, num_samples)
    if compute_metrics and num_samples > 0:
        avg_metrics = {k: v / num_samples for k, v in metrics_accumulator.items()}
    else:
        avg_metrics = {}

    return avg_loss, avg_metrics
