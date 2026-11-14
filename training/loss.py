"""
Loss Functions for Image Colorization
Supports L1 Loss (Primary), Smooth L1 Loss, and MSE Loss.
"""

import torch
import torch.nn as nn


class ColorizationLoss(nn.Module):
    """
    Computes loss between predicted color channels (ab) and ground truth channels (ab).
    Primary loss specified in README: L1 Loss = Mean(|Actual Color - Predicted Color|).
    """
    def __init__(self, loss_type: str = "L1"):
        super(ColorizationLoss, self).__init__()
        loss_type = loss_type.upper()
        if loss_type == "L1":
            self.criterion = nn.L1Loss()
        elif loss_type in ["SMOOTH_L1", "HUBER"]:
            self.criterion = nn.SmoothL1Loss()
        elif loss_type == "MSE":
            self.criterion = nn.MSELoss()
        else:
            raise ValueError(f"Unsupported loss type: {loss_type}. Choose from 'L1', 'HUBER', 'MSE'.")

    def forward(self, pred_ab: torch.Tensor, target_ab: torch.Tensor) -> torch.Tensor:
        """
        Forward computation.
        Args:
            pred_ab: Predicted ab tensor (B, 2, H, W)
            target_ab: Ground truth ab tensor (B, 2, H, W)
        """
        return self.criterion(pred_ab, target_ab)
