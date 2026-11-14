"""
Checkpoint Management Utilities
Functions to save, load, and manage model weights during training.
"""

import os
from typing import Optional, Dict, Any, Union
import torch
import torch.nn as nn


def save_checkpoint(
    state: Dict[str, Any],
    is_best: bool = False,
    checkpoint_dir: str = "checkpoints",
    model_name: str = "model",
    filename: Optional[str] = None
) -> str:
    """
    Saves model checkpoint.
    
    Args:
        state: Dict containing epoch, model_state_dict, optimizer_state_dict, loss, etc.
        is_best: If True, saves as best_{model_name}.pth
        checkpoint_dir: Directory where checkpoints are saved
        model_name: Name identifier for the model (e.g. 'unet', 'basic_cnn')
        filename: Custom file name if provided
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    if filename is None:
        filename = f"latest_{model_name}.pth"
    filepath = os.path.join(checkpoint_dir, filename)
    torch.save(state, filepath)

    if is_best:
        best_path = os.path.join(checkpoint_dir, f"best_{model_name}.pth")
        torch.save(state, best_path)
        return best_path

    return filepath


def load_checkpoint(
    checkpoint_path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: Union[str, torch.device] = "cpu"
) -> Dict[str, Any]:
    """
    Loads model checkpoint.
    
    Args:
        checkpoint_path: Path to .pth file
        model: PyTorch model instance
        optimizer: Optional PyTorch optimizer instance
        device: Device to map tensors to
        
    Returns:
        Checkpoint dictionary containing metadata (epoch, loss, etc.)
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    elif "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    else:
        model.load_state_dict(checkpoint)

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return checkpoint
