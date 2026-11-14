"""
Color Space Conversion Utilities
Handles conversions between RGB and CIELAB color spaces for image colorization.
"""

from typing import Union, Tuple
import cv2
import numpy as np
import torch


def rgb_to_lab(
    image: Union[np.ndarray, torch.Tensor]
) -> Tuple[Union[np.ndarray, torch.Tensor], Union[np.ndarray, torch.Tensor]]:
    """
    Converts an RGB image to normalized LAB representation.
    
    L channel is normalized to [0, 1].
    a and b channels are normalized to [-1, 1].
    
    Args:
        image: RGB image as numpy array (H, W, 3) in [0, 255] uint8 or [0, 1] float,
               or torch Tensor (C, H, W) or (B, C, H, W).
               
    Returns:
        Tuple of (L_channel, ab_channels):
            - L_channel: shape (1, H, W) or (H, W) or (B, 1, H, W) in [0, 1]
            - ab_channels: shape (2, H, W) or (H, W, 2) or (B, 2, H, W) in [-1, 1]
    """
    is_torch = isinstance(image, torch.Tensor)
    if is_torch:
        device = image.device
        img_np = image.detach().cpu().numpy()
        # Handle batch or single tensor
        if img_np.ndim == 4:
            # (B, C, H, W) -> process each
            L_list, ab_list = [], []
            for b in range(img_np.shape[0]):
                l, ab = rgb_to_lab(img_np[b].transpose(1, 2, 0))
                L_list.append(l)
                ab_list.append(ab)
            L_tensor = torch.from_numpy(np.stack(L_list, axis=0)).unsqueeze(1).float().to(device)
            ab_tensor = torch.from_numpy(np.stack(ab_list, axis=0)).permute(0, 3, 1, 2).float().to(device)
            return L_tensor, ab_tensor
        elif img_np.ndim == 3:
            if img_np.shape[0] == 3:
                img_np = img_np.transpose(1, 2, 0)
    else:
        img_np = image

    # Ensure float32 in [0, 1]
    if img_np.dtype == np.uint8:
        img_float = img_np.astype(np.float32) / 255.0
    else:
        img_float = np.clip(img_np.astype(np.float32), 0.0, 1.0)

    # OpenCV RGB to LAB: L in [0, 100], a, b in [-127, 127]
    lab = cv2.cvtColor(img_float, cv2.COLOR_RGB2LAB)
    L = lab[:, :, 0] / 100.0              # Normalized to [0, 1]
    ab = lab[:, :, 1:] / 128.0           # Normalized to [-1, 1]

    if is_torch:
        L_t = torch.from_numpy(L).unsqueeze(0).float().to(device)       # (1, H, W)
        ab_t = torch.from_numpy(ab).permute(2, 0, 1).float().to(device) # (2, H, W)
        return L_t, ab_t

    return L, ab


def lab_to_rgb(
    L: Union[np.ndarray, torch.Tensor],
    ab: Union[np.ndarray, torch.Tensor]
) -> Union[np.ndarray, torch.Tensor]:
    """
    Reconstructs an RGB image from normalized L and ab channels.
    
    Args:
        L: Normalized Lightness channel [0, 1]
           Shape: (H, W), (1, H, W), or (B, 1, H, W)
        ab: Normalized color channels [-1, 1]
           Shape: (H, W, 2), (2, H, W), or (B, 2, H, W)
           
    Returns:
        RGB image as uint8 numpy array [0, 255] (H, W, 3) or (B, H, W, 3)
    """
    is_torch = isinstance(L, torch.Tensor)
    if is_torch:
        L_np = L.detach().cpu().numpy()
        ab_np = ab.detach().cpu().numpy()
    else:
        L_np = np.array(L, dtype=np.float32)
        ab_np = np.array(ab, dtype=np.float32)

    # Handle batch dimension
    if L_np.ndim == 4:  # (B, 1, H, W)
        batch_size = L_np.shape[0]
        rgb_list = []
        for i in range(batch_size):
            l_single = L_np[i, 0]
            ab_single = ab_np[i].transpose(1, 2, 0)
            rgb_list.append(lab_to_rgb(l_single, ab_single))
        return np.stack(rgb_list, axis=0)

    # Handle (1, H, W) -> (H, W)
    if L_np.ndim == 3 and L_np.shape[0] == 1:
        L_np = L_np[0]

    # Handle (2, H, W) -> (H, W, 2)
    if ab_np.ndim == 3 and ab_np.shape[0] == 2:
        ab_np = ab_np.transpose(1, 2, 0)

    # Denormalize
    L_denorm = np.clip(L_np * 100.0, 0.0, 100.0)
    ab_denorm = np.clip(ab_np * 128.0, -128.0, 127.0)

    lab_float = np.zeros((L_np.shape[0], L_np.shape[1], 3), dtype=np.float32)
    lab_float[:, :, 0] = L_denorm
    lab_float[:, :, 1:] = ab_denorm

    rgb_float = cv2.cvtColor(lab_float, cv2.COLOR_LAB2RGB)
    rgb_uint8 = np.clip(rgb_float * 255.0, 0.0, 255.0).astype(np.uint8)
    return rgb_uint8
