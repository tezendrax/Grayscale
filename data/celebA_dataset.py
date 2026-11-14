"""
CelebA Dataset and DataLoader for Image Colorization
Loads celebrity face images, resizes to 128x128, and converts to LAB color space.
"""

import os
import sys
from typing import Tuple, List, Optional, Dict, Any
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader

# Add project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from preprocessing.color_space import rgb_to_lab


class CelebADataset(Dataset):
    """
    PyTorch Dataset for CelebA Face Colorization.
    
    Returns:
        L: Normalized Lightness tensor of shape (1, H, W) in [0, 1]
        ab: Normalized color channels tensor of shape (2, H, W) in [-1, 1]
        rgb: Original RGB tensor of shape (3, H, W) in [0, 1]
    """
    def __init__(
        self,
        image_paths: List[str],
        image_size: int = 128,
        is_train: bool = True
    ):
        self.image_paths = image_paths
        self.image_size = image_size
        self.is_train = is_train

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        path = self.image_paths[idx]
        with Image.open(path) as img:
            img = img.convert("RGB")
            # Resize image to target resolution (128x128)
            img = img.resize((self.image_size, self.image_size), Image.Resampling.BILINEAR)
            
            # Simple data augmentation for training
            if self.is_train and np.random.rand() > 0.5:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
                
            rgb_np = np.array(img, dtype=np.uint8)

        # Convert to LAB
        L_np, ab_np = rgb_to_lab(rgb_np)
        
        # Format tensors
        L_tensor = torch.from_numpy(L_np).unsqueeze(0).float()           # (1, 128, 128)
        ab_tensor = torch.from_numpy(ab_np).permute(2, 0, 1).float()     # (2, 128, 128)
        rgb_tensor = torch.from_numpy(rgb_np).permute(2, 0, 1).float() / 255.0  # (3, 128, 128)

        return {
            "L": L_tensor,
            "ab": ab_tensor,
            "rgb": rgb_tensor,
            "path": path
        }


def get_image_paths(directory: str) -> List[str]:
    """Recursively collects all supported image paths from a directory."""
    supported_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    image_paths = []
    if os.path.isdir(directory):
        for root, _, files in os.walk(directory):
            for file in sorted(files):
                if os.path.splitext(file)[1].lower() in supported_exts:
                    image_paths.append(os.path.join(root, file))
    return image_paths


def get_dataloaders(
    data_dir: str,
    batch_size: int = 16,
    image_size: int = 128,
    train_split: float = 0.8,
    val_split: float = 0.1,
    test_split: float = 0.1,
    num_workers: int = 0,
    seed: int = 42
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Creates train, validation, and test DataLoaders from image folder.
    """
    image_paths = get_image_paths(data_dir)
    if not image_paths:
        raise FileNotFoundError(f"No valid images found in directory: {data_dir}")

    # Deterministic split
    rng = np.random.RandomState(seed)
    indices = rng.permutation(len(image_paths))

    n_total = len(image_paths)
    n_train = int(n_total * train_split)
    n_val = int(n_total * val_split)

    train_indices = indices[:n_train]
    val_indices = indices[n_train:n_train + n_val]
    test_indices = indices[n_train + n_val:]

    train_paths = [image_paths[i] for i in train_indices]
    val_paths = [image_paths[i] for i in val_indices]
    test_paths = [image_paths[i] for i in test_indices]

    train_dataset = CelebADataset(train_paths, image_size=image_size, is_train=True)
    val_dataset = CelebADataset(val_paths, image_size=image_size, is_train=False)
    test_dataset = CelebADataset(test_paths, image_size=image_size, is_train=False)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=False
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=False
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=False
    )

    return train_loader, val_loader, test_loader
