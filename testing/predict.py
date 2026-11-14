"""
Inference and Image Colorization Script
Generates colorized RGB images from grayscale inputs using trained CNN models.
"""

import os
import sys
import argparse
from typing import Union
import numpy as np
from PIL import Image
import cv2
import torch
import torch.nn as nn

# Add project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from preprocessing.color_space import rgb_to_lab, lab_to_rgb
from models.unet import UNet
from models.basic_cnn import BasicCNN
from utils.checkpoint import load_checkpoint
from utils.visualization import plot_side_by_side


def predict_color_channels(
    model: nn.Module,
    l_tensor: torch.Tensor,
    device: torch.device
) -> torch.Tensor:
    """
    Predicts the 'a' and 'b' color channels from normalized L channel tensor.
    
    Args:
        model: Trained PyTorch model
        l_tensor: Lightness tensor of shape (B, 1, H, W) or (1, H, W) in [0, 1]
        device: torch.device
        
    Returns:
        Predicted ab tensor of shape (B, 2, H, W) or (2, H, W) in [-1, 1]
    """
    model.eval()
    if l_tensor.ndim == 3:
        l_tensor = l_tensor.unsqueeze(0)  # (1, 1, H, W)
    l_tensor = l_tensor.to(device)

    with torch.no_grad():
        pred_ab = model(l_tensor)

    return pred_ab


def colorize_image(
    model: nn.Module,
    image_input: Union[str, np.ndarray, Image.Image],
    device: torch.device,
    image_size: int = 128
) -> np.ndarray:
    """
    Takes a grayscale or color image, extracts L, predicts ab, and reconstructs RGB.
    
    Args:
        model: Trained PyTorch model
        image_input: File path, numpy array, or PIL Image
        device: torch.device
        image_size: Target resolution (128x128)
        
    Returns:
        Reconstructed RGB uint8 image (128, 128, 3)
    """
    if isinstance(image_input, str):
        img_pil = Image.open(image_input).convert("RGB")
    elif isinstance(image_input, np.ndarray):
        if image_input.ndim == 2:
            img_pil = Image.fromarray(image_input).convert("RGB")
        else:
            img_pil = Image.fromarray(image_input)
    elif isinstance(image_input, Image.Image):
        img_pil = image_input.convert("RGB")
    else:
        raise TypeError(f"Unsupported image input type: {type(image_input)}")

    # Resize to target dimension
    img_resized = img_pil.resize((image_size, image_size), Image.Resampling.BILINEAR)
    rgb_np = np.array(img_resized, dtype=np.uint8)

    # Extract L channel
    L_norm, _ = rgb_to_lab(rgb_np)
    L_tensor = torch.from_numpy(L_norm).unsqueeze(0).unsqueeze(0).float()  # (1, 1, 128, 128)

    # Predict ab channels
    pred_ab_tensor = predict_color_channels(model, L_tensor, device)  # (1, 2, 128, 128)

    # Reconstruct RGB
    pred_rgb = lab_to_rgb(L_tensor.squeeze(0), pred_ab_tensor.squeeze(0))
    return pred_rgb


def main():
    parser = argparse.ArgumentParser(description="Colorize Grayscale Image")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint .pth")
    parser.add_argument("--model-type", type=str, default="unet", choices=["unet", "basic_cnn"], help="Architecture type")
    parser.add_argument("--output", type=str, default="outputs/colorized_images/colorized_output.png", help="Save path")
    parser.add_argument("--device", type=str, default=None, help="cuda or cpu")
    args = parser.parse_args()

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Loading {args.model_type.upper()} model on {device}...")

    if args.model_type == "unet":
        model = UNet(in_channels=1, out_channels=2)
    else:
        model = BasicCNN(in_channels=1, out_channels=2)

    load_checkpoint(args.checkpoint, model, device=device)
    model.to(device)

    print(f"Colorizing image: {args.image}...")
    colorized_rgb = colorize_image(model, args.image, device=device)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    Image.fromarray(colorized_rgb).save(args.output)
    print(f"Colorized result saved to: {args.output}")


if __name__ == "__main__":
    main()
