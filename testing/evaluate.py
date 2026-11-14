"""
Model Evaluation Script
Evaluates trained models on the test set using MAE, PSNR, and SSIM metrics.
"""

import os
import sys
import argparse
from typing import Dict
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

# Add project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.celebA_dataset import get_dataloaders
from models.unet import UNet
from models.basic_cnn import BasicCNN
from utils.checkpoint import load_checkpoint
from utils.metrics import calculate_all_metrics
from preprocessing.color_space import lab_to_rgb


def evaluate_model(
    model: nn.Module,
    test_loader,
    device: torch.device
) -> Dict[str, float]:
    """
    Evaluates model performance across test dataset.
    Computes average MAE, PSNR, and SSIM.
    """
    model.eval()
    accumulators = {"MAE": 0.0, "PSNR": 0.0, "SSIM": 0.0}
    total_samples = 0

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            l_t = batch["L"].to(device)
            rgb_t = batch["rgb"]  # (B, 3, H, W) in [0, 1]

            pred_ab = model(l_t)
            pred_rgb_batch = lab_to_rgb(l_t, pred_ab)  # (B, H, W, 3) in uint8 [0, 255]
            gt_rgb_batch = (rgb_t.permute(0, 2, 3, 1).cpu().numpy() * 255.0).astype(np.uint8)

            for i in range(len(pred_rgb_batch)):
                metrics = calculate_all_metrics(pred_rgb_batch[i], gt_rgb_batch[i])
                for k in accumulators:
                    accumulators[k] += metrics[k]
                total_samples += 1

    results = {k: v / max(1, total_samples) for k, v in accumulators.items()}
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate Colorization Model on Test Set")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint .pth file")
    parser.add_argument("--model-type", type=str, default="unet", choices=["unet", "basic_cnn"], help="Model type")
    parser.add_argument("--data-dir", type=str, default="data/processed", help="Path to test data directory")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--device", type=str, default=None, help="cuda or cpu")
    args = parser.parse_args()

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Loading {args.model_type.upper()} model onto {device}...")

    if args.model_type == "unet":
        model = UNet(in_channels=1, out_channels=2)
    else:
        model = BasicCNN(in_channels=1, out_channels=2)

    load_checkpoint(args.checkpoint, model, device=device)
    model.to(device)

    _, _, test_loader = get_dataloaders(args.data_dir, batch_size=args.batch_size)
    print(f"Evaluating on {len(test_loader.dataset)} test images...")

    results = evaluate_model(model, test_loader, device=device)

    print("\n" + "=" * 45)
    print(f"  EVALUATION RESULTS: {args.model_type.upper()}")
    print("=" * 45)
    print(f"  Mean Absolute Error (MAE): {results['MAE']:.4f} (Lower is better)")
    print(f"  Peak SNR (PSNR):           {results['PSNR']:.2f} dB (Higher is better)")
    print(f"  Structural Similarity:     {results['SSIM']:.4f} (Higher is better)")
    print("=" * 45 + "\n")


if __name__ == "__main__":
    main()
