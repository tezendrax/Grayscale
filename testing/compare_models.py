"""
Comparative Analysis: Basic CNN vs U-Net
Generates side-by-side 4-panel visual comparisons and metric benchmarks (MAE, PSNR, SSIM).
"""

import os
import sys
import argparse
import numpy as np
import torch
from PIL import Image

# Add project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.celebA_dataset import get_dataloaders
from models.unet import UNet
from models.basic_cnn import BasicCNN
from utils.checkpoint import load_checkpoint
from utils.metrics import calculate_all_metrics
from utils.visualization import plot_comparison_grid
from preprocessing.color_space import lab_to_rgb
from testing.evaluate import evaluate_model


def compare_models(
    basic_cnn_path: str,
    unet_path: str,
    data_dir: str = "data/processed",
    output_dir: str = "outputs/comparisons",
    num_visual_samples: int = 5,
    device: torch.device = torch.device("cpu")
):
    os.makedirs(output_dir, exist_ok=True)

    print("Loading Basic CNN model...")
    basic_model = BasicCNN(in_channels=1, out_channels=2).to(device)
    load_checkpoint(basic_cnn_path, basic_model, device=device)

    print("Loading U-Net model...")
    unet_model = UNet(in_channels=1, out_channels=2).to(device)
    load_checkpoint(unet_path, unet_model, device=device)

    _, _, test_loader = get_dataloaders(data_dir=data_dir, batch_size=16)
    print(f"Loaded test set with {len(test_loader.dataset)} samples.")

    print("\nRunning comprehensive evaluation across test set...")
    basic_metrics = evaluate_model(basic_model, test_loader, device=device)
    unet_metrics = evaluate_model(unet_model, test_loader, device=device)

    print("\n" + "=" * 65)
    print("              MODEL COMPARISON BENCHMARK RESULTS")
    print("=" * 65)
    print(f"{'Metric':<15} | {'Basic CNN (Baseline)':<22} | {'U-Net (Primary)':<20}")
    print("-" * 65)
    print(f"{'MAE (Lower)':<15} | {basic_metrics['MAE']:<22.4f} | {unet_metrics['MAE']:<20.4f}")
    print(f"{'PSNR (Higher)':<15} | {basic_metrics['PSNR']:<19.2f} dB | {unet_metrics['PSNR']:<17.2f} dB")
    print(f"{'SSIM (Higher)':<15} | {basic_metrics['SSIM']:<22.4f} | {unet_metrics['SSIM']:<20.4f}")
    print("=" * 65)

    # Visual comparisons on test samples
    print(f"\nGenerating {num_visual_samples} 4-panel visual comparisons in '{output_dir}'...")
    sample_count = 0
    basic_model.eval()
    unet_model.eval()

    with torch.no_grad():
        for batch in test_loader:
            l_t = batch["L"].to(device)
            rgb_t = batch["rgb"]

            pred_ab_basic = basic_model(l_t)
            pred_ab_unet = unet_model(l_t)

            pred_rgb_basic = lab_to_rgb(l_t, pred_ab_basic)
            pred_rgb_unet = lab_to_rgb(l_t, pred_ab_unet)
            gt_rgb = (rgb_t.permute(0, 2, 3, 1).cpu().numpy() * 255.0).astype(np.uint8)

            for i in range(len(l_t)):
                if sample_count >= num_visual_samples:
                    break
                sample_count += 1
                
                single_gt = gt_rgb[i]
                single_l = (l_t[i, 0].cpu().numpy() * 255.0).astype(np.uint8)
                single_basic = pred_rgb_basic[i]
                single_unet = pred_rgb_unet[i]

                sample_basic_m = calculate_all_metrics(single_basic, single_gt)
                sample_unet_m = calculate_all_metrics(single_unet, single_gt)

                save_path = os.path.join(output_dir, f"comparison_sample_{sample_count}.png")
                plot_comparison_grid(
                    original_rgb=single_gt,
                    grayscale_l=single_l,
                    basic_cnn_rgb=single_basic,
                    unet_rgb=single_unet,
                    basic_metrics=sample_basic_m,
                    unet_metrics=sample_unet_m,
                    save_path=save_path
                )

            if sample_count >= num_visual_samples:
                break

    print(f"All {sample_count} visual comparison panels successfully saved to '{output_dir}'.")


def main():
    parser = argparse.ArgumentParser(description="Compare Basic CNN and U-Net Models")
    parser.add_argument("--basic-checkpoint", type=str, default="checkpoints/best_basic_cnn.pth", help="Basic CNN path")
    parser.add_argument("--unet-checkpoint", type=str, default="checkpoints/best_unet.pth", help="U-Net path")
    parser.add_argument("--data-dir", type=str, default="data/processed", help="Data directory")
    parser.add_argument("--output-dir", type=str, default="outputs/comparisons", help="Output comparison directory")
    parser.add_argument("--num-samples", type=int, default=5, help="Number of comparison figures to generate")
    parser.add_argument("--device", type=str, default=None, help="cuda or cpu")
    args = parser.parse_args()

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    compare_models(
        basic_cnn_path=args.basic_checkpoint,
        unet_path=args.unet_checkpoint,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        num_visual_samples=args.num_samples,
        device=device
    )


if __name__ == "__main__":
    main()
