"""
Grayscale Dataset Generation Script
Converts original color CelebA images into high-quality black-and-white (grayscale) images.
This forms the core input dataset for the CNN colorization project.
"""

import os
import sys
import argparse
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

# Add project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def convert_rgb_to_grayscale(rgb_image: np.ndarray) -> np.ndarray:
    """
    Converts RGB image to Grayscale using standard perceptual luminance weighting:
    Y = 0.299*R + 0.587*G + 0.114*B
    
    Args:
        rgb_image: RGB image uint8 of shape (H, W, 3)
    Returns:
        Grayscale image uint8 of shape (H, W)
    """
    return cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)


def create_grayscale_dataset(
    input_dir: str = "data/processed",
    output_dir: str = "data/grayscale",
    image_size: int = 128
):
    """
    Reads all color images from input_dir, resizes to image_size, converts to Grayscale,
    and writes them to output_dir.
    """
    os.makedirs(output_dir, exist_ok=True)
    supported_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    files = [f for f in os.listdir(input_dir) if os.path.splitext(f)[1].lower() in supported_exts]

    if not files:
        print(f"No image files found in '{input_dir}'.")
        return

    print(f"\n=======================================================")
    print(f" Converting {len(files)} Colored Images to Grayscale Dataset")
    print(f" Source:      {input_dir}")
    print(f" Destination: {output_dir}")
    print(f" Target Size: {image_size} x {image_size}")
    print(f"=======================================================\n")

    for filename in tqdm(files, desc="Converting to Grayscale"):
        src_path = os.path.join(input_dir, filename)
        dst_path = os.path.join(output_dir, filename)

        try:
            with Image.open(src_path) as img:
                img = img.convert("RGB")
                if img.size != (image_size, image_size):
                    img = img.resize((image_size, image_size), Image.Resampling.BILINEAR)
                
                rgb_np = np.array(img, dtype=np.uint8)
                gray_np = convert_rgb_to_grayscale(rgb_np)
                
                # Save as 8-bit single-channel grayscale PNG/JPG
                gray_pil = Image.fromarray(gray_np, mode="L")
                gray_pil.save(dst_path, quality=95)
        except Exception as e:
            print(f"Error processing {filename}: {e}")

    print(f"\n Successfully generated Grayscale dataset in: {output_dir}")


def generate_side_by_side_samples(
    rgb_dir: str = "data/processed",
    gray_dir: str = "data/grayscale",
    output_preview: str = "outputs/grayscale_samples/rgb_vs_grayscale.png",
    num_samples: int = 4
):
    """
    Creates a visual side-by-side verification grid of Original Color vs Grayscale.
    Great for presentations and screen recordings!
    """
    os.makedirs(os.path.dirname(output_preview), exist_ok=True)
    files = sorted([f for f in os.listdir(gray_dir) if f.endswith(('.jpg', '.png'))])[:num_samples]
    
    if not files:
        return

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(num_samples, 2, figsize=(6, 3 * num_samples))
    if num_samples == 1:
        axes = np.expand_dims(axes, 0)

    for i, f in enumerate(files):
        rgb_img = Image.open(os.path.join(rgb_dir, f)).convert("RGB")
        gray_img = Image.open(os.path.join(gray_dir, f)).convert("L")

        axes[i, 0].imshow(rgb_img)
        axes[i, 0].set_title(f"Original Color: {f}", fontsize=10)
        axes[i, 0].axis("off")

        axes[i, 1].imshow(gray_img, cmap="gray")
        axes[i, 1].set_title(f"Grayscale Input: {f}", fontsize=10)
        axes[i, 1].axis("off")

    plt.tight_layout()
    plt.savefig(output_preview, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Visual comparison preview saved to: {output_preview}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert RGB images to Grayscale dataset")
    parser.add_argument("--input-dir", type=str, default="data/processed", help="Source color images")
    parser.add_argument("--output-dir", type=str, default="data/grayscale", help="Destination grayscale folder")
    parser.add_argument("--image-size", type=int, default=128, help="Target image resolution")
    parser.add_argument("--preview", action="store_true", default=True, help="Generate visual side-by-side preview")
    args = parser.parse_args()

    create_grayscale_dataset(args.input_dir, args.output_dir, args.image_size)
    if args.preview:
        generate_side_by_side_samples(args.input_dir, args.output_dir)
