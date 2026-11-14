"""
CelebA Dataset Preparation and Preprocessing Script
Handles image validation, resizing to 128x128, and synthetic sample generation for testing.
"""

import os
import argparse
from typing import Optional
from PIL import Image, ImageDraw
import numpy as np
from tqdm import tqdm


def generate_sample_dataset(
    output_dir: str = "data/raw",
    num_samples: int = 40,
    image_size: int = 128
) -> None:
    """
    Generates realistic synthetic facial portrait images with diverse skin tones,
    hair colors, eyes, lips, and backgrounds for testing and development.
    """
    os.makedirs(output_dir, exist_ok=True)
    print(f"Generating {num_samples} sample images in '{output_dir}'...")

    skin_tones = [
        (255, 224, 189), (240, 200, 160), (224, 172, 105),
        (198, 134, 66),  (141, 85, 36),   (255, 230, 205),
        (210, 161, 140), (165, 114, 84),  (115, 68, 42)
    ]
    hair_colors = [
        (30, 25, 20),    (90, 50, 25),    (160, 82, 45),
        (218, 165, 32),  (50, 50, 50),    (180, 100, 50)
    ]
    bg_colors = [
        (135, 206, 235), (220, 220, 220), (240, 230, 140),
        (176, 196, 222), (255, 192, 203), (200, 214, 229)
    ]

    rng = np.random.RandomState(42)

    for i in range(num_samples):
        bg = bg_colors[rng.randint(len(bg_colors))]
        skin = skin_tones[rng.randint(len(skin_tones))]
        hair = hair_colors[rng.randint(len(hair_colors))]

        img = Image.new("RGB", (image_size, image_size), color=bg)
        draw = ImageDraw.Draw(img)

        # Draw hair background
        draw.ellipse([20, 10, 108, 110], fill=hair)

        # Draw head / face
        draw.ellipse([30, 25, 98, 105], fill=skin)

        # Draw hair foreground (bangs / top)
        draw.chord([25, 12, 103, 60], start=180, end=360, fill=hair)

        # Draw eyes
        eye_y = 52
        draw.ellipse([44, eye_y, 52, eye_y + 8], fill=(255, 255, 255))
        draw.ellipse([46, eye_y + 2, 50, eye_y + 6], fill=(40, 30, 20))

        draw.ellipse([76, eye_y, 84, eye_y + 8], fill=(255, 255, 255))
        draw.ellipse([78, eye_y + 2, 82, eye_y + 6], fill=(40, 30, 20))

        # Eyebrows
        draw.line([42, 48, 54, 48], fill=hair, width=2)
        draw.line([74, 48, 86, 48], fill=hair, width=2)

        # Nose
        draw.line([64, 58, 64, 72], fill=(max(0, skin[0] - 40), max(0, skin[1] - 40), max(0, skin[2] - 40)), width=2)

        # Mouth / Lips
        lip_color = (min(255, skin[0] + 30), max(0, skin[1] - 30), max(0, skin[2] - 20))
        draw.ellipse([54, 80, 74, 88], fill=lip_color)

        filename = f"{i+1:06d}.jpg"
        img.save(os.path.join(output_dir, filename), quality=95)

    print(f"Sample dataset successfully generated at '{output_dir}'.")


def preprocess_directory(
    raw_dir: str,
    output_dir: str,
    image_size: int = 128
) -> None:
    """Resizes all raw images to standard target dimensions."""
    os.makedirs(output_dir, exist_ok=True)
    supported = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    raw_files = [f for f in os.listdir(raw_dir) if os.path.splitext(f)[1].lower() in supported]

    print(f"Preprocessing {len(raw_files)} images from '{raw_dir}' to '{output_dir}'...")
    for filename in tqdm(raw_files, desc="Resizing"):
        raw_path = os.path.join(raw_dir, filename)
        save_path = os.path.join(output_dir, filename)
        try:
            with Image.open(raw_path) as img:
                img = img.convert("RGB")
                img = img.resize((image_size, image_size), Image.Resampling.BILINEAR)
                img.save(save_path, quality=95)
        except Exception as e:
            print(f"Skipping corrupted image {filename}: {e}")

    print("Preprocessing completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare CelebA Dataset")
    parser.add_argument("--raw-dir", type=str, default="data/raw", help="Path to raw image folder")
    parser.add_argument("--output-dir", type=str, default="data/processed", help="Path to processed folder")
    parser.add_argument("--image-size", type=int, default=128, help="Target image size (e.g. 128)")
    parser.add_argument("--create-sample", action="store_true", help="Generate synthetic sample dataset")
    parser.add_argument("--num-samples", type=int, default=60, help="Number of sample images to generate")
    args = parser.parse_args()

    if args.create_sample or not os.path.exists(args.raw_dir) or len(os.listdir(args.raw_dir) if os.path.exists(args.raw_dir) else []) == 0:
        generate_sample_dataset(args.raw_dir, num_samples=args.num_samples, image_size=args.image_size)

    preprocess_directory(args.raw_dir, args.output_dir, image_size=args.image_size)
