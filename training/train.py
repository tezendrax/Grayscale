"""
Training Entry Point for CNN Image Colorization
Trains Basic CNN and/or U-Net models on CelebA dataset.
"""

import os
import sys
import argparse
import yaml
import torch
import torch.optim as optim

# Add project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.celebA_dataset import get_dataloaders
from preprocessing.prepare_dataset import generate_sample_dataset, preprocess_directory
from models.basic_cnn import BasicCNN
from models.unet import UNet
from training.loss import ColorizationLoss
from training.trainer import Trainer


def load_config(config_path: str = "configs/config.yaml") -> dict:
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    return {}


def train_single_model(
    model_name: str,
    train_loader,
    val_loader,
    device: torch.device,
    config: dict,
    epochs: int,
    lr: float
):
    print(f"\n Initializing {model_name.upper()}...")
    if model_name == "basic_cnn":
        model = BasicCNN(in_channels=1, out_channels=2)
    elif model_name == "unet":
        model = UNet(in_channels=1, out_channels=2)
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    criterion = ColorizationLoss(loss_type=config.get("training", {}).get("loss", "L1"))
    optimizer = optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=config.get("training", {}).get("weight_decay", 1e-5)
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

    checkpoint_dir = config.get("training", {}).get("checkpoint_dir", "checkpoints")
    graphs_dir = config.get("paths", {}).get("training_graphs_dir", "outputs/training_graphs")

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        model_name=model_name,
        checkpoint_dir=checkpoint_dir,
        graphs_dir=graphs_dir,
        scheduler=scheduler,
        early_stopping_patience=config.get("training", {}).get("early_stopping_patience", 10)
    )

    history = trainer.fit(epochs=epochs)
    return history


def main():
    parser = argparse.ArgumentParser(description="Train Colorization Models")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to YAML config")
    parser.add_argument("--model", type=str, default="all", choices=["unet", "basic_cnn", "all"], help="Model to train")
    parser.add_argument("--epochs", type=int, default=None, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    parser.add_argument("--data-dir", type=str, default=None, help="Dataset directory")
    parser.add_argument("--device", type=str, default=None, help="cuda or cpu")
    args = parser.parse_args()

    config = load_config(args.config)

    # Resolve device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Dataset directory check
    data_dir = args.data_dir or config.get("dataset", {}).get("processed_dir", "data/processed")
    if not os.path.exists(data_dir) or len(os.listdir(data_dir) if os.path.exists(data_dir) else []) == 0:
        raw_dir = config.get("dataset", {}).get("raw_dir", "data/raw")
        if not os.path.exists(raw_dir) or len(os.listdir(raw_dir) if os.path.exists(raw_dir) else []) == 0:
            print(f"No existing images found in '{raw_dir}'. Generating synthetic CelebA sample dataset...")
            generate_sample_dataset(raw_dir, num_samples=60, image_size=128)
        print(f"Preprocessing raw dataset into '{data_dir}'...")
        preprocess_directory(raw_dir, data_dir, image_size=128)

    # Training parameters
    epochs = args.epochs or config.get("training", {}).get("epochs", 30)
    batch_size = args.batch_size or config.get("training", {}).get("batch_size", 16)
    lr = args.lr or config.get("training", {}).get("learning_rate", 0.001)

    print(f"Loading dataset from: {data_dir}")
    train_loader, val_loader, test_loader = get_dataloaders(
        data_dir=data_dir,
        batch_size=batch_size,
        image_size=config.get("dataset", {}).get("image_size", 128),
        train_split=config.get("dataset", {}).get("train_split", 0.8),
        val_split=config.get("dataset", {}).get("val_split", 0.1),
        test_split=config.get("dataset", {}).get("test_split", 0.1),
        seed=config.get("project", {}).get("seed", 42)
    )
    print(f"Dataset split: {len(train_loader.dataset)} train, {len(val_loader.dataset)} val, {len(test_loader.dataset)} test")

    models_to_train = ["basic_cnn", "unet"] if args.model == "all" else [args.model]

    for m in models_to_train:
        train_single_model(
            model_name=m,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            config=config,
            epochs=epochs,
            lr=lr
        )

    print("\n Training process completed successfully!")


if __name__ == "__main__":
    main()
