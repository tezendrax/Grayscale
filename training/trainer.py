"""
Trainer Module for CNN-Based Image Colorization
Orchestrates training loops, validation, logging, and checkpoint management.
"""

import os
from typing import Dict, List, Any, Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from training.validate import validate_one_epoch
from utils.checkpoint import save_checkpoint
from utils.visualization import plot_loss_curves


def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device
) -> float:
    """
    Trains the model for one epoch.
    
    Args:
        model: PyTorch model
        train_loader: DataLoader for training data
        criterion: Loss function
        optimizer: Optimizer
        device: torch.device
        
    Returns:
        Average training loss for the epoch
    """
    model.train()
    running_loss = 0.0
    total_samples = 0

    for batch in train_loader:
        l_channel = batch["L"].to(device)
        target_ab = batch["ab"].to(device)

        optimizer.zero_grad()
        predicted_ab = model(l_channel)
        loss = criterion(predicted_ab, target_ab)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * l_channel.size(0)
        total_samples += l_channel.size(0)

    return running_loss / max(1, total_samples)


class Trainer:
    """
    Main Trainer class to train, validate, and checkpoint colorization models.
    """
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        model_name: str = "model",
        checkpoint_dir: str = "checkpoints",
        graphs_dir: str = "outputs/training_graphs",
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        early_stopping_patience: int = 10
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.model_name = model_name
        self.checkpoint_dir = checkpoint_dir
        self.graphs_dir = graphs_dir
        self.scheduler = scheduler
        self.early_stopping_patience = early_stopping_patience

        self.train_losses: List[float] = []
        self.val_losses: List[float] = []
        self.best_val_loss = float("inf")

    def fit(self, epochs: int) -> Dict[str, Any]:
        """
        Runs the full training loop across the requested number of epochs.
        """
        print(f"\n{'='*50}")
        print(f"Starting training: {self.model_name.upper()} for {epochs} epochs on {self.device}")
        print(f"{'='*50}")

        patience_counter = 0

        for epoch in range(1, epochs + 1):
            train_loss = train_one_epoch(
                self.model, self.train_loader, self.criterion, self.optimizer, self.device
            )
            val_loss, _ = validate_one_epoch(
                self.model, self.val_loader, self.criterion, self.device, compute_metrics=False
            )

            if self.scheduler is not None:
                self.scheduler.step(val_loss)

            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)

            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1

            # Save checkpoint
            state = {
                "epoch": epoch,
                "model_name": self.model_name,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "train_loss": train_loss,
                "val_loss": val_loss,
                "best_val_loss": self.best_val_loss
            }
            save_checkpoint(
                state=state,
                is_best=is_best,
                checkpoint_dir=self.checkpoint_dir,
                model_name=self.model_name
            )

            status_star = "(* Best)" if is_best else ""
            print(f"Epoch [{epoch:02d}/{epochs:02d}] - Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} {status_star}")

            if patience_counter >= self.early_stopping_patience:
                print(f"Early stopping triggered after {epoch} epochs (no improvement for {self.early_stopping_patience} epochs).")
                break

        # Save training loss curve
        graph_path = os.path.join(self.graphs_dir, f"{self.model_name}_loss_curve.png")
        plot_loss_curves(self.train_losses, self.val_losses, model_name=self.model_name, save_path=graph_path)
        print(f"Training loss curves saved to: {graph_path}")

        return {
            "model_name": self.model_name,
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "best_val_loss": self.best_val_loss,
        }
