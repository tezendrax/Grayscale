"""
Basic CNN Architecture (Baseline Model)
A standard sequential convolutional neural network for image colorization.
"""

import torch
import torch.nn as nn


class BasicCNN(nn.Module):
    """
    Baseline Convolutional Neural Network for grayscale-to-color prediction.
    Takes 1-channel Lightness (L) image and outputs 2-channel color (ab) representation.
    Without skip connections, demonstrating the baseline performance.
    """
    def __init__(self, in_channels: int = 1, out_channels: int = 2):
        super(BasicCNN, self).__init__()
        
        # Feature extraction / Encoder
        self.encoder = nn.Sequential(
            # 128x128 -> 128x128
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            # 128x128 -> 64x64
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            # 64x64 -> 32x32
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            # 32x32 -> 16x16
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            # Bottleneck representation
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        
        # Decoder / Reconstruction
        self.decoder = nn.Sequential(
            # 16x16 -> 32x32
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            # 32x32 -> 64x64
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            # 64x64 -> 128x128
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            # Final output layer: 128x128 -> 2 channels (a, b)
            nn.Conv2d(32, out_channels, kernel_size=3, padding=1),
            nn.Tanh()  # Produces output in [-1, 1] matching normalized ab channels
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Args:
            x: Input tensor of shape (B, 1, 128, 128) representing L channel.
        Returns:
            Output tensor of shape (B, 2, 128, 128) representing predicted ab channels.
        """
        feat = self.encoder(x)
        out = self.decoder(feat)
        return out
