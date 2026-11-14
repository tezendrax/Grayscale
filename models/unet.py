"""
U-Net CNN Architecture (Primary Model)
Encoder-decoder network with skip connections for high-fidelity image colorization.
"""

import torch
import torch.nn as nn
from models.blocks import DoubleConv, DownBlock, UpBlock


class UNet(nn.Module):
    """
    U-Net Architecture for Image Colorization.
    
    Structure:
    - Input: Grayscale (L channel) [128 x 128 x 1]
    - Encoder 1: Double Conv, 32 filters  [128 x 128 x 32]
    - Encoder 2: Double Conv, 64 filters  [64 x 64 x 64]
    - Encoder 3: Double Conv, 128 filters [32 x 32 x 128]
    - Encoder 4: Double Conv, 256 filters [16 x 16 x 256]
    - Bottleneck: Double Conv, 512 filters [8 x 8 x 512]
    - Decoder 1: Upsampling + Skip [16 x 16 x 256]
    - Decoder 2: Upsampling + Skip [32 x 32 x 128]
    - Decoder 3: Upsampling + Skip [64 x 64 x 64]
    - Decoder 4: Upsampling + Skip [128 x 128 x 32]
    - Output: 1 x 1 Conv, 2 filters (a, b channels) [128 x 128 x 2]
    """
    def __init__(self, in_channels: int = 1, out_channels: int = 2):
        super(UNet, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Encoder stages
        self.inc = DoubleConv(in_channels, 32)
        self.down1 = DownBlock(32, 64)
        self.down2 = DownBlock(64, 128)
        self.down3 = DownBlock(128, 256)
        
        # Bottleneck stage
        self.bottleneck = DownBlock(256, 512)

        # Decoder stages with skip connections
        self.up1 = UpBlock(512, 256)
        self.up2 = UpBlock(256, 128)
        self.up3 = UpBlock(128, 64)
        self.up4 = UpBlock(64, 32)

        # Final 1x1 Convolution layer followed by Tanh
        self.outc = nn.Sequential(
            nn.Conv2d(32, out_channels, kernel_size=1),
            nn.Tanh()  # Normalized color values in [-1, 1]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with skip connections.
        Args:
            x: Input tensor (B, 1, H, W)
        Returns:
            Predicted ab channels tensor (B, 2, H, W)
        """
        # Encoder
        x1 = self.inc(x)         # 128x128x32
        x2 = self.down1(x1)      # 64x64x64
        x3 = self.down2(x2)      # 32x32x128
        x4 = self.down3(x3)      # 16x16x256
        
        # Bottleneck
        x5 = self.bottleneck(x4) # 8x8x512

        # Decoder with skip connections
        d1 = self.up1(x5, x4)    # 16x16x256
        d2 = self.up2(d1, x3)    # 32x32x128
        d3 = self.up3(d2, x2)    # 64x64x64
        d4 = self.up4(d3, x1)    # 128x128x32

        # Output
        logits = self.outc(d4)   # 128x128x2
        return logits
