"""
Reusable Convolutional Blocks for Colorization Models
Contains DoubleConv and up/down sampling building blocks.
"""

import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    """
    [Conv2d -> BatchNorm -> ReLU] x 2 block.
    Standard building block for U-Net encoder and decoder stages.
    """
    def __init__(self, in_channels: int, out_channels: int, mid_channels: int = None):
        super(DoubleConv, self).__init__()
        if mid_channels is None:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)


class DownBlock(nn.Module):
    """
    Downsampling block: MaxPool2d(2) followed by DoubleConv.
    """
    def __init__(self, in_channels: int, out_channels: int):
        super(DownBlock, self).__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(x)
        return self.conv(x)


class UpBlock(nn.Module):
    """
    Upsampling block: Upsample / ConvTranspose2d followed by skip connection concatenation
    and DoubleConv.
    """
    def __init__(self, in_channels: int, out_channels: int, bilinear: bool = False):
        super(UpBlock, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        # Pad if x1 and x2 have slight spatial mismatch
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]
        if diff_y != 0 or diff_x != 0:
            x1 = nn.functional.pad(
                x1, [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2]
            )
        # Concatenate skip connection
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)
