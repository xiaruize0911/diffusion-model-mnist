"""U-Net implementation for diffusion model."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class DoubleConvBlock(nn.Module):
    """Convolutional Block"""
    def __init__(self,in_channels: int,out_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)

class Encoder(nn.Module):
    """Encoder"""

    def __init__(self, channels: list[int]):
        """
        Initialize Encoder.
        
        Args:
            in_channels (int): Number of input channels (includes skip connection)
            out_channels (int): Number of output channels
        """
        super().__init__()
        self.encoder_blocks = nn.ModuleList()
        for i in range(len(channels) - 1):
            self.encoder_blocks.append(
                DoubleConvBlock(channels[i], channels[i + 1])
            )

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """
        Forward pass of Encoder`.
        
        Args:
            x (torch.Tensor): Upsampled features
            
        Returns:
            torch.Tensor: Processed features of shape (B, out_channels, H*2, W*2)
        """

        encoder_features = []
        for block in self.encoder_blocks:
            x = block(x)
            encoder_features.append(x)
        return encoder_features

class Decoder(nn.Module):
    """Decoder"""
    def __init__(self, channels: list[int]):
        """
        Initialize Decoder.
        
        Args:
            channels (list[int]): List of channel sizes for each layer
        """
        super().__init__()
        self.decoder_blocks = nn.ModuleList()
        for i in range(len(channels) - 1):
            self.decoder_blocks.append(
                nn.Sequential(
                    nn.ConvTranspose2d(channels[i], channels[i + 1], kernel_size=2, stride=2),
                    DoubleConvBlock(channels[i], channels[i+1])
                )
            )

    def crop_to_fit(self, feature: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Crop the feature map to fit the target size.

        Args:
            feature (torch.Tensor): The feature map to crop.
            target (torch.Tensor): The target tensor to match size.

        Returns:
            torch.Tensor: Cropped feature map.
        """
        _, _, H, W = target.size()
        return F.interpolate(feature, size=(H, W), mode="bilinear", align_corners=False)
    
    def forward(self, x: torch.Tensor, encoder_features: list[torch.Tensor]) -> torch.Tensor:
        for i, block in enumerate(self.decoder_blocks):
            if i < len(encoder_features):
                x = torch.cat([x, self.crop_to_fit(encoder_features[i], x)], dim=1)
            x = block(x)
        return x

class UNet(nn.Module):
    """Main U-Net architecture for noise prediction."""

    def __init__(self, in_channels: int, out_channels: int, mid_channels: int) -> None:
        super().__init__()
        channels = torch.linspace(in_channels, mid_channels, steps=5).int().tolist()
        self.encoder = Encoder(channels)
        self.decoder = Decoder(channels[::-1][:-1])
        self.output_conv = nn.Conv2d(channels[1], out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoder_features = self.encoder(x)
        x = self.decoder(x, encoder_features)
        return self.output_conv(x)