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

    def forward(self, x: torch.Tensor):
        """
        Forward pass of Encoder.
        
        Args:
            x (torch.Tensor): Input features
            
        Returns:
            tuple: (encoder_features, final_encoded_features)
        """
        encoder_features = []
        for block in self.encoder_blocks:
            encoder_features.append(x)
            x = block(x)
        encoder_features.append(x)
        return encoder_features, x

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
        
        for i in range(0, len(channels) - 1):
            input_channels = channels[i]
            output_channels = channels[i + 1]
            
            self.decoder_blocks.append(
                nn.Sequential(
                    nn.ConvTranspose2d(channels[i]*2, channels[i], kernel_size=1, stride=1),
                    DoubleConvBlock(input_channels, output_channels)
                )
            )

    def forward(self, x: torch.Tensor, encoder_features: list[torch.Tensor]) -> torch.Tensor:
        for i, block in enumerate(self.decoder_blocks):
            # Debug: print shape of x after first encoder block
            # Debug: print shape of encoder features at each stage
            x = torch.cat([x, encoder_features[i]], dim=1)
            # Debug: print shape of x after decoder block
            x = block(x)
        return x

class UNet2(nn.Module):
    """Main U-Net architecture for noise prediction."""

    def __init__(self, in_channels: int, out_channels: int, mid_channels: int) -> None:
        super().__init__()
        channels = [1, 2, 4, 8, 16, 32, 64, 128]
        self.encoder = Encoder(channels)
    # Prepare decoder channels by reversing and removing input channel
        decoder_channels = channels[::-1]  # [64, 32, 16, 8, 4, 2, 1]
        self.decoder = Decoder(decoder_channels[:-1])  # [64, 32, 16, 8, 4, 2]
        self.output_conv = nn.Conv2d(2, out_channels, kernel_size=1)  # Final output layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # print("Input shape:", x.shape)
        encoder_features, x = self.encoder(x)
        encoder_features = list(reversed(encoder_features))
        # print(x.shape)
        x = self.decoder(x, encoder_features)
        return self.output_conv(x)