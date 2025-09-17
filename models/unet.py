"""U-Net implementation for diffusion model with timestep embedding."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional
from .timestep_embedding import TimestepEmbedder, ResBlockWithTimestep

class DoubleConvBlock(nn.Module):
    """Convolutional Block with timestep conditioning"""
    def __init__(self, in_channels: int, out_channels: int, timestep_dim: int = 128):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        
        # Timestep projection for modulating conv layers
        self.timestep_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(timestep_dim, out_channels)
        )
        
    def forward(self, x: torch.Tensor, t_emb: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        
        # Apply timestep conditioning if provided
        if t_emb is not None:
            t_proj = self.timestep_proj(t_emb)  # [B, out_channels]
            # Reshape for broadcasting: [B, C, 1, 1]
            t_proj = t_proj.unsqueeze(-1).unsqueeze(-1)
            x = x + t_proj
        
        x = F.relu(self.conv2(x))
        return x

class Encoder(nn.Module):
    """Encoder with timestep conditioning"""

    def __init__(self, channels: list[int], timestep_dim: int = 128):
        """
        Initialize Encoder.
        
        Args:
            channels (list[int]): List of channel sizes for each layer
            timestep_dim (int): Timestep embedding dimension
        """
        super().__init__()
        self.encoder_blocks = nn.ModuleList()
        for i in range(len(channels) - 1):
            self.encoder_blocks.append(
                DoubleConvBlock(channels[i], channels[i + 1], timestep_dim)
            )

    def forward(self, x: torch.Tensor, t_emb: Optional[torch.Tensor] = None):
        """
        Forward pass of Encoder.
        
        Args:
            x (torch.Tensor): Input features
            t_emb (torch.Tensor): Timestep embedding
            
        Returns:
            tuple: (encoder_features, final_encoded_features)
        """
        encoder_features = []
        for block in self.encoder_blocks:
            encoder_features.append(x)
            x = block(x, t_emb)
        encoder_features.append(x)
        return encoder_features, x

class Decoder(nn.Module):
    """Decoder with timestep conditioning"""
    def __init__(self, channels: list[int], timestep_dim: int = 128):
        """
        Initialize Decoder.
        
        Args:
            channels (list[int]): List of channel sizes for each layer
            timestep_dim (int): Timestep embedding dimension
        """
        super().__init__()
        self.decoder_blocks = nn.ModuleList()
        
        for i in range(0, len(channels) - 1):
            input_channels = channels[i]
            output_channels = channels[i + 1]
            
            self.decoder_blocks.append(
                nn.Sequential(
                    nn.ConvTranspose2d(channels[i]*2, channels[i], kernel_size=1, stride=1),
                    DoubleConvBlock(input_channels, output_channels, timestep_dim)
                )
            )

    def forward(self, x: torch.Tensor, encoder_features: list[torch.Tensor], 
                t_emb: Optional[torch.Tensor] = None) -> torch.Tensor:
        for i, block in enumerate(self.decoder_blocks):
            # Concatenate with skip connections from encoder
            x = torch.cat([x, encoder_features[i]], dim=1)
            
            # Apply transpose conv
            x = block[0](x)  # ConvTranspose2d
            
            # Apply double conv block with timestep conditioning
            x = block[1](x, t_emb)  # DoubleConvBlock
        return x

class UNet(nn.Module):
    """Main U-Net architecture for noise prediction with timestep conditioning."""

    def __init__(self, in_channels: int, out_channels: int, mid_channels: int, 
                 timestep_dim: int = 128) -> None:
        super().__init__()
        channels = [1, 2, 4, 8, 16, 32, 64]
        
        # Timestep embedding
        self.timestep_embedder = TimestepEmbedder(timestep_dim)
        
        # Encoder and decoder with timestep conditioning
        self.encoder = Encoder(channels, timestep_dim)
        
        # Prepare decoder channels by reversing and removing input channel
        decoder_channels = channels[::-1]  # [64, 32, 16, 8, 4, 2, 1]
        self.decoder = Decoder(decoder_channels[:-1], timestep_dim)  # [64, 32, 16, 8, 4, 2]
        
        # Final output layer
        self.output_conv = nn.Conv2d(2, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor, t: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass with timestep conditioning.
        
        Args:
            x (torch.Tensor): Input noisy images [B, C, H, W]
            t (torch.Tensor): Timesteps [B]
            
        Returns:
            torch.Tensor: Predicted noise [B, C, H, W]
        """
        # Generate timestep embeddings
        if t is not None:
            t_emb = self.timestep_embedder(t)  # [B, timestep_dim]
        else:
            # If no timestep provided, use zeros
            t_emb = torch.zeros(x.shape[0], 128, device=x.device)
        
        # Encoder with timestep conditioning
        encoder_features, x = self.encoder(x, t_emb)
        encoder_features = list(reversed(encoder_features))
        
        # Decoder with timestep conditioning
        x = self.decoder(x, encoder_features, t_emb)
        
        # Final output
        return self.output_conv(x)