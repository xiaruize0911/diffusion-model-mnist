"""Shared timestep embedding utilities for diffusion models."""

import torch
import torch.nn as nn
import math
from typing import Optional


class TimestepEmbedder(nn.Module):
    """Timestep embedding for conditioning diffusion models."""
    
    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        """
        Initialize TimestepEmbedder.
        
        Args:
            hidden_size (int): Hidden size of the model
            frequency_embedding_size (int): Size of frequency embedding
        """
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
        """
        Create sinusoidal timestep embeddings.
        
        Args:
            t (torch.Tensor): 1-D tensor of N indices, one per batch element
            dim (int): Dimension of the embedding
            max_period (int): Maximum period for sinusoidal encoding
            
        Returns:
            torch.Tensor: Positional embeddings [N, dim]
        """
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of TimestepEmbedder.
        
        Args:
            t (torch.Tensor): Timestep tensor
            
        Returns:
            torch.Tensor: Timestep embeddings
        """
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class SpatialTimestepEmbedder(nn.Module):
    """Spatial timestep embedding that broadcasts to spatial dimensions."""
    
    def __init__(self, timestep_dim: int, spatial_dim: int):
        """
        Initialize spatial timestep embedder.
        
        Args:
            timestep_dim (int): Dimension of timestep embedding
            spatial_dim (int): Spatial dimension to broadcast to
        """
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(timestep_dim, spatial_dim),
            nn.SiLU(),
            nn.Linear(spatial_dim, spatial_dim)
        )
    
    def forward(self, t_emb: torch.Tensor, spatial_shape: tuple) -> torch.Tensor:
        """
        Forward pass to create spatial timestep embedding.
        
        Args:
            t_emb (torch.Tensor): Timestep embedding [B, timestep_dim]
            spatial_shape (tuple): Spatial shape (H, W)
            
        Returns:
            torch.Tensor: Spatial embedding [B, spatial_dim, H, W]
        """
        # Project timestep embedding
        projected = self.projection(t_emb)  # [B, spatial_dim]
        
        # Expand to spatial dimensions
        B, spatial_dim = projected.shape
        H, W = spatial_shape
        
        # Reshape and broadcast
        spatial_emb = projected.view(B, spatial_dim, 1, 1).expand(B, spatial_dim, H, W)
        
        return spatial_emb


class GroupNormWithTimestep(nn.Module):
    """Group normalization with timestep conditioning."""
    
    def __init__(self, num_groups: int, num_channels: int, timestep_dim: int):
        """
        Initialize GroupNorm with timestep conditioning.
        
        Args:
            num_groups (int): Number of groups for GroupNorm
            num_channels (int): Number of channels
            timestep_dim (int): Timestep embedding dimension
        """
        super().__init__()
        self.group_norm = nn.GroupNorm(num_groups, num_channels)
        
        # Timestep modulation
        self.timestep_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(timestep_dim, 2 * num_channels)
        )
    
    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with timestep conditioning.
        
        Args:
            x (torch.Tensor): Input features [B, C, H, W]
            t_emb (torch.Tensor): Timestep embedding [B, timestep_dim]
            
        Returns:
            torch.Tensor: Normalized and modulated features
        """
        # Apply group normalization
        x = self.group_norm(x)
        
        # Get scale and shift from timestep
        scale_shift = self.timestep_proj(t_emb)  # [B, 2*C]
        scale, shift = scale_shift.chunk(2, dim=1)  # Each [B, C]
        
        # Apply modulation
        x = x * (1 + scale.unsqueeze(-1).unsqueeze(-1)) + shift.unsqueeze(-1).unsqueeze(-1)
        
        return x


class ResBlockWithTimestep(nn.Module):
    """Residual block with timestep conditioning."""
    
    def __init__(self, in_channels: int, out_channels: int, timestep_dim: int, 
                 use_conv_shortcut: bool = False, dropout: float = 0.0):
        """
        Initialize ResBlock with timestep conditioning.
        
        Args:
            in_channels (int): Input channels
            out_channels (int): Output channels  
            timestep_dim (int): Timestep embedding dimension
            use_conv_shortcut (bool): Whether to use conv for shortcut
            dropout (float): Dropout probability
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = use_conv_shortcut
        
        # First conv block
        self.norm1 = GroupNormWithTimestep(8, in_channels, timestep_dim)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        
        # Second conv block
        self.norm2 = GroupNormWithTimestep(8, out_channels, timestep_dim)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        
        # Shortcut connection
        if in_channels != out_channels:
            if use_conv_shortcut:
                self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
            else:
                self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()
    
    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with timestep conditioning.
        
        Args:
            x (torch.Tensor): Input features [B, C, H, W]
            t_emb (torch.Tensor): Timestep embedding [B, timestep_dim]
            
        Returns:
            torch.Tensor: Output features [B, out_channels, H, W]
        """
        # Shortcut connection
        skip = self.shortcut(x)
        
        # First block
        h = self.norm1(x, t_emb)
        h = torch.relu(h)
        h = self.conv1(h)
        
        # Second block
        h = self.norm2(h, t_emb)
        h = torch.relu(h)
        h = self.dropout(h)
        h = self.conv2(h)
        
        return h + skip


def zero_module(module: nn.Module) -> nn.Module:
    """Zero out the parameters of a module and return it."""
    for p in module.parameters():
        p.detach().zero_()
    return module