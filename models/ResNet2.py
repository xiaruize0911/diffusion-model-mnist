"""ResNet2 implementation for diffusion model with timestep embedding."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from .timestep_embedding import TimestepEmbedder

class Residual(nn.Module):
    """Residual block with timestep conditioning."""
    def __init__(self, input_channels, num_channels, use_1x1conv=True, strides=1, timestep_dim=128):
        super(Residual, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, num_channels, kernel_size=3, stride=strides, padding=1)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(input_channels, num_channels, kernel_size=1, stride=strides, padding=0)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)
        
        # Timestep conditioning
        self.timestep_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(timestep_dim, num_channels)
        )

    def forward(self, x, t_emb: Optional[torch.Tensor] = None):
        Y = F.relu(self.bn1(self.conv1(x)))
        Y = self.bn2(self.conv2(Y))
        
        # Apply timestep conditioning
        if t_emb is not None:
            t_proj = self.timestep_proj(t_emb).unsqueeze(-1).unsqueeze(-1)
            Y = Y + t_proj
        
        if self.conv3 is not None:
            x = self.conv3(x)
        Y += x
        return F.relu(Y)

def resnet_block(input_channels, num_channels, num_residuals, first_block=False, timestep_dim=128):
    """Create ResNet block with timestep conditioning."""
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(Residual(input_channels, num_channels, use_1x1conv=True, timestep_dim=timestep_dim))
        else:
            blk.append(Residual(num_channels, num_channels, timestep_dim=timestep_dim))
    return blk

class ResNet2(nn.Module):
    """ResNet2 with timestep conditioning for diffusion model."""
    def __init__(self, in_channels=1, out_channels=1, timestep_dim=128):
        super().__init__()
        self.timestep_dim = timestep_dim
        
        # Timestep embedding
        self.timestep_embedder = TimestepEmbedder(timestep_dim)
        
        # Initial layers
        self.initial_conv = nn.Conv2d(in_channels, out_channels=4, kernel_size=3, padding=1)
        self.initial_bn = nn.BatchNorm2d(4)
        
        # ResNet blocks
        self.block1 = nn.ModuleList(resnet_block(4, 4, 2, first_block=True, timestep_dim=timestep_dim))
        self.block2 = nn.ModuleList(resnet_block(4, 8, 3, timestep_dim=timestep_dim))
        self.block3 = nn.ModuleList(resnet_block(8, 16, 3, timestep_dim=timestep_dim))
        self.block4 = nn.ModuleList(resnet_block(16, 32, 3, timestep_dim=timestep_dim))
        self.block5 = nn.ModuleList(resnet_block(32, 64, 3, timestep_dim=timestep_dim))
        self.block6 = nn.ModuleList(resnet_block(64, 128, 3, timestep_dim=timestep_dim))
        
        # Final layer
        self.final_conv = nn.Conv2d(128, out_channels, kernel_size=1)
    
    def forward(self, x, t: Optional[torch.Tensor] = None):
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
            t_emb = torch.zeros(x.shape[0], self.timestep_dim, device=x.device)
        
        # Initial layers
        x = F.relu(self.initial_bn(self.initial_conv(x)))
        
        # Apply ResNet blocks with timestep conditioning
        for block in self.block1:
            x = block(x, t_emb)
        for block in self.block2:
            x = block(x, t_emb)
        for block in self.block3:
            x = block(x, t_emb)
        for block in self.block4:
            x = block(x, t_emb)
        for block in self.block5:
            x = block(x, t_emb)
        for block in self.block6:
            x = block(x, t_emb)
        
        # Final layer
        return self.final_conv(x)

