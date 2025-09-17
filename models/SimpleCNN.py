"""Simple CNN implementation for diffusion model with timestep embedding."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from .timestep_embedding import TimestepEmbedder


class SimpleCNN(nn.Module):
    """Simple CNN with timestep conditioning for diffusion model."""
    
    def __init__(self, in_channels=1, out_channels=1, features=16, timestep_dim=128):
        super(SimpleCNN, self).__init__()
        
        # Timestep embedding
        self.timestep_embedder = TimestepEmbedder(timestep_dim)
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels, features, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(features, features, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(features, features * 2, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(features * 2, features * 4, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(features * 4, features * 2, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(features * 2, features, kernel_size=3, padding=1)
        self.conv7 = nn.Conv2d(features, out_channels, kernel_size=3, padding=1)
        
        # Timestep projections for each layer
        self.time_proj1 = nn.Linear(timestep_dim, features)
        self.time_proj2 = nn.Linear(timestep_dim, features)
        self.time_proj3 = nn.Linear(timestep_dim, features * 2)
        self.time_proj4 = nn.Linear(timestep_dim, features * 4)
        self.time_proj5 = nn.Linear(timestep_dim, features * 2)
        self.time_proj6 = nn.Linear(timestep_dim, features)

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
            t_emb = torch.zeros(x.shape[0], 128, device=x.device)
        
        # Layer 1 with timestep conditioning
        x1 = F.relu(self.conv1(x))
        if t is not None:
            t1 = self.time_proj1(t_emb).unsqueeze(-1).unsqueeze(-1)  # [B, features, 1, 1]
            x1 = x1 + t1
        
        # Layer 2 with timestep conditioning
        x2 = F.relu(self.conv2(x1))
        if t is not None:
            t2 = self.time_proj2(t_emb).unsqueeze(-1).unsqueeze(-1)
            x2 = x2 + t2
        
        # Layer 3 with timestep conditioning
        x3 = F.relu(self.conv3(x2))
        if t is not None:
            t3 = self.time_proj3(t_emb).unsqueeze(-1).unsqueeze(-1)
            x3 = x3 + t3
        
        # Layer 4 with timestep conditioning
        x4 = F.relu(self.conv4(x3))
        if t is not None:
            t4 = self.time_proj4(t_emb).unsqueeze(-1).unsqueeze(-1)
            x4 = x4 + t4
        
        # Layer 5 with timestep conditioning
        x5 = F.relu(self.conv5(x4))
        if t is not None:
            t5 = self.time_proj5(t_emb).unsqueeze(-1).unsqueeze(-1)
            x5 = x5 + t5
        
        # Layer 6 with timestep conditioning
        x6 = F.relu(self.conv6(x5))
        if t is not None:
            t6 = self.time_proj6(t_emb).unsqueeze(-1).unsqueeze(-1)
            x6 = x6 + t6
        
        # Final layer (no timestep conditioning needed)
        x7 = self.conv7(x6)
        
        return x7