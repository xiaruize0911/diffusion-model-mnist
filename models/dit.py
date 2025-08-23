"""DiT (Diffusion Transformer) implementation for diffusion model."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional


class PatchEmbed(nn.Module):
    """2D Image to Patch Embedding for DiT."""
    
    def __init__(self, img_size: int = 28, patch_size: int = 4, in_channels: int = 1, embed_dim: int = 192):
        """
        Initialize PatchEmbed.
        
        Args:
            img_size (int): Size of input image (assumes square)
            patch_size (int): Size of patches
            in_channels (int): Number of input channels
            embed_dim (int): Embedding dimension
        """
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of PatchEmbed.
        
        Args:
            x (torch.Tensor): Input image [B, C, H, W]
            
        Returns:
            torch.Tensor: Patch embeddings [B, num_patches, embed_dim]
        """
        B, C, H, W = x.shape
        x = self.proj(x)  # [B, embed_dim, H//patch_size, W//patch_size]
        x = x.flatten(2).transpose(1, 2)  # [B, num_patches, embed_dim]
        return x


class TimestepEmbedder(nn.Module):
    """Timestep embedding for conditioning DiT."""
    
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


class DiTBlock(nn.Module):
    """DiT transformer block with adaptive layer norm."""
    
    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float = 4.0):
        """
        Initialize DiTBlock.
        
        Args:
            hidden_size (int): Hidden dimension size
            num_heads (int): Number of attention heads
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim
        """
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, hidden_size)
        )
        
        # Adaptive layer norm parameters
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of DiTBlock.
        
        Args:
            x (torch.Tensor): Input features [B, N, D]
            c (torch.Tensor): Conditioning signal [B, D]
            
        Returns:
            torch.Tensor: Output features [B, N, D]
        """
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        
        # Self-attention block with adaptive layer norm
        x_norm = self.norm1(x)
        x_norm = x_norm * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + gate_msa.unsqueeze(1) * attn_out
        
        # MLP block with adaptive layer norm
        x_norm = self.norm2(x)
        x_norm = x_norm * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(x_norm)
        
        return x


class FinalLayer(nn.Module):
    """Final layer for DiT."""
    
    def __init__(self, hidden_size: int, patch_size: int, out_channels: int):
        """
        Initialize FinalLayer.
        
        Args:
            hidden_size (int): Hidden dimension size
            patch_size (int): Patch size for reconstruction
            out_channels (int): Number of output channels
        """
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of FinalLayer.
        
        Args:
            x (torch.Tensor): Input features [B, N, D]
            c (torch.Tensor): Conditioning signal [B, D]
            
        Returns:
            torch.Tensor: Output patches [B, N, patch_size*patch_size*out_channels]
        """
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = self.norm_final(x) * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        x = self.linear(x)
        return x


class DiT(nn.Module):
    """Diffusion Transformer (DiT) for MNIST."""
    
    def __init__(
        self, 
        in_channels: int = 1, 
        out_channels: int = 1, 
        img_size: int = 28,
        patch_size: int = 4,
        embed_dim: int = 192,
        depth: int = 12,
        num_heads: int = 3,
        mlp_ratio: float = 4.0
    ):
        """
        Initialize DiT.
        
        Args:
            in_channels (int): Number of input channels
            out_channels (int): Number of output channels
            img_size (int): Input image size (assumes square)
            patch_size (int): Patch size
            embed_dim (int): Embedding dimension
            depth (int): Number of transformer blocks
            num_heads (int): Number of attention heads
            mlp_ratio (float): MLP ratio
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        
        # Patch embedding
        self.x_embedder = PatchEmbed(img_size, patch_size, in_channels, embed_dim)
        
        # Timestep embedding
        self.t_embedder = TimestepEmbedder(embed_dim)
        
        # Positional embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim), requires_grad=False)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            DiTBlock(embed_dim, num_heads, mlp_ratio) for _ in range(depth)
        ])
        
        # Final layer
        self.final_layer = FinalLayer(embed_dim, patch_size, out_channels)
        
        self.initialize_weights()

    def initialize_weights(self):
        """Initialize weights following DiT paper."""
        # Initialize positional embeddings
        pos_embed = self._get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.num_patches**0.5))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        
        # Initialize patch embedding
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        
        # Initialize all linear layers with reasonable defaults
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        
        self.apply(_basic_init)

    def _get_2d_sincos_pos_embed(self, embed_dim: int, grid_size: int, temperature: int = 10000):
        """Get 2D sinusoidal positional embedding."""
        import numpy as np
        
        grid_h = np.arange(grid_size, dtype=np.float32)
        grid_w = np.arange(grid_size, dtype=np.float32)
        grid = np.meshgrid(grid_w, grid_h)  # here w goes first
        grid = np.stack(grid, axis=0)

        grid = grid.reshape([2, 1, grid_size, grid_size])
        pos_embed = self._get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
        return pos_embed

    def _get_2d_sincos_pos_embed_from_grid(self, embed_dim: int, grid):
        """Get 2D sinusoidal positional embedding from grid."""
        import numpy as np
        
        assert embed_dim % 2 == 0

        # use half of dimensions to encode grid_h
        emb_h = self._get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
        emb_w = self._get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

        emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
        return emb

    def _get_1d_sincos_pos_embed_from_grid(self, embed_dim: int, pos):
        """Get 1D sinusoidal positional embedding from grid."""
        import numpy as np
        
        assert embed_dim % 2 == 0
        omega = np.arange(embed_dim // 2, dtype=np.float64)
        omega /= embed_dim / 2.
        omega = 1. / 10000**omega  # (D/2,)

        pos = pos.reshape(-1)  # (M,)
        out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

        emb_sin = np.sin(out)  # (M, D/2)
        emb_cos = np.cos(out)  # (M, D/2)

        emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
        return emb

    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        """
        Unpatchify patches to image.
        
        Args:
            x (torch.Tensor): Patches [B, N, patch_size**2 * C]
            
        Returns:
            torch.Tensor: Images [B, C, H, W]
        """
        c = self.out_channels
        p = self.patch_size
        h = w = int(x.shape[1]**0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs

    def forward(self, x: torch.Tensor, t: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass of DiT.
        
        Args:
            x (torch.Tensor): Input images [B, C, H, W]
            t (torch.Tensor): Timesteps [B]
            
        Returns:
            torch.Tensor: Predicted noise [B, C, H, W]
        """
        # Patch embedding
        x = self.x_embedder(x)  # [B, N, D]
        x = x + self.pos_embed  # Add positional embedding
        
        # Timestep embedding
        if t is not None:
            t = self.t_embedder(t)  # [B, D]
        else:
            # If no timestep provided, use zeros
            t = torch.zeros(x.shape[0], x.shape[-1], device=x.device)
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x, t)
        
        # Final layer
        x = self.final_layer(x, t)  # [B, N, patch_size**2 * out_channels]
        
        # Unpatchify
        x = self.unpatchify(x)  # [B, out_channels, H, W]
        
        return x
