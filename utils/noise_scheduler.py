"""Noise scheduling utilities for diffusion models."""

import sys
import torch
import torch.nn.functional as F
from config import Config
from typing import Optional


class NoiseScheduler:
    """Utility class for managing noise schedules in diffusion models."""

    def __init__(self, timesteps: int = Config.TIMESTEPS, beta_start: float = Config.BETA_START,
                 beta_end: float = Config.BETA_END, schedule_type: str = Config.SCHEDULER_TYPE, device: torch.device = Config.DEVICE):
        """
        Initialize noise scheduler.
        
        Args:
            timesteps (int): Number of diffusion timesteps
            beta_start (float): Starting noise variance
            beta_end (float): Ending noise variance
            schedule_type (str): Type of schedule ('linear', 'cosine')
            device (torch.device): Device for computations
        """
        self.timesteps = timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.schedule_type = schedule_type
        self.device = device
        
    # Precompute noise schedule for all timesteps
        if self.schedule_type == 'cosine':
            self.betas = self.cosine_beta_schedule()
        else:
            self.betas = self.linear_beta_schedule()
            
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alpha_bars = torch.sqrt(self.alpha_bars)
        self.sqrt_one_minus_alpha_bars = torch.sqrt(1.0 - self.alpha_bars)

    def cosine_beta_schedule(self, s: float = 0.008) -> torch.Tensor:
        """
        Cosine noise schedule as proposed in https://arxiv.org/abs/2102.09672
        
        Args:
            s (float): Small offset parameter
            
        Returns:
            torch.Tensor: Cosine beta schedule tensor
        """
        steps = self.timesteps + 1
        t = torch.linspace(0, self.timesteps, steps, dtype=torch.float32, device=self.device)
        alpha_bar = torch.cos((t / self.timesteps + s)/(1+s) * torch.pi / 2)**2
        alpha_bar = alpha_bar / alpha_bar[0]
        betas = 1 - alpha_bar[1:] / alpha_bar[:-1]
        return torch.clamp(betas, min=0, max=0.99)

    def linear_beta_schedule(self) -> torch.Tensor:
        """
        Linear noise schedule.

        Returns:
            torch.Tensor: Linear beta schedule tensor
        """
        return torch.linspace(self.beta_start, self.beta_end, self.timesteps, dtype=torch.float32, device=self.device)

    def add_noise(self, x0: torch.Tensor, t:torch.Tensor, noise: Optional[torch.Tensor] = None) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Add noise to clean images according to forward diffusion process.

        Args:
            x0 (torch.Tensor): Clean images
            t (torch.Tensor): Timesteps
            noise (torch.Tensor, optional): Noise tensor, will be sampled if None

        Returns:
            tuple[torch.Tensor, torch.Tensor]: (noisy_images, noise)
        """
        if noise is None:
            noise = torch.randn_like(x0)
        
    # Retrieve alpha_bar values for specified timesteps
    # Debug: print shape of t tensor
        alpha_bar_t = self.alpha_bars[t].view(-1, *([1] * (x0.ndim - 1)))
        sqrt_alpha_bar_t = torch.sqrt(alpha_bar_t)
        sqrt_one_minus_alpha_bar_t = torch.sqrt(1.0 - alpha_bar_t)

    # Apply forward diffusion equation: x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * noise
        x_t = sqrt_alpha_bar_t * x0 + sqrt_one_minus_alpha_bar_t * noise
        return x_t, noise