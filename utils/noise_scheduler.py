"""Noise scheduling utilities for diffusion models."""

import torch
import torch.nn.functional as F
from zmq import NULL
from config import Config
from typing import Optional


class NoiseScheduler:
    """Utility class for managing noise schedules in diffusion models."""
    
    def __init__(self, timesteps: int = 1000, beta_start: float = 1e-4, 
                 beta_end: float = 0.02, schedule_type: str = 'linear'):
        """
        Initialize noise scheduler.
        
        Args:
            timesteps (int): Number of diffusion timesteps
            beta_start (float): Starting noise variance
            beta_end (float): Ending noise variance
            schedule_type (str): Type of schedule ('linear', 'cosine')
        """
        # TODO: Initialize betas, alphas, cumulative products, useful precomputed values
        self.timesteps = timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.schedule_type = schedule_type
        self.device = Config.DEVICE

    def cosine_beta_schedule(self, s: float = 0.008) -> torch.Tensor:
        """
        Cosine noise schedule as proposed in https://arxiv.org/abs/2102.09672
        
        Args:
            s (float): Small offset parameter
            
        Returns:
            torch.Tensor: Cosine beta schedule tensor
        """
        # TODO: Implement cosine schedule as in improved DDPM paper
        steps = self.timesteps + 1
        t = torch.linspace(0, self.timesteps, steps, dtype=torch.float64, device=Config.DEVICE)
        alpha_bar = torch.cos((t / self.timesteps + s)/(1+s) * torch.pi / 2)**2
        alpha_bar = alpha_bar / alpha_bar[0]
        betas = 1 - alpha_bar[1:] / alpha_bar[:-1]
        return torch.clamp(betas, min=0, max =0.99)

    def linear_beta_schedule(self) -> torch.Tensor:
        """
        Linear noise schedule.

        Returns:
            torch.Tensor: Linear beta schedule tensor
        """
        return torch.linspace(self.beta_start, self.beta_end, self.timesteps, dtype=torch.float64, device=Config.DEVICE)

    def sample_timesteps(self, batch_size: int, device: torch.device = Config.DEVICE):
        """
        Sample random timesteps for training.
        
        Args:
            batch_size (int): Number of timesteps to sample
            device (torch.device): Computation device
            
        Returns:
            torch.Tensor: Random timesteps of shape (batch_size,)
        """
        # TODO: Sample random integers from 0 to timesteps-1
        beta = None
        if self.schedule_type == 'cosine':
            beta= self.cosine_beta_schedule()
        elif self.schedule_type == 'linear':
            beta= self.linear_beta_schedule()
        else:
            beta= self.linear_beta_schedule()
        alpha = 1.0 - beta
        alpha_bar = torch.cumprod(alpha, dim=0)
        return alpha,alpha_bar.to(device)

    def add_noise(self, x0: torch.Tensor, t: torch.Tensor, alpha_bars: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Add noise to clean images according to forward diffusion process.

        Args:
            x0 (torch.Tensor): Clean images
            t (torch.Tensor): Timesteps
            alpha_bars (torch.Tensor): Cumulative product of alphas
            noise (torch.Tensor, optional): Noise tensor, will be sampled if None

        Returns:
            torch.Tensor: Noisy images
        """
        if noise is None:
            noise = torch.randn_like(x0)
        a_bar = alpha_bars[t].view(-1, *([1] * (x0.ndim - 1)))
        sqrt_a_bar = torch.sqrt(a_bar)
        sqrt_1_minus_a_bar = torch.sqrt(1.0 - a_bar)

        x_t = sqrt_a_bar * x0 + sqrt_1_minus_a_bar * noise
        return x_t