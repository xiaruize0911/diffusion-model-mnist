"""Diffusion model implementation."""

from utils.noise_scheduler import NoiseScheduler
import torch
import torch.nn as nn
import torch.nn.functional as F
from .unet import UNet
from .SimpleCNN import SimpleCNN
from .ResNet import ResNet
from config import Config
from utils.noise_scheduler import NoiseScheduler


class DiffusionModel(nn.Module):
    """Complete diffusion model with forward and reverse processes."""

    def __init__(self, timesteps: int = Config.TIMESTEPS, beta_start: float = Config.BETA_START, beta_end: float = Config.BETA_END,
                 in_channels: int = 1, out_channels: int = 1, mid_channels: int = Config.MODEL_CHANNELS):
        """
        Initialize DiffusionModel.
        
        Args:
            timesteps (int): Number of diffusion timesteps
            beta_start (float): Starting noise variance
            beta_end (float): Ending noise variance
            in_channels (int): Number of input image channels
            model_channels (int): Base number of model channels
            time_embed_dim (int): Dimension of time embedding
        """
        super().__init__()
        self.timesteps = timesteps
        self.beta_schedule = NoiseScheduler(timesteps=timesteps, beta_start=beta_start, beta_end=beta_end,schedule_type=Config.SCHEDULER_TYPE)
        if Config.MODEL_TYPE == 'unet':
            self.net = UNet(in_channels=in_channels, out_channels=out_channels, mid_channels=mid_channels)
        elif Config.MODEL_TYPE == 'cnn':
            self.net = SimpleCNN()
        elif Config.MODEL_TYPE == 'resnet':
            self.net = ResNet(in_channels=in_channels, out_channels=out_channels)

    def reverse_diffusion_step(self, xt: torch.Tensor, t: torch.Tensor, predicted_noise: torch.Tensor) -> torch.Tensor:
        """
        Single step of reverse diffusion process p(x_{t-1} | x_t).
        
        Args:
            xt (torch.Tensor): Noisy images at timestep t
            t (torch.Tensor): Current timestep
            predicted_noise (torch.Tensor): Noise predicted by the model
            
        Returns:
            torch.Tensor: Less noisy images x_{t-1} of shape (B, C, H, W)
        """
        alpha_t = self.beta_schedule.alphas[t].view(-1, 1, 1, 1)  # Scaling factor: controls retained portion of x_t
        beta_t = self.beta_schedule.betas[t].view(-1, 1, 1, 1)    # Variance added at current diffusion step
        alpha_bar_t = self.beta_schedule.alpha_bars[t].view(-1, 1, 1, 1)  # Cumulative product of alphas up to step t
        sigma_t = torch.sqrt(beta_t)
        mean = (1.0 / torch.sqrt(alpha_t)) * (
            xt - ((1.0 - alpha_t) / torch.sqrt(1.0 - alpha_bar_t)) * predicted_noise
        )
        return mean + sigma_t * torch.randn_like(xt)

    def forward(self, x0: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for training.
        
        Args:
            x0 (torch.Tensor): Clean images
            t (torch.Tensor): Random timesteps
            
        Returns:
            tuple[torch.Tensor, torch.Tensor]: (predicted_noise, actual_noise)
        """
        # Debug: print shape of t during forward diffusion
        noised_x, actual_noise = self.beta_schedule.add_noise(x0, t)
        # Debug: print shapes of noised_x and actual_noise
        predicted_noise = self.net(noised_x)
        return predicted_noise, actual_noise

    @torch.no_grad()
    def sample(self, shape: tuple, device: torch.device = Config.DEVICE) -> torch.Tensor:
        """
        Generate samples by reverse diffusion.
        
        Args:
            shape (tuple): Shape of samples to generate (B, C, H, W)
            device (torch.device): Device to generate samples on
        """
        # TODO: Implement iterative denoising starting from pure noise
        xt = torch.randn(shape, device=device)
        for t in reversed(range(self.beta_schedule.timesteps)):
            t_tensor = torch.full((shape[0],), t, device=device)
            predicted_noise = self.net(xt)
            xt = self.reverse_diffusion_step(xt, t_tensor, predicted_noise)
        return xt

    def compute_loss(self, predicted_noise: torch.Tensor, actual_noise: torch.Tensor) -> torch.Tensor:
        """
        Compute the diffusion loss (MSE between predicted and actual noise).
        
        Args:
            x0 (torch.Tensor): Clean images
            t (torch.Tensor): Random timesteps
            
        Returns:
            torch.Tensor: Scalar loss value
        """
        return F.mse_loss(predicted_noise, actual_noise)
