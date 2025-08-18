"""
TensorBoard logging utilities for diffusion model training.

This module provides a comprehensive TensorBoard logging interface
for monitoring diffusion model training progress including loss curves,
generated samples, model parameters, and training hyperparameters.
"""

import torch
from torch.utils.tensorboard import SummaryWriter
import os
from typing import Optional
import torchvision
from torchvision.utils import make_grid
import numpy as np


class TensorBoardLogger:
    """
    Comprehensive TensorBoard logger for diffusion model training monitoring.
    
    Provides methods for logging scalars, images, histograms, and other
    training metrics to TensorBoard for real-time monitoring and analysis.
    """
    
    def __init__(self, log_dir: str = "./runs", experiment_name: str = "diffusion_model"):
        """
        Initialize TensorBoard logger with specified directory structure.
        
        Args:
            log_dir (str): Base directory for all TensorBoard logs
            experiment_name (str): Unique name for this experiment run
        """
        self.log_dir = os.path.join(log_dir, experiment_name)
        os.makedirs(self.log_dir, exist_ok=True)
        self.writer = SummaryWriter(self.log_dir)
        self.step = 0
        
    def log_scalar(self, tag: str, value: float, step: Optional[int] = None):
        """
        Log scalar metrics like loss, learning rate, etc.
        
        Args:
            tag (str): Metric name (e.g., 'Loss/Training', 'Learning_Rate')
            value (float): Scalar value to log
            step (Optional[int]): Step number (uses internal step if None)
        """
        if step is None:
            step = self.step
        self.writer.add_scalar(tag, value, step)
        
    def log_images(self, tag: str, images: torch.Tensor, step: Optional[int] = None, 
                   nrow: int = 4, normalize: bool = True):
        """
        Log batch of images as a grid to TensorBoard.
        
        Args:
            tag (str): Image collection name (e.g., 'Generated/Samples')
            images (torch.Tensor): Batch of images with shape [B, C, H, W]
            step (Optional[int]): Step number (uses internal step if None)
            nrow (int): Number of images per row in the grid
            normalize (bool): Whether to normalize image pixel values
        """
        if step is None:
            step = self.step
            
    # Generate image grid and log to TensorBoard
        grid = make_grid(images, nrow=nrow, normalize=normalize, scale_each=True)
        self.writer.add_image(tag, grid, step)
        
    def log_histogram(self, tag: str, values: torch.Tensor, step: Optional[int] = None):
        """
        Log histogram of tensor values for distribution analysis.
        
        Args:
            tag (str): Histogram name (e.g., 'Weights/Layer1')
            values (torch.Tensor): Tensor values to analyze
            step (Optional[int]): Step number (uses internal step if None)
        """
        if step is None:
            step = self.step
        self.writer.add_histogram(tag, values, step)
        
    def log_model_parameters(self, model: torch.nn.Module, step: Optional[int] = None):
        """
        Log model parameter and gradient histograms for training analysis.
        
        Automatically logs weight and gradient distributions for all
        trainable parameters in the model.
        
        Args:
            model (torch.nn.Module): Model to analyze
            step (Optional[int]): Step number (uses internal step if None)
        """
        if step is None:
            step = self.step
            
        for name, param in model.named_parameters():
            if param.requires_grad:
                # Log model parameter distributions to TensorBoard
                self.log_histogram(f'Parameters/{name}', param.data, step)
                
                # Log gradient distributions if available
                if param.grad is not None:
                    self.log_histogram(f'Gradients/{name}', param.grad.data, step)
                    
    def log_learning_rate(self, optimizer: torch.optim.Optimizer, step: Optional[int] = None):
        """
        Log current learning rate from optimizer.
        
        Useful for monitoring learning rate schedules and decay.
        
        Args:
            optimizer (torch.optim.Optimizer): Optimizer to extract learning rate from
            step (Optional[int]): Step number (uses internal step if None)
        """
        if step is None:
            step = self.step
            
        for i, param_group in enumerate(optimizer.param_groups):
            lr = param_group['lr']
            self.log_scalar(f'Learning_Rate/Group_{i}', lr, step)
            
    def log_noise_schedule(self, betas: torch.Tensor, alphas: torch.Tensor, 
                          alpha_bars: torch.Tensor, step: Optional[int] = None):
        """
        Log noise schedule parameters for diffusion process analysis.
        
        Visualizes how noise variance changes across timesteps,
        helping understand the diffusion process dynamics.
        
        Args:
            betas (torch.Tensor): Beta values (noise variance coefficients)
            alphas (torch.Tensor): Alpha values (1 - beta)
            alpha_bars (torch.Tensor): Cumulative alpha products
            step (Optional[int]): Step number (uses internal step if None)
        """
        if step is None:
            step = self.step
            
    # Log noise schedule values as histograms in TensorBoard
        self.log_histogram('Noise_Schedule/Betas', betas, step)
        self.log_histogram('Noise_Schedule/Alphas', alphas, step)
        self.log_histogram('Noise_Schedule/Alpha_Bars', alpha_bars, step)
            
    def increment_step(self):
        """Increment internal step counter for automatic step tracking."""
        self.step += 1
        
    def close(self):
        """Close TensorBoard writer and flush remaining data."""
        self.writer.close()
        
    def __enter__(self):
        """Context manager entry."""
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with automatic cleanup."""
        self.close()