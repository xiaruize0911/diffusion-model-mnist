"""Visualization utilities for diffusion model."""

import matplotlib.pyplot as plt
import torch
import torchvision
from torchvision.utils import make_grid
import os
from typing import Optional


def save_images(images: torch.Tensor, filepath: str, nrow: int = 4, normalize: bool = True) -> None:
    """
    Save a batch of grayscale images as a single grid to a specified file.

    Args:
        images (torch.Tensor): Batch of images to save (shape: [N, 1, H, W]).
        filepath (str): Path to save the single image file.
        nrow (int): Number of images per row in the grid.
        normalize (bool): Whether to normalize images from [-1, 1] to [0, 1].

    Returns:
        None
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)  # Ensure the output directory exists

    # Create a grid of images and save as a single file
    grid = make_grid(images, nrow=nrow, normalize=normalize)
    torchvision.utils.save_image(grid, filepath)
    print(f"Image grid saved to {filepath}")


def plot_images(images: torch.Tensor, title: str = "Generated Images", 
               nrow: int = 4, figsize: tuple = (10, 10), normalize: bool = True) -> None:
    """
    Plot a grid of images using matplotlib.
    
    Args:
        images (torch.Tensor): Tensor of images to plot
        title (str): Title for the plot
        nrow (int): Number of images per row
        figsize (tuple): Figure size (width, height)
        normalize (bool): Whether to normalize images from [-1, 1] to [0, 1]
    
    Returns:
        None
    """
    grid = make_grid(images, nrow=nrow, normalize=normalize)
    plt.figure(figsize=figsize)
    plt.imshow(grid.permute(1, 2, 0).cpu().numpy())
    plt.title(title)
    plt.axis("off")
    plt.show()


def plot_loss(losses: list, save_path: Optional[str] = None, title: str = "Training Loss") -> None:
    """
    Plot training loss curve.
    
    Args:
        losses (list): List of loss values
        save_path (Optional[str]): Optional path to save the plot
        title (str): Title for the plot
    
    Returns:
        None
    """
    plt.plot(losses)
    plt.title(title)
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    if save_path:
        plt.savefig(save_path)
    plt.show()


def visualize_diffusion_process(model, x0: torch.Tensor, 
                               timesteps_to_show: list = [0, 100, 300, 500, 700, 999]) -> torch.Tensor:
    """
    Visualize the forward diffusion process by showing how noise is added over time.
    
    Args:
        model: DiffusionModel instance
        x0 (torch.Tensor): Clean image to add noise to
        timesteps_to_show (list): List of timesteps to visualize
    
    Returns:
        torch.Tensor: Concatenated noisy images at different timesteps
    """
    noisy_images = []
    for t in timesteps_to_show:
        noise = torch.randn_like(x0) * model.get_noise_std(t)
        x_t = model.forward_diffusion(x0, noise, t)
        noisy_images.append(x_t)
    return torch.cat(noisy_images, dim=0)
