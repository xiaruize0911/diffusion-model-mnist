"""Data loading utilities for MNIST dataset."""

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os
from config import Config


def get_mnist_dataloader(data_dir: str = Config.DATA_DIR, batch_size: int = Config.BATCH_SIZE, 
                        train: bool = True, download: bool = True) -> DataLoader:
    """
    Create MNIST dataloader with appropriate transforms for diffusion model.
    
    Args:
        data_dir (str): Directory to store MNIST data
        batch_size (int): Batch size for dataloader
        train (bool): Whether to load training or test set
        download (bool): Whether to download MNIST if not present
    
    Returns:
        DataLoader: PyTorch DataLoader for MNIST dataset
    """
    mnist = datasets.MNIST(data_dir, train=train, download=download,
                            transform=transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,))
                            ]))
    dataloader = DataLoader(mnist,
                            batch_size=batch_size, shuffle=True)
    return dataloader


def denormalize_images(images: torch.Tensor) -> torch.Tensor:
    """
    Convert images from [-1, 1] range back to [0, 1] range.
    
    Args:
        images (torch.Tensor): Tensor of images in [-1, 1] range
    
    Returns:
        torch.Tensor: Images in [0, 1] range
    """
    return (images + 1) / 2


def normalize_images(images: torch.Tensor) -> torch.Tensor:
    """
    Convert images from [0, 1] range to [-1, 1] range.
    
    Args:
        images (torch.Tensor): Tensor of images in [0, 1] range
    
    Returns:
        torch.Tensor: Images in [-1, 1] range
    """
    return 2 * images - 1
