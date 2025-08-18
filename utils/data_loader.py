"""Data loading utilities for MNIST dataset."""

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os
from config import Config


def get_mnist_dataloader(data_dir: str = Config.DATA_DIR, batch_size: int = Config.BATCH_SIZE, 
                        train: bool = True, download: bool = True) -> DataLoader:
    """
    Download MNIST dataset and create a DataLoader with normalization for diffusion models.
    
    Args:
        data_dir (str): Directory to store or load MNIST data
        batch_size (int): Number of samples per batch
        train (bool): Load training set if True, test set if False
        download (bool): Download MNIST if not already present
    
    Returns:
        DataLoader: PyTorch DataLoader for normalized MNIST dataset
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
    Transform images from [-1, 1] range to [0, 1] range for visualization or saving.
    
    Args:
        images (torch.Tensor): Input images in [-1, 1] range
    
    Returns:
        torch.Tensor: Output images in [0, 1] range
    """
    return (images + 1) / 2


def normalize_images(images: torch.Tensor) -> torch.Tensor:
    """
    Transform images from [0, 1] range to [-1, 1] range for model input.
    
    Args:
        images (torch.Tensor): Input images in [0, 1] range
    
    Returns:
        torch.Tensor: Output images in [-1, 1] range
    """
    return 2 * images - 1
