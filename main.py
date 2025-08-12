"""Main training script for MNIST diffusion model."""

import torch
import torch.optim as optim
import os
from tqdm import tqdm
import argparse

from config import Config
from models import DiffusionModel
from utils import get_mnist_dataloader, save_images, plot_loss


def train_model():
    """
    Train the diffusion model on MNIST.
    
    Returns:
        tuple: (trained_model, loss_history)
    """
    # TODO: 
    # 1. Setup device, create directories
    # 2. Load MNIST data
    # 3. Initialize model and optimizer
    # 4. Training loop with loss computation
    # 5. Periodic checkpointing and sample generation
    # 6. Return trained model and losses
    dataloader = get_mnist_dataloader()


def main() -> None:
    """
    Main function with argument parsing.
    
    Returns:
        None
    """
    # TODO: Parse command line arguments, update config, call train_model
    train_model()


if __name__ == "__main__":
    main()
