"""Image generation script for trained diffusion model."""

import torch
import os
import argparse
from typing import Optional
from config import Config
from models import DiffusionModel
from utils import save_images, plot_images


def load_model(checkpoint_path: str, device: torch.device) -> DiffusionModel:
    """
    Load trained diffusion model from checkpoint.
    
    Args:
        checkpoint_path (str): Path to model checkpoint
        device (torch.device): Computation device
        
    Returns:
        DiffusionModel: Loaded model
    """
    file_names = os.listdir(checkpoint_path)  # List all files in the checkpoint directory
    file_names.sort()
    latest_checkpoint = file_names[-1] if file_names else None
    
    if latest_checkpoint is None:
        raise FileNotFoundError(f"No checkpoint files found in {checkpoint_path}")
    
    checkpoint_full_path = os.path.join(checkpoint_path, latest_checkpoint)
    model = DiffusionModel().to(device)
    model.load_state_dict(torch.load(checkpoint_full_path, map_location=device))
    model.eval()  # Set model to evaluation mode for generation
    return model


def generate_images(model: DiffusionModel, num_samples: int, device: torch.device, 
                   save_path: Optional[str] = None, show_images: bool = True):
    """
    Generate images using the trained diffusion model.
    
    Args:
        model (DiffusionModel): Trained diffusion model
        num_samples (int): Number of samples to generate
        device (torch.device): Computation device
        save_path (Optional[str]): Optional path to save images
        show_images (bool): Whether to display images
        
    Returns:
        torch.Tensor: Generated images tensor
    """
    with torch.no_grad():
        generated_images = model.sample((num_samples, Config.CHANNELS, Config.IMAGE_SIZE, Config.IMAGE_SIZE), device)
    generated_images = torch.clamp(generated_images, 0.0, 1.0)  # Clamp generated image values to [0, 1]
    if save_path:
        save_images(generated_images, save_path)
    if show_images:
        plot_images(generated_images)
    return generated_images

def main() -> None:
    """
    Main function with argument parsing.
    
    Returns:
        None
    """
    model = load_model(Config.CHECKPOINT_DIR, Config.DEVICE)
    img = generate_images(model, Config.NUM_SAMPLES, Config.DEVICE, save_path=Config.OUTPUT_DIR+'pic.jpg', show_images=True)  # Generate and optionally display images, saving to file

if __name__ == "__main__":
    main()
