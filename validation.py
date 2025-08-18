import argparse
from operator import ge
import torch
from torchvision import transforms, models
from torch.utils.data import DataLoader
import numpy as np
from scipy.linalg import sqrtm
import torch.nn as nn
import torch.nn.functional as F
from torcheval.metrics import FrechetInceptionDistance
import os
import sys

from utils.data_loader import get_mnist_dataloader
from models.diffusion import DiffusionModel
from config import Config

class FIDValidator:
    def __init__(self, device=Config.DEVICE):
        self.device = device
        self.fid = FrechetInceptionDistance(feature_dim=2048).to(device)
        
    def compute_fid(self, real_images, generated_images):
        """
        Compute FID score between real and generated images
        
        Args:
            real_images: Tensor of real images [N, C, H, W]
            generated_images: Tensor of generated images [N, C, H, W]
        
        Returns:
            FID score as float
        """
    # Convert images to float32 and ensure values are in [0, 1] for FID computation
        real_images = real_images.to(torch.float32).clamp(0, 1)
        generated_images = generated_images.to(torch.float32).clamp(0, 1)

    # Update FID metric with real and generated images
        self.fid.update(real_images, is_real=True)
        self.fid.update(generated_images, is_real=False)

    # Calculate the FID score using the accumulated statistics
        fid_score = self.fid.compute()

    # Reset the FID metric to clear internal state for future computations
        self.fid.reset()

        return fid_score.item()
    
    def validate_model(self, model, real_dataloader, num_samples=1024):
        """
        Validate a generative model using FID score
        
        Args:
            model: Generative model
            real_dataloader: DataLoader with real images
            num_samples: Number of samples to generate
        
        Returns:
            FID score
        """
        model.eval()
        real_images = []
        generated_images = []
        
    # Gather real images from the dataloader
        with torch.no_grad():
            for batch_idx, (images, _) in enumerate(real_dataloader):
                real_images.append(images)
                if len(real_images) * images.size(0) >= num_samples:
                    break
        
        real_images = torch.cat(real_images, dim=0)[:num_samples]
        
    # Generate synthetic images using the diffusion model
        generated_images = model.sample(shape=(num_samples,1,28,28))
        
    # Move images to the target device and compute FID score
        real_images = real_images.to(self.device)
        generated_images = generated_images.to(self.device)

        if real_images.size(1) == 1:
            real_images = real_images.repeat(1, 3, 1, 1)
        if generated_images.size(1) == 1:
            generated_images = generated_images.repeat(1, 3, 1, 1)

        real_images = real_images.to(torch.float32)
        generated_images = generated_images.to(torch.float32)
        fid_score = self.compute_fid(real_images, generated_images)
        
        return fid_score

# Main entry point for validation script
def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_name', type=str, default=None, help='Custom experiment name for TensorBoard')
    parser.add_argument('--model_type', type=str, default=Config.MODEL_TYPE, choices=['unet', 'cnn', 'resnet'], help='Type of model to use (unet or cnn)')
    parser.add_argument('--model_index', type=str, help='The index number of the model')
    args = parser.parse_args()
    Config.MODEL_TYPE = args.model_type

    # Create FIDValidator instance for evaluation
    validator = FIDValidator()

    real_dataloader = get_mnist_dataloader()
    model = DiffusionModel()
    model_path = os.path.join(Config.CHECKPOINT_DIR,args.experiment_name, f"model_epoch_{args.model_index}.pth")
    if not os.path.isfile(model_path):
        print(f"Error: Model file not found at '{model_path}'. Please check the path and try again.")
        sys.exit(1)
    model.load_state_dict(torch.load(model_path, map_location=Config.DEVICE))
    model.to(Config.DEVICE)
    fid_score = validator.validate_model(model, real_dataloader)
    print(f"FID Score: {fid_score:.4f}")
    

if __name__ == "__main__":
    main()