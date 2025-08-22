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
import re
from typing import List, Tuple

from utils.data_loader import get_mnist_dataloader
from models.diffusion import DiffusionModel
from utils.tensorboard_logger import TensorBoardLogger
from config import Config

class FIDValidator:
    def __init__(self, device=Config.DEVICE):
        self.device = device
        # Force FID computation to use CPU to avoid MPS linalg issues
        self.fid = FrechetInceptionDistance(feature_dim=2048).to('cpu')
        
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
        # Move to CPU for FID computation to avoid MPS linalg issues
        real_images = real_images.to('cpu').to(torch.float32).clamp(0, 1)
        generated_images = generated_images.to('cpu').to(torch.float32).clamp(0, 1)

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

def get_checkpoint_files(checkpoint_dir: str) -> List[Tuple[int, str]]:
    """
    Get all checkpoint files and their epoch numbers from a directory.
    
    Args:
        checkpoint_dir (str): Directory containing checkpoint files
        
    Returns:
        List[Tuple[int, str]]: List of (epoch_number, file_path) tuples sorted by epoch
    """
    checkpoint_files = []
    if not os.path.exists(checkpoint_dir):
        print(f"Checkpoint directory {checkpoint_dir} does not exist!")
        return checkpoint_files
        
    for filename in os.listdir(checkpoint_dir):
        if filename.endswith('.pth') and 'model_epoch_' in filename:
            # Extract epoch number from filename
            match = re.search(r'model_epoch_(\d+)\.pth', filename)
            if match:
                epoch_num = int(match.group(1))
                file_path = os.path.join(checkpoint_dir, filename)
                checkpoint_files.append((epoch_num, file_path))
    
    # Sort by epoch number
    checkpoint_files.sort(key=lambda x: x[0])
    return checkpoint_files

def validate_all_checkpoints(experiment_name: str, model_type: str, num_samples: int = 512):
    """
    Validate all checkpoints in an experiment and log results to TensorBoard.
    
    Args:
        experiment_name (str): Name of the experiment directory
        model_type (str): Type of model (unet, cnn, resnet)
        num_samples (int): Number of samples to use for FID computation
    """
    # Set up paths and configuration
    original_model_type = Config.MODEL_TYPE
    Config.MODEL_TYPE = model_type  # type: ignore
    checkpoint_dir = os.path.join(Config.CHECKPOINT_DIR, experiment_name)
    
    # Initialize TensorBoard logger for validation results
    logger = TensorBoardLogger(Config.TENSORBOARD_LOG_DIR, f"{experiment_name}_validation")
    
    # Create FIDValidator instance
    validator = FIDValidator()
    
    # Get real data for comparison
    real_dataloader = get_mnist_dataloader()
    
    # Get all checkpoint files
    checkpoint_files = get_checkpoint_files(checkpoint_dir)
    
    if not checkpoint_files:
        print(f"No checkpoint files found in {checkpoint_dir}")
        logger.close()
        return
    
    print(f"Found {len(checkpoint_files)} checkpoints to validate")
    print(f"Using {num_samples} samples for FID computation")
    print(f"Logging validation results to TensorBoard: {logger.log_dir}")
    
    fid_scores = []
    epochs = []
    
    for epoch_num, checkpoint_path in checkpoint_files:
        print(f"\nValidating checkpoint: {os.path.basename(checkpoint_path)} (Epoch {epoch_num})")
        
        try:
            # Load model
            model = DiffusionModel()
            model.load_state_dict(torch.load(checkpoint_path, map_location=Config.DEVICE))
            model.to(Config.DEVICE)
            
            # Compute FID score
            fid_score = validator.validate_model(model, real_dataloader, num_samples=num_samples)
            
            # Log to TensorBoard
            logger.log_scalar('Validation/FID_Score', fid_score, epoch_num)
            
            # Store for summary
            fid_scores.append(fid_score)
            epochs.append(epoch_num)
            
            print(f"Epoch {epoch_num}: FID Score = {fid_score:.4f}")
            
            # Clean up model to free memory
            del model
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
        except Exception as e:
            print(f"Error validating checkpoint {checkpoint_path}: {str(e)}")
            continue
    
    # Log summary statistics
    if fid_scores:
        best_fid = min(fid_scores)
        best_epoch = epochs[fid_scores.index(best_fid)]
        worst_fid = max(fid_scores)
        worst_epoch = epochs[fid_scores.index(worst_fid)]
        avg_fid = sum(fid_scores) / len(fid_scores)
        
        print(f"\n=== Validation Summary ===")
        print(f"Best FID Score: {best_fid:.4f} (Epoch {best_epoch})")
        print(f"Worst FID Score: {worst_fid:.4f} (Epoch {worst_epoch})")
        print(f"Average FID Score: {avg_fid:.4f}")
        
        # Log summary statistics
        logger.log_scalar('Validation/Best_FID', best_fid, 0)
        logger.log_scalar('Validation/Worst_FID', worst_fid, 0)
        logger.log_scalar('Validation/Average_FID', avg_fid, 0)
        
        # Log the epoch numbers for best/worst performance
        logger.log_scalar('Validation/Best_FID_Epoch', best_epoch, 0)
        logger.log_scalar('Validation/Worst_FID_Epoch', worst_epoch, 0)
    
    logger.close()
    print(f"\nValidation complete! TensorBoard logs saved to: {logger.log_dir}")

def main():
    """Main entry point for validation script with argument parsing."""
    parser = argparse.ArgumentParser(description="Validate diffusion model checkpoints")
    parser.add_argument('--experiment_name', type=str, required=True, 
                       help='Experiment name (checkpoint directory name)')
    parser.add_argument('--model_type', type=str, default=Config.MODEL_TYPE, 
                       choices=['unet', 'cnn', 'resnet'], 
                       help='Type of model to use')
    parser.add_argument('--model_index', type=str, default=None,
                       help='Specific model epoch to validate (if not provided, validates all)')
    parser.add_argument('--num_samples', type=int, default=512,
                       help='Number of samples to use for FID computation')
    parser.add_argument('--validate_all', action='store_true',
                       help='Validate all checkpoints in the experiment')
    
    args = parser.parse_args()
    
    if args.validate_all or args.model_index is None:
        # Validate all checkpoints
        validate_all_checkpoints(args.experiment_name, args.model_type, args.num_samples)
    else:
        # Validate single checkpoint (original functionality)
        Config.MODEL_TYPE = args.model_type  # type: ignore
        
        # Create FIDValidator instance for evaluation
        validator = FIDValidator()

        real_dataloader = get_mnist_dataloader()
        model = DiffusionModel()
        model_path = os.path.join(Config.CHECKPOINT_DIR, args.experiment_name, f"model_epoch_{args.model_index}.pth")
        
        if not os.path.isfile(model_path):
            print(f"Error: Model file not found at '{model_path}'. Please check the path and try again.")
            sys.exit(1)
            
        model.load_state_dict(torch.load(model_path, map_location=Config.DEVICE))
        model.to(Config.DEVICE)
        fid_score = validator.validate_model(model, real_dataloader, num_samples=args.num_samples)
        print(f"FID Score: {fid_score:.4f}")
    

if __name__ == "__main__":
    main()