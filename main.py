"""Main training script for MNIST diffusion model with TensorBoard monitoring."""

import torch
import torch.optim as optim
import os
from tqdm import tqdm
import argparse
from datetime import datetime

from generate import generate_images
from config import Config
from models import DiffusionModel
from models.modiff import MoDiffModel
from utils import get_mnist_dataloader, save_images, plot_loss, TensorBoardLogger

def train_model(experiment_name=None):
    """
    Train the diffusion model on MNIST with comprehensive TensorBoard monitoring.
    
    Args:
        experiment_name (str, optional): Custom experiment name for TensorBoard logging
    
    Returns:
        tuple: (trained_model, loss_history)
    """
    # Set up TensorBoard logging using either a timestamp or a custom experiment name
    if experiment_name is None:
        experiment_name = f"diffusion_mnist_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    logger = TensorBoardLogger(Config.TENSORBOARD_LOG_DIR, experiment_name)
    
    dataloader = get_mnist_dataloader(batch_size=Config.BATCH_SIZE)
    device = Config.DEVICE
    
    # Create base diffusion model
    base_model = DiffusionModel().to(device)
    
    # Wrap with MoDiff if enabled
    if Config.ENABLE_MODIFF:
        model = MoDiffModel(
            base_model=base_model,
            bit_width=Config.MODIFF_BIT_WIDTH,
            enable_quantization=False,  # Disable during training
            enable_error_compensation=Config.ENABLE_ERROR_COMPENSATION
        ).to(device)
        print(f"Using MoDiff with {Config.MODIFF_BIT_WIDTH}-bit quantization")
    else:
        model = base_model
        print("Using standard diffusion model")
    
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    loss_history = []
    
    # Log the model architecture and noise schedule to TensorBoard
    logger.log_noise_schedule(
        model.beta_schedule.betas,
        model.beta_schedule.alphas, 
        model.beta_schedule.alpha_bars,
        step=0
    )
    
    print(f"Starting training on {device}")
    print(f"TensorBoard logs: {logger.log_dir}")
    print(f"Run 'tensorboard --logdir {Config.TENSORBOARD_LOG_DIR}' to monitor training")

    global_step = 0
    for epoch in tqdm(range( Config.EPOCHS), desc="Training epochs"):
        epoch_losses = []
        
        for batch_idx, (batch, _) in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}", leave=False)):
            optimizer.zero_grad()
            x0 = batch.to(device)
            t = torch.randint(0, Config.TIMESTEPS, (x0.size(0),), device=device).long()
            pred_noise, actual_noise = model(x0, t)
            loss = model.compute_loss(pred_noise, actual_noise)
            loss.backward()
            optimizer.step()
            
            # Record training metrics for each batch
            logger.log_scalar('Loss/Train_Step', loss.item(), global_step)
            logger.log_learning_rate(optimizer, global_step)
            
            loss_history.append(loss.item())
            epoch_losses.append(loss.item())
            global_step += 1
            
            # Flush TensorBoard logs periodically for real-time visualization
            if global_step % 10 == 0:
                logger.writer.flush()
        
        # Log metrics and samples at the end of each epoch
        avg_epoch_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else 0
        logger.log_scalar('Loss/Epoch_Average', avg_epoch_loss, epoch)
        
        print(f'Epoch {epoch+1}: avg_loss {avg_epoch_loss:.6f}')
        
        # Save model checkpoints and generate sample images
        if (epoch % Config.SAVE_EACH_EPOCHS == 0) or (epoch == 1999):
            checkpoint_path = os.path.join(Config.CHECKPOINT_DIR, f"model_epoch_{epoch}.pth")
            os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")
            
            # Generate sample images and log them to TensorBoard
            with torch.no_grad():
                if Config.ENABLE_MODIFF and isinstance(model, MoDiffModel):
                    # Enable quantization for sampling
                    model.enable_quantization = True
                    generated_images = model.sample((Config.NUM_SAMPLES, Config.CHANNELS, Config.IMAGE_SIZE, Config.IMAGE_SIZE), device)
                    model.enable_quantization = False  # Disable for training
                    
                    # Log MoDiff statistics
                    stats = model.get_statistics()
                    logger.log_scalar('MoDiff/BOPs_Reduction', stats['theoretical_bops_reduction'], epoch)
                    logger.log_scalar('MoDiff/Memory_Savings', stats['memory_savings_ratio'], epoch)
                    logger.log_scalar('MoDiff/Quantization_Calls', stats['quantization_calls'], epoch)
                else:
                    generated_images = model.sample((Config.NUM_SAMPLES, Config.CHANNELS, Config.IMAGE_SIZE, Config.IMAGE_SIZE), device)
                
                generated_images = torch.clamp(generated_images, 0.0, 1.0)
                
                # Log generated images to TensorBoard
                logger.log_images('Generated/Samples', generated_images, epoch, nrow=4)
                
                # Save generated images to output directory
                save_path = Config.OUTPUT_DIR + f'/pic{epoch}.jpg'
                save_images(generated_images, save_path, normalize=False)
        
        # Log model parameters and gradients for analysis
        if epoch % Config.LOG_PARAMS_EVERY == 0:
            logger.log_model_parameters(model, epoch)
            
        # Ensure all TensorBoard logs are flushed to disk
        logger.writer.flush()

    logger.close()
    return model, loss_history


def main() -> None:
    """
    Main function with argument parsing for experiment configuration.
    
    Returns:
        None
    """
    parser = argparse.ArgumentParser(description="Train MNIST diffusion model with TensorBoard monitoring")
    parser.add_argument('--epochs', type=int, default=Config.EPOCHS, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=Config.LEARNING_RATE, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=Config.BATCH_SIZE, help='Batch size')
    parser.add_argument('--timesteps', type=int, default=Config.TIMESTEPS, help='Number of diffusion timesteps')
    parser.add_argument('--experiment_name', type=str, default=None, help='Custom experiment name for TensorBoard')
    parser.add_argument('--model_type', type=str, default=Config.MODEL_TYPE, choices=['unet', 'cnn', 'resnet', 'unet2', 'resnet2', 'dit'], help='Type of model to use')
    parser.add_argument('--enable_modiff', action='store_true', help='Enable MoDiff quantization')
    parser.add_argument('--modiff_bit_width', type=int, default=Config.MODIFF_BIT_WIDTH, choices=[3, 4, 6, 8], help='MoDiff quantization bit width')
    parser.add_argument('--disable_error_compensation', action='store_true', help='Disable MoDiff error compensation')
    args = parser.parse_args()
    
    # Update configuration parameters using command line arguments
    if args.epochs != Config.EPOCHS:
        Config.EPOCHS = args.epochs  
    if args.lr != Config.LEARNING_RATE:
        Config.LEARNING_RATE = args.lr  
    if args.batch_size != Config.BATCH_SIZE:
        Config.BATCH_SIZE = args.batch_size  
    if args.timesteps != Config.TIMESTEPS:
        Config.TIMESTEPS = args.timesteps  
    if args.model_type != Config.MODEL_TYPE:
        Config.MODEL_TYPE = args.model_type  
    if args.enable_modiff:
        Config.ENABLE_MODIFF = True
    if args.modiff_bit_width != Config.MODIFF_BIT_WIDTH:
        Config.MODIFF_BIT_WIDTH = args.modiff_bit_width
    if args.disable_error_compensation:
        Config.ENABLE_ERROR_COMPENSATION = False
    if args.experiment_name is not None:
        # Use custom experiment name if specified
        experiment_name = args.experiment_name
    else:
        modiff_suffix = f"_modiff_{Config.MODIFF_BIT_WIDTH}bit" if Config.ENABLE_MODIFF else ""
        experiment_name = f'diffusion_model_{Config.MODEL_TYPE}_{Config.EPOCHS}epochs_{Config.TIMESTEPS}timesteps_{Config.LEARNING_RATE}lr{modiff_suffix}'
    Config.CHECKPOINT_DIR = os.path.join(Config.CHECKPOINT_DIR, experiment_name)  
    Config.OUTPUT_DIR = os.path.join(Config.OUTPUT_DIR, experiment_name)  
    
    print(f"Training configuration:")
    print(f"  Epochs: {Config.EPOCHS}")
    print(f"  Learning Rate: {Config.LEARNING_RATE}")
    print(f"  Batch Size: {Config.BATCH_SIZE}")
    print(f"  Timesteps: {Config.TIMESTEPS}")
    print(f"  Model Type: {Config.MODEL_TYPE}")
    print(f"  Device: {Config.DEVICE}")
    if Config.ENABLE_MODIFF:
        print(f"  MoDiff: Enabled ({Config.MODIFF_BIT_WIDTH}-bit, error_compensation={Config.ENABLE_ERROR_COMPENSATION})")
    else:
        print(f"  MoDiff: Disabled")
    if experiment_name:
        print(f"  Experiment: {experiment_name}")
    
    train_model(experiment_name)


if __name__ == "__main__":
    main()
