"""Configuration settings for the diffusion model."""

import torch


class Config:
    """Configuration class containing all hyperparameters and settings."""
    
    # Configuration for dataset and data loading
    DATA_DIR = "./data"
    BATCH_SIZE = 128
    IMAGE_SIZE = 28
    CHANNELS = 1
    
    # Model architecture and hyperparameter settings
    MODEL_CHANNELS = 64
    SCHEDULER_TYPE = 'cosine'
    MODEL_TYPE = 'unet'

    # Diffusion process parameters
    TIMESTEPS = 300
    BETA_START = 1e-4
    BETA_END = 0.02
    
    # Training loop parameters
    EPOCHS = 10000
    LEARNING_RATE = 1e-4
    DEVICE = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')

    # Directory paths for checkpoints, outputs, and logs
    CHECKPOINT_DIR = "./checkpoints"
    OUTPUT_DIR = "./outputs"
    TENSORBOARD_LOG_DIR = "./runs"
    SAVE_EVERY = 100
    NUM_SAMPLES = 16

    # Configuration saving options
    SAVE_EACH_EPOCHS = 100
    
    # TensorBoard logging and monitoring settings
    LOG_PARAMS_EVERY = 50  # Log model parameters every N epochs