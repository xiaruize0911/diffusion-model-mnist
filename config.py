"""Configuration settings for the diffusion model."""

import torch


class Config:
    """Configuration class containing all hyperparameters and settings."""
    
    # Data settings
    DATA_DIR = "./data"
    BATCH_SIZE = 128
    IMAGE_SIZE = 28
    CHANNELS = 1
    
    # Model settings
    MODEL_CHANNELS = 64
    SCHEDULER_TYPE = 'cosine'

    # Diffusion settings
    TIMESTEPS = 300
    BETA_START = 1e-4
    BETA_END = 0.02
    
    # Training settings
    EPOCHS = 10000
    LEARNING_RATE = 1e-4
    DEVICE = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')

    # Paths
    CHECKPOINT_DIR = "./checkpoints"
    OUTPUT_DIR = "./outputs"
    SAVE_EVERY = 100
    NUM_SAMPLES = 16

    # Save Config
    SAVE_EACH_EPOCHS = 100