"""Configuration settings for the diffusion model."""

from ctypes.wintypes import SC_HANDLE
import torch
from transformers import SchedulerType


class Config:
    """Configuration class containing all hyperparameters and settings."""
    
    # Data settings
    DATA_DIR = "./data"
    BATCH_SIZE = 128
    IMAGE_SIZE = 28
    CHANNELS = 1
    
    # Model settings
    MODEL_CHANNELS = 64
    TIME_EMBED_DIM = 256
    SCHEDULER_TYPE = 'cosine'

    # Diffusion settings
    TIMESTEPS = 1000
    BETA_START = 1e-4
    BETA_END = 0.02
    
    # Training settings
    EPOCHS = 100
    LEARNING_RATE = 1e-4
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Paths
    CHECKPOINT_DIR = "./checkpoints"
    OUTPUT_DIR = "./outputs"
    SAVE_EVERY = 10
    NUM_SAMPLES = 16
    PRINT_EVERY = 100
