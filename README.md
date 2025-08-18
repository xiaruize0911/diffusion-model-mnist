# Diffusion Model for MNIST Generation

This repository provides a PyTorch implementation of a diffusion model for generating MNIST digit images. The project includes training, validation, and image generation scripts, as well as modular components for model architecture, data loading, and visualization.

## Project Structure

```
Diffusion_Model/
├── README.md               # Project documentation
├── requirements.txt        # Python dependencies
├── config.py               # Configuration settings
├── main.py                 # Main training script
├── generate.py             # Image generation script
├── validation.py           # Model validation and FID computation
├── models/
│   ├── __init__.py
│   ├── unet.py             # U-Net architecture
│   ├── SimpleCNN.py        # Simple CNN architecture
│   ├── ResNet.py           # ResNet architecture
│   └── diffusion.py        # Diffusion process implementation
├── utils/
│   ├── __init__.py
│   ├── data_loader.py      # MNIST data loading utilities
│   ├── noise_scheduler.py  # Noise scheduling utilities
│   ├── tensorboard_logger.py # TensorBoard logging utilities
│   └── visualization.py    # Image plotting and saving
├── checkpoints/            # Model checkpoints directory
├── outputs/                # Generated images directory
├── runs/                   # TensorBoard logs
└── data/                   # MNIST dataset
```

## Installation

Install all required Python packages:

```bash
pip install -r requirements.txt
```

## Usage

### Training

Train a diffusion model on MNIST:

```bash
python main.py
# Or with custom parameters:
python main.py --epochs 200 --batch_size 64 --lr 1e-4
```

### Generation

Generate new MNIST samples from a trained model checkpoint:

```bash
python generate.py --checkpoint checkpoints/diffusion_model_unet_2000epochs_300timesteps_0.0001lr/model_epoch_1999.pth --num_samples 16
```

### Validation

Compute FID score for a trained model:

```bash
python validation.py --experiment_name 'diffusion_model_unet_2000epochs_300timesteps_0.0001lr' --model_type 'unet' --path_to_model 'checkpoints/diffusion_model_unet_2000epochs_300timesteps_0.0001lr/model_epoch_1999.pth'
```

## Key Components

- **U-Net**: Neural network for noise prediction in the diffusion process
- **Diffusion Process**: Implements forward and reverse diffusion steps
- **Training Loop**: Minimizes MSE loss between predicted and true noise
- **Generation**: Iteratively denoises random noise to produce images
- **Validation**: Computes FID score to evaluate generative quality

## Validation Scores

| Model Type | FID Score |
|------------|-----------|
| U-Net      |  34.3883  |
| CNN        |  57.8870  |
| ResNet     |  45.0198  |