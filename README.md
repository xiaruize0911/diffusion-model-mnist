# MNIST Diffusion Model

A PyTorch implementation of a diffusion model for generating MNIST digit images from scratch.

## Project Structure

```
Diffusion_Model/
├── README.md                 # Project documentation
├── requirements.txt          # Python dependencies
├── config.py                # Configuration settings
├── main.py                  # Main training script
├── generate.py              # Image generation script
├── models/
│   ├── __init__.py
│   ├── unet.py             # U-Net architecture
│   └── diffusion.py        # Diffusion process implementation
├── utils/
│   ├── __init__.py
│   ├── data_loader.py      # MNIST data loading
│   ├── noise_scheduler.py  # Noise scheduling utilities
│   └── visualization.py    # Plotting and visualization
├── checkpoints/            # Model checkpoints directory
└── outputs/               # Generated images directory
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Training
```bash
# Basic training
python main.py

# Custom parameters
python main.py --epochs 200 --batch_size 64 --lr 1e-4
```

### Generation
```bash
# Generate 16 samples
python generate.py --checkpoint checkpoints/final_model.pth --num_samples 16

# Generate interpolation
python generate.py --checkpoint checkpoints/final_model.pth --interpolation --interp_steps 10
```

## Implementation Guide

Follow the `PROJECT_STRUCTURE.md` and `IMPLEMENTATION_CHECKLIST.md` files for detailed implementation instructions.

## Key Components

1. **U-Net**: Neural network for noise prediction
2. **Diffusion Process**: Forward and reverse diffusion implementations
3. **Training Loop**: MSE loss between predicted and actual noise
4. **Generation**: Iterative denoising from pure noise

## Mathematical Foundation

- Forward: q(x_t | x_0) = N(√ᾱ_t · x_0, (1-ᾱ_t)I)
- Reverse: p_θ(x_{t-1} | x_t) learned by predicting noise
- Loss: E[||ε - ε_θ(x_t, t)||²]
