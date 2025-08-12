# MNIST Diffusion Model - Implementation Checklist

## Quick Start Guide

This document provides a step-by-step checklist for implementing the diffusion model. Follow this order for best results.

---

## Phase 1: Basic Setup

### ✅ 1. Create `requirements.txt`
```
torch>=1.9.0
torchvision>=0.10.0
matplotlib>=3.3.0
numpy>=1.21.0
tqdm>=4.62.0
Pillow>=8.3.0
```

### ✅ 2. Create `config.py`
- [ ] Define `Config` class with all hyperparameters
- [ ] Set data parameters (batch_size=128, image_size=28, channels=1)
- [ ] Set model parameters (model_channels=64, time_embed_dim=256)
- [ ] Set diffusion parameters (timesteps=1000, beta_start=1e-4, beta_end=0.02)
- [ ] Set training parameters (epochs=100, lr=1e-4)
- [ ] Set paths and device configuration

---

## Phase 2: Data and Utilities

### ✅ 3. Implement `utils/data_loader.py`
- [ ] `get_mnist_dataloader()`: Load MNIST, normalize to [-1, 1], return DataLoader
- [ ] `denormalize_images()`: Convert [-1, 1] back to [0, 1]
- [ ] `normalize_images()`: Convert [0, 1] to [-1, 1]

**Testing**: Load a batch and verify image normalization

### ✅ 4. Implement `utils/visualization.py`
- [ ] `save_images()`: Save image grids using torchvision.utils.make_grid
- [ ] `plot_images()`: Display images with matplotlib
- [ ] `plot_loss()`: Plot training curves

**Testing**: Load MNIST batch and visualize images

---

## Phase 3: Core Model Architecture

### ✅ 5. Implement `models/unet.py`

#### TimeEmbedding Class
- [ ] `__init__()`: Store embed_dim
- [ ] `forward()`: 
  - Use sinusoidal encoding: `emb = torch.arange(half_dim) * -math.log(10000) / (half_dim - 1)`
  - Apply to timesteps: `emb = timesteps[:, None] * torch.exp(emb)[None, :]`
  - Concatenate sin and cos: `torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)`

#### DownBlock Class
- [ ] `__init__()`: Conv2d layers, GroupNorm, Linear for time projection
- [ ] `forward()`: conv -> norm -> relu -> add time -> conv -> norm -> relu

#### UpBlock Class  
- [ ] `__init__()`: Similar to DownBlock but handles concatenation
- [ ] `forward()`: upsample -> concat with skip -> conv blocks

#### UNet Class
- [ ] `__init__()`: Create encoder (down), bottleneck, decoder (up) paths
- [ ] `forward()`: 
  - Encode with skip connections
  - Process through bottleneck
  - Decode with skip connections
  - Return predicted noise

**Testing**: Create UNet, pass random noise and timesteps, check output shape

---

## Phase 4: Diffusion Process

### ✅ 6. Implement `models/diffusion.py`

#### DiffusionModel Class
- [ ] `__init__()`: 
  - Initialize UNet
  - Create noise schedule: `betas = torch.linspace(beta_start, beta_end, timesteps)`
  - Precompute: `alphas = 1 - betas`, `alphas_cumprod = torch.cumprod(alphas, dim=0)`
  - Precompute: `sqrt_alphas_cumprod`, `sqrt_one_minus_alphas_cumprod`

- [ ] `forward_diffusion()`:
  - Sample noise if not provided
  - Apply: `xt = sqrt_alphas_cumprod[t] * x0 + sqrt_one_minus_alphas_cumprod[t] * noise`
  - Return (xt, noise)

- [ ] `reverse_diffusion_step()`:
  - Predict x0: `pred_x0 = (xt - sqrt_one_minus_alphas_cumprod[t] * pred_noise) / sqrt_alphas_cumprod[t]`
  - Compute posterior mean using pred_x0
  - Add noise if t > 0

- [ ] `forward()`: Call forward_diffusion + UNet prediction

- [ ] `sample()`: 
  - Start with random noise
  - Loop backwards through timesteps
  - Apply reverse_diffusion_step at each step

- [ ] `compute_loss()`: MSE between predicted and actual noise

**Testing**: Test forward/reverse diffusion, verify noise prediction

---

## Phase 5: Training Infrastructure

### ✅ 7. Implement `main.py`

- [ ] `train_model()`:
  - Setup device and directories
  - Load MNIST dataloader
  - Initialize model and optimizer (Adam)
  - Training loop:
    - Sample random timesteps for each batch
    - Compute diffusion loss
    - Backpropagate and update
    - Log progress
  - Save checkpoints periodically
  - Generate sample images during training

- [ ] `main()`: Parse arguments, call train_model()

**Testing**: Run short training (few batches) to verify training loop works

---

## Phase 6: Generation and Evaluation

### ✅ 8. Implement `generate.py`

- [ ] `load_model()`: Load checkpoint and initialize model
- [ ] `generate_images()`: Use model.sample() to generate images
- [ ] `generate_interpolation()`: Interpolate between noise vectors
- [ ] `main()`: Parse arguments, load model, generate images

**Testing**: Generate images from a trained checkpoint

### ✅ 9. Optional: Implement `utils/noise_scheduler.py`
- [ ] Alternative noise schedules (cosine)
- [ ] Utility functions for noise scheduling

---

## Key Implementation Tips

### Mathematical Formulas to Implement:

**Forward Diffusion (q(x_t | x_0))**:
```
x_t = √(ᾱ_t) * x_0 + √(1 - ᾱ_t) * ε
```

**Reverse Diffusion Mean**:
```
μ_t = 1/√(α_t) * (x_t - β_t/√(1-ᾱ_t) * ε_θ(x_t, t))
```

**Training Loss**:
```
L = ||ε - ε_θ(x_t, t)||²
```

### Common Debugging Steps:

1. **Shape Checking**: Verify tensor shapes at each step
2. **Visualization**: Plot noisy images at different timesteps
3. **Loss Monitoring**: Loss should decrease over training
4. **Sample Quality**: Generated images should improve over epochs

### Performance Tips:

1. **Precompute Values**: Store sqrt_alphas_cumprod, etc. as buffers
2. **GPU Usage**: Move tensors to device efficiently
3. **Memory Management**: Use torch.no_grad() during evaluation
4. **Checkpointing**: Save model state regularly

---

## Validation Checklist

Before considering implementation complete:

- [ ] MNIST images load and display correctly
- [ ] Forward diffusion gradually adds noise
- [ ] U-Net predicts noise with correct output shape
- [ ] Training loss decreases over time
- [ ] Generated samples look like MNIST digits
- [ ] Checkpointing and loading works
- [ ] Generation script produces new images

---

## Expected Timeline

- **Phase 1-2**: 1-2 hours (setup and data)
- **Phase 3**: 2-3 hours (U-Net implementation)
- **Phase 4**: 2-3 hours (diffusion process)
- **Phase 5**: 1-2 hours (training loop)
- **Phase 6**: 1 hour (generation)
- **Testing/Debugging**: 2-4 hours

**Total**: 8-15 hours depending on experience level

This checklist provides a practical roadmap for implementing your diffusion model step by step!
