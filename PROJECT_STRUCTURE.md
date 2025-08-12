# MNIST Diffusion Model - Project Structure Guide

## File Structure Overview

```
Diffusion_Model/
├── README.md
├── requirements.txt
├── config.py
├── main.py
├── generate.py
├── models/
│   ├── __init__.py
│   ├── unet.py
│   └── diffusion.py
├── utils/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── noise_scheduler.py
│   └── visualization.py
├── checkpoints/
└── outputs/
```

---

## 1. Configuration File (`config.py`)

### Purpose
Centralized configuration for all hyperparameters and settings.

### Structure
```python
class Config:
    # Data settings
    DATA_DIR = "./data"
    BATCH_SIZE = 128
    IMAGE_SIZE = 28
    CHANNELS = 1
    
    # Model settings
    MODEL_CHANNELS = 64
    TIME_EMBED_DIM = 256
    
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
```

---

## 2. Models Package (`models/`)

### 2.1 U-Net Architecture (`models/unet.py`)

#### Classes and Functions to Implement:

##### `TimeEmbedding(nn.Module)`
**Purpose**: Convert timestep integers to sinusoidal embeddings

```python
def __init__(self, embed_dim: int):
    # Input: embed_dim - dimension of time embedding
    # Initialize embedding dimension
    pass

def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
    # Input: timesteps - tensor of shape (batch_size,) with timestep values
    # Output: time embeddings of shape (batch_size, embed_dim)
    # Implementation: Use sinusoidal positional encoding
    pass
```

##### `DownBlock(nn.Module)`
**Purpose**: Downsampling block for U-Net encoder

```python
def __init__(self, in_channels: int, out_channels: int, time_embed_dim: int):
    # Input: in_channels, out_channels, time_embed_dim
    # Initialize: conv layers, normalization, time projection
    pass

def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
    # Input: x - feature maps (B, C, H, W), time_emb - time embeddings (B, embed_dim)
    # Output: processed features (B, out_channels, H, W)
    # Implementation: conv -> norm -> activation -> time integration -> conv -> norm -> activation
    pass
```

##### `UpBlock(nn.Module)`
**Purpose**: Upsampling block for U-Net decoder with skip connections

```python
def __init__(self, in_channels: int, out_channels: int, time_embed_dim: int):
    # Input: in_channels (includes skip connection), out_channels, time_embed_dim
    # Initialize: conv layers, normalization, time projection
    pass

def forward(self, x: torch.Tensor, skip: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
    # Input: x - upsampled features, skip - skip connection, time_emb - time embeddings
    # Output: processed features (B, out_channels, H*2, W*2)
    # Implementation: upsample -> concat with skip -> conv blocks with time integration
    pass
```

##### `UNet(nn.Module)`
**Purpose**: Main U-Net architecture

```python
def __init__(self, in_channels: int = 1, model_channels: int = 64, time_embed_dim: int = 256):
    # Input: in_channels, base model channels, time embedding dimension
    # Initialize: time embedding, down blocks, bottleneck, up blocks, output conv
    pass

def forward(self, x: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
    # Input: x - noisy images (B, C, H, W), timesteps - timestep values (B,)
    # Output: predicted noise (B, C, H, W)
    # Implementation: encoder path -> bottleneck -> decoder path with skip connections
    pass
```

### 2.2 Diffusion Process (`models/diffusion.py`)

##### `DiffusionModel(nn.Module)`
**Purpose**: Complete diffusion model with forward and reverse processes

```python
def __init__(self, timesteps: int = 1000, beta_start: float = 1e-4, beta_end: float = 0.02, 
             in_channels: int = 1, model_channels: int = 64, time_embed_dim: int = 256):
    # Input: diffusion parameters and model architecture parameters
    # Initialize: U-Net, noise schedule, precomputed values (alphas, betas, etc.)
    pass

def forward_diffusion(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor = None) -> tuple[torch.Tensor, torch.Tensor]:
    # Input: x0 - clean images (B, C, H, W), t - timesteps (B,), noise - optional noise tensor
    # Output: (noisy_images, noise) both of shape (B, C, H, W)
    # Implementation: q(x_t | x_0) = N(sqrt(alpha_cumprod_t) * x0, (1 - alpha_cumprod_t) * I)
    pass

def reverse_diffusion_step(self, xt: torch.Tensor, t: torch.Tensor, predicted_noise: torch.Tensor) -> torch.Tensor:
    # Input: xt - noisy images, t - current timestep, predicted_noise - model prediction
    # Output: x_{t-1} - less noisy images (B, C, H, W)
    # Implementation: p(x_{t-1} | x_t) using predicted noise and posterior mean/variance
    pass

def forward(self, x0: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    # Input: x0 - clean images, t - random timesteps
    # Output: (predicted_noise, actual_noise)
    # Implementation: forward diffusion + noise prediction
    pass

@torch.no_grad()
def sample(self, shape: tuple, device: torch.device) -> torch.Tensor:
    # Input: shape - (B, C, H, W), device - computation device
    # Output: generated samples (B, C, H, W)
    # Implementation: start from noise, iteratively denoise for all timesteps
    pass

def compute_loss(self, x0: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    # Input: x0 - clean images, t - random timesteps
    # Output: scalar loss value
    # Implementation: MSE loss between predicted and actual noise
    pass
```

---

## 3. Utilities Package (`utils/`)

### 3.1 Data Loading (`utils/data_loader.py`)

```python
def get_mnist_dataloader(data_dir: str = "./data", batch_size: int = 128, 
                        train: bool = True, download: bool = True) -> DataLoader:
    # Input: data directory, batch size, train/test flag, download flag
    # Output: PyTorch DataLoader
    # Implementation: Load MNIST, normalize to [-1, 1], create DataLoader
    pass

def denormalize_images(images: torch.Tensor) -> torch.Tensor:
    # Input: images in [-1, 1] range
    # Output: images in [0, 1] range
    # Implementation: (images + 1) / 2
    pass

def normalize_images(images: torch.Tensor) -> torch.Tensor:
    # Input: images in [0, 1] range  
    # Output: images in [-1, 1] range
    # Implementation: 2 * images - 1
    pass
```

### 3.2 Noise Scheduling (`utils/noise_scheduler.py`)

```python
class NoiseScheduler:
    def __init__(self, timesteps: int = 1000, beta_start: float = 1e-4, 
                 beta_end: float = 0.02, schedule_type: str = 'linear'):
        # Input: timesteps, beta range, schedule type
        # Initialize: betas, alphas, cumulative products, useful precomputed values
        pass
    
    def _cosine_beta_schedule(self, s: float = 0.008) -> torch.Tensor:
        # Input: s - small offset
        # Output: cosine beta schedule tensor
        # Implementation: cosine schedule as in improved DDPM paper
        pass
    
    def sample_timesteps(self, batch_size: int, device: torch.device) -> torch.Tensor:
        # Input: batch size, device
        # Output: random timesteps (batch_size,)
        # Implementation: random integers from 0 to timesteps-1
        pass
    
    def add_noise(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor = None) -> tuple[torch.Tensor, torch.Tensor]:
        # Input: clean images, timesteps, optional noise
        # Output: (noisy_images, noise)
        # Implementation: forward diffusion formula
        pass
```

### 3.3 Visualization (`utils/visualization.py`)

```python
def save_images(images: torch.Tensor, filepath: str, nrow: int = 4, normalize: bool = True) -> None:
    # Input: images tensor, save path, images per row, normalize flag
    # Output: None (saves image file)
    # Implementation: create grid, denormalize if needed, save with torchvision
    pass

def plot_images(images: torch.Tensor, title: str = "Generated Images", 
               nrow: int = 4, figsize: tuple = (10, 10), normalize: bool = True) -> None:
    # Input: images, plot title, grid layout, figure size, normalize flag
    # Output: None (displays plot)
    # Implementation: create grid, plot with matplotlib
    pass

def plot_loss(losses: list, save_path: str = None, title: str = "Training Loss") -> None:
    # Input: list of loss values, optional save path, plot title
    # Output: None (displays/saves plot)
    # Implementation: matplotlib line plot
    pass

def visualize_diffusion_process(model: DiffusionModel, x0: torch.Tensor, 
                               timesteps_to_show: list = [0, 100, 300, 500, 700, 999]) -> torch.Tensor:
    # Input: trained model, clean image, timesteps to visualize
    # Output: concatenated noisy images at different timesteps
    # Implementation: apply forward diffusion at specified timesteps, plot results
    pass
```

---

## 4. Training Script (`main.py`)

```python
def train_model() -> tuple[DiffusionModel, list]:
    # Input: None (uses Config)
    # Output: (trained_model, loss_history)
    # Implementation: 
    # 1. Setup device, create directories
    # 2. Load MNIST data
    # 3. Initialize model and optimizer
    # 4. Training loop with loss computation
    # 5. Periodic checkpointing and sample generation
    # 6. Return trained model and losses
    pass

def main() -> None:
    # Input: None (parses command line arguments)
    # Output: None
    # Implementation: argument parsing, config updates, call train_model()
    pass
```

---

## 5. Generation Script (`generate.py`)

```python
def load_model(checkpoint_path: str, device: torch.device) -> DiffusionModel:
    # Input: path to checkpoint, device
    # Output: loaded model
    # Implementation: load checkpoint, initialize model, load state dict
    pass

def generate_images(model: DiffusionModel, num_samples: int, device: torch.device, 
                   save_path: str = None, show_images: bool = True) -> torch.Tensor:
    # Input: trained model, number of samples, device, optional save path, display flag
    # Output: generated images tensor
    # Implementation: call model.sample(), save/display results
    pass

def generate_interpolation(model: DiffusionModel, num_steps: int, device: torch.device, 
                          save_path: str = None) -> torch.Tensor:
    # Input: trained model, interpolation steps, device, optional save path
    # Output: interpolated images
    # Implementation: interpolate between two noise vectors, generate from each
    pass

def main() -> None:
    # Input: None (parses command line arguments)
    # Output: None
    # Implementation: argument parsing, load model, generate images/interpolations
    pass
```

---

## Implementation Order

1. **Start with `config.py`** - Set up all hyperparameters
2. **Implement `utils/data_loader.py`** - Get data loading working first
3. **Build `models/unet.py`** - Core architecture for noise prediction
4. **Implement `models/diffusion.py`** - Diffusion processes and loss computation
5. **Create `utils/visualization.py`** - For monitoring training progress
6. **Build `main.py`** - Training loop
7. **Implement `generate.py`** - Generation and evaluation
8. **Add `utils/noise_scheduler.py`** - Advanced noise scheduling (optional)

## Key Mathematical Concepts

### Forward Diffusion (Add Noise)
```
q(x_t | x_0) = N(x_t; √(ᾱ_t) x_0, (1 - ᾱ_t)I)
x_t = √(ᾱ_t) x_0 + √(1 - ᾱ_t) ε
```

### Reverse Diffusion (Remove Noise)
```
p_θ(x_{t-1} | x_t) = N(x_{t-1}; μ_θ(x_t, t), σ_t²I)
μ_θ(x_t, t) = 1/√(α_t) * (x_t - β_t/√(1-ᾱ_t) * ε_θ(x_t, t))
```

### Training Loss
```
L = E_{x_0, ε, t}[||ε - ε_θ(√(ᾱ_t) x_0 + √(1-ᾱ_t) ε, t)||²]
```

## Dependencies (`requirements.txt`)
```
torch>=1.9.0
torchvision>=0.10.0
matplotlib>=3.3.0
numpy>=1.21.0
tqdm>=4.62.0
Pillow>=8.3.0
```

This structure provides a complete roadmap for implementing a diffusion model from scratch. Each function has clear inputs, outputs, and implementation guidelines to follow.
