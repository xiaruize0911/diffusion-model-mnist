"""MoDiff (Modulated Diffusion) implementation for accelerated diffusion sampling."""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any
from .diffusion import DiffusionModel
from utils.quantization import modulated_quantize, MoDiffState, uniform_quantize


class MoDiffModel(nn.Module):
    """
    MoDiff wrapper that adds modulated quantization to any diffusion model.
    
    This implementation follows the MoDiff paper which quantizes temporal residuals
    between adjacent timesteps rather than raw activations, achieving better
    compression with minimal quality loss.
    """
    
    def __init__(self, 
                 base_model: DiffusionModel,
                 bit_width: int = 3,
                 enable_quantization: bool = True,
                 enable_error_compensation: bool = True):
        """
        Initialize MoDiff wrapper.
        
        Args:
            base_model (DiffusionModel): Base diffusion model to wrap
            bit_width (int): Quantization bit width (3-8 bits)
            enable_quantization (bool): Whether to enable quantization
            enable_error_compensation (bool): Whether to enable error compensation
        """
        super().__init__()
        self.base_model = base_model
        self.bit_width = bit_width
        self.enable_quantization = enable_quantization
        self.enable_error_compensation = enable_error_compensation
        
        # State management for sampling
        self.modiff_state = MoDiffState()
        
        # Statistics tracking
        self.stats = {
            'total_timesteps': 0,
            'quantization_calls': 0,
            'bops_savings': 0.0
        }
        
    def reset_state(self):
        """Reset MoDiff state for new sampling process."""
        self.modiff_state.reset()
        self.stats['total_timesteps'] = 0
        self.stats['quantization_calls'] = 0
        
    def forward(self, x0: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for training (same as base model).
        
        Args:
            x0 (torch.Tensor): Clean images
            t (torch.Tensor): Random timesteps
            
        Returns:
            tuple[torch.Tensor, torch.Tensor]: (predicted_noise, actual_noise)
        """
        return self.base_model(x0, t)
    
    def compute_loss(self, predicted_noise: torch.Tensor, actual_noise: torch.Tensor) -> torch.Tensor:
        """Compute loss (delegates to base model)."""
        return self.base_model.compute_loss(predicted_noise, actual_noise)
    
    def reverse_diffusion_step(self, xt: torch.Tensor, t: torch.Tensor, predicted_noise: torch.Tensor) -> torch.Tensor:
        """Reverse diffusion step (delegates to base model)."""
        return self.base_model.reverse_diffusion_step(xt, t, predicted_noise)
    
    def _quantized_forward(self, xt: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with MoDiff quantization applied to activations.
        
        Args:
            xt (torch.Tensor): Noisy images at timestep t
            t (torch.Tensor): Current timestep
            
        Returns:
            torch.Tensor: Predicted noise with quantized activations
        """
        if not self.enable_quantization:
            # If quantization disabled, use base model directly
            return self.base_model.net(xt, t)
        
        # Apply MoDiff quantization to model activations
        # Store original activations and apply quantization layer by layer
        quantized_activations = {}
        
        def create_quantization_hook(layer_name):
            def hook_fn(module, input, output):
                if isinstance(output, torch.Tensor) and output.requires_grad == False:
                    # Get previous state for this layer
                    prev_activation, prev_error = self.modiff_state.get_prev_state(layer_name)
                    
                    # Apply modulated quantization
                    if self.enable_error_compensation and prev_activation is not None:
                        quantized_output, error = modulated_quantize(
                            output, prev_activation, prev_error, 
                            self.bit_width, training=False
                        )
                    else:
                        # First timestep or no error compensation
                        quantized_output = uniform_quantize(output, self.bit_width, training=False)
                        error = output - quantized_output
                    
                    # Update state for next timestep
                    self.modiff_state.update(layer_name, output, error)
                    quantized_activations[layer_name] = quantized_output
                    self.stats['quantization_calls'] += 1
                    
                    # Return quantized output
                    return quantized_output
                return output
            return hook_fn
        
        # Register hooks only for specific layers to avoid interference
        hooks = []
        hook_count = 0
        for name, module in self.base_model.net.named_modules():
            # Only hook key computational layers, not all layers
            if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)) and hook_count < 5:
                hook = module.register_forward_hook(create_quantization_hook(f"{name}_{hook_count}"))
                hooks.append(hook)
                hook_count += 1
        
        try:
            # Run forward pass with quantization hooks
            predicted_noise = self.base_model.net(xt, t)
        finally:
            # Always clean up hooks
            for hook in hooks:
                hook.remove()
        
        return predicted_noise
    
    @torch.no_grad()
    def sample(self, shape: tuple, device: torch.device) -> torch.Tensor:
        """
        Generate samples using MoDiff accelerated sampling.
        
        Args:
            shape (tuple): Shape of samples to generate (B, C, H, W)
            device (torch.device): Device to generate samples on
            
        Returns:
            torch.Tensor: Generated samples
        """
        # Reset state for new sampling
        self.reset_state()
        
        # If quantization is disabled, use base model directly
        if not self.enable_quantization:
            return self.base_model.sample(shape, device)
        
        # Start with random noise
        xt = torch.randn(shape, device=device)
        
        # Set model to eval mode for sampling
        self.eval()
        
        # Iterative denoising with MoDiff quantization
        for i, t in enumerate(reversed(range(self.base_model.timesteps))):
            t_tensor = torch.full((shape[0],), t, device=device)
            
            # Get predicted noise with quantized activations
            predicted_noise = self._quantized_forward(xt, t_tensor)
            
            # Perform reverse diffusion step
            xt = self.reverse_diffusion_step(xt, t_tensor, predicted_noise)
            
            self.stats['total_timesteps'] += 1
        
        return xt
    
    @torch.no_grad()
    def sample_with_comparison(self, shape: tuple, device: torch.device) -> Dict[str, torch.Tensor]:
        """
        Generate samples with both MoDiff and baseline for comparison.
        
        Args:
            shape (tuple): Shape of samples to generate
            device (torch.device): Device to generate samples on
            
        Returns:
            Dict[str, torch.Tensor]: Dictionary with 'modiff' and 'baseline' samples
        """
        # Generate with MoDiff
        quantization_enabled = self.enable_quantization
        self.enable_quantization = True
        modiff_samples = self.sample(shape, device)
        
        # Generate with baseline (no quantization)
        self.enable_quantization = False
        baseline_samples = self.sample(shape, device)
        
        # Restore original setting
        self.enable_quantization = quantization_enabled
        
        return {
            'modiff': modiff_samples,
            'baseline': baseline_samples
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get MoDiff performance statistics."""
        bops_reduction = (32 ** 2) / (self.bit_width ** 2) if self.bit_width > 0 else 1.0
        
        return {
            'bit_width': self.bit_width,
            'quantization_enabled': self.enable_quantization,
            'error_compensation_enabled': self.enable_error_compensation,
            'total_timesteps': self.stats['total_timesteps'],
            'quantization_calls': self.stats['quantization_calls'],
            'theoretical_bops_reduction': bops_reduction,
            'memory_savings_ratio': 32 / self.bit_width if self.bit_width > 0 else 1.0
        }
    
    @property
    def timesteps(self):
        """Access timesteps from base model."""
        return self.base_model.timesteps
    
    @property 
    def beta_schedule(self):
        """Access beta schedule from base model."""
        return self.base_model.beta_schedule
    
    def train(self, mode: bool = True):
        """Set training mode."""
        super().train(mode)
        self.base_model.train(mode)
        return self
    
    def eval(self):
        """Set evaluation mode."""
        super().eval()
        self.base_model.eval()
        return self
