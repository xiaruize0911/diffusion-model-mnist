"""Quantization utilities for MoDiff implementation."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any


def uniform_quantize(x: torch.Tensor, bit_width: int, training: bool = False) -> torch.Tensor:
    """
    Uniform min-max quantization for activations.
    
    Args:
        x (torch.Tensor): Input tensor to quantize
        bit_width (int): Number of bits for quantization (e.g., 3, 4, 8)
        training (bool): Whether in training mode (affects gradient handling)
        
    Returns:
        torch.Tensor: Quantized tensor
    """
    if bit_width >= 32:
        return x  # No quantization for high bit widths
    
    # Calculate quantization parameters
    x_min = x.min()
    x_max = x.max()
    
    # Handle edge case where all values are the same
    if x_max == x_min:
        return x
    
    # Number of quantization levels
    n_levels = 2 ** bit_width
    scale = (x_max - x_min) / (n_levels - 1)
    
    # Quantize
    x_normalized = (x - x_min) / scale
    x_quantized = torch.round(x_normalized).clamp(0, n_levels - 1)
    x_dequantized = x_quantized * scale + x_min
    
    # Straight-through estimator for gradient computation
    if training:
        return x + (x_dequantized - x).detach()
    else:
        return x_dequantized


def modulated_quantize(activation: torch.Tensor, 
                      prev_activation: Optional[torch.Tensor],
                      prev_error: Optional[torch.Tensor],
                      bit_width: int = 3,
                      training: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Modulated quantization with error compensation as described in MoDiff paper.
    
    Args:
        activation (torch.Tensor): Current activation tensor
        prev_activation (Optional[torch.Tensor]): Previous timestep activation (for residual)
        prev_error (Optional[torch.Tensor]): Previous quantization error (for compensation)
        bit_width (int): Number of bits for quantization
        training (bool): Whether in training mode
        
    Returns:
        tuple[torch.Tensor, torch.Tensor]: (quantized_activation, quantization_error)
    """
    if prev_activation is None:
        # First timestep - quantize directly
        quantized = uniform_quantize(activation, bit_width, training)
        error = activation - quantized
        return quantized, error
    
    # Calculate temporal difference (residual)
    residual = activation - prev_activation
    
    # Add error compensation from previous step
    if prev_error is not None:
        residual = residual + prev_error
    
    # Quantize the residual
    quantized_residual = uniform_quantize(residual, bit_width, training)
    
    # Reconstruct quantized activation
    quantized_activation = quantized_residual + prev_activation
    
    # Calculate new quantization error
    error = activation - quantized_activation
    
    return quantized_activation, error


class QuantizedLinear(nn.Module):
    """Linear layer with activation quantization support."""
    
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias)
        
    def forward(self, x: torch.Tensor, quantize: bool = False, bit_width: int = 8) -> torch.Tensor:
        """
        Forward pass with optional activation quantization.
        
        Args:
            x (torch.Tensor): Input tensor
            quantize (bool): Whether to quantize activations
            bit_width (int): Bit width for quantization
            
        Returns:
            torch.Tensor: Output tensor
        """
        if quantize:
            x = uniform_quantize(x, bit_width, self.training)
        return self.linear(x)


class QuantizedConv2d(nn.Module):
    """Conv2d layer with activation quantization support."""
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, 
                 stride: int = 1, padding: int = 0, bias: bool = True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        
    def forward(self, x: torch.Tensor, quantize: bool = False, bit_width: int = 8) -> torch.Tensor:
        """
        Forward pass with optional activation quantization.
        
        Args:
            x (torch.Tensor): Input tensor
            quantize (bool): Whether to quantize activations
            bit_width (int): Bit width for quantization
            
        Returns:
            torch.Tensor: Output tensor
        """
        if quantize:
            x = uniform_quantize(x, bit_width, self.training)
        return self.conv(x)


class MoDiffState:
    """State management for MoDiff sampling process."""
    
    def __init__(self):
        self.prev_activations: Dict[str, torch.Tensor] = {}
        self.prev_errors: Dict[str, torch.Tensor] = {}
        self.step_count = 0
        
    def reset(self):
        """Reset state for new sampling process."""
        self.prev_activations.clear()
        self.prev_errors.clear()
        self.step_count = 0
        
    def update(self, layer_name: str, activation: torch.Tensor, error: torch.Tensor):
        """Update state with new activation and error."""
        self.prev_activations[layer_name] = activation.clone()
        self.prev_errors[layer_name] = error.clone()
        
    def get_prev_state(self, layer_name: str) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Get previous activation and error for a layer."""
        prev_activation = self.prev_activations.get(layer_name)
        prev_error = self.prev_errors.get(layer_name)
        return prev_activation, prev_error


def compute_bops_reduction(original_bits: int = 32, quantized_bits: int = 3) -> float:
    """
    Compute BOPs (Bit Operations) reduction factor.
    
    Args:
        original_bits (int): Original precision (e.g., 32 for float32)
        quantized_bits (int): Quantized precision
        
    Returns:
        float: Reduction factor
    """
    return (original_bits ** 2) / (quantized_bits ** 2)


def analyze_activation_distribution(activations: torch.Tensor, residuals: torch.Tensor) -> Dict[str, float]:
    """
    Analyze activation and residual distributions for quantization analysis.
    
    Args:
        activations (torch.Tensor): Original activations
        residuals (torch.Tensor): Temporal residuals
        
    Returns:
        Dict[str, float]: Statistics about distributions
    """
    stats = {
        'activation_mean': activations.mean().item(),
        'activation_std': activations.std().item(),
        'activation_range': (activations.max() - activations.min()).item(),
        'residual_mean': residuals.mean().item(),
        'residual_std': residuals.std().item(),
        'residual_range': (residuals.max() - residuals.min()).item(),
        'range_reduction_ratio': ((residuals.max() - residuals.min()) / (activations.max() - activations.min())).item()
    }
    return stats
