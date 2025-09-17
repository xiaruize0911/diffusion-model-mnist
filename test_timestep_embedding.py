#!/usr/bin/env python3
"""Quick test script to verify timestep embedding integration."""

import torch
from models import DiffusionModel
from config import Config

def test_model(model_type):
    """Test a specific model type."""
    print(f"\n=== Testing {model_type.upper()} ===")
    
    try:
        # Set config
        Config.MODEL_TYPE = model_type
        
        # Create model
        model = DiffusionModel()
        
        # Test data
        batch_size = 2
        x = torch.randn(batch_size, 1, 28, 28)
        t = torch.randint(0, 300, (batch_size,))
        
        print(f"Input shape: {x.shape}")
        print(f"Timestep shape: {t.shape}")
        
        # Test forward pass (training mode)
        pred_noise, actual_noise = model(x, t)
        print(f"✓ Forward pass: {x.shape} -> {pred_noise.shape}")
        
        # Test sampling (inference mode)
        model.eval()
        with torch.no_grad():
            samples = model.sample((1, 1, 28, 28))
        print(f"✓ Sampling: generated {samples.shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error: {str(e)}")
        return False

def main():
    print("Testing timestep embedding integration across all models...")
    
    # Test all model types
    model_types = ['unet', 'cnn', 'resnet', 'unet2', 'resnet2', 'dit']
    
    success_count = 0
    for model_type in model_types:
        if test_model(model_type):
            success_count += 1
    
    print(f"\n=== SUMMARY ===")
    print(f"Successful: {success_count}/{len(model_types)} models")
    
    if success_count == len(model_types):
        print("🎉 All models successfully updated with timestep embedding!")
    else:
        print("⚠️  Some models have issues that need fixing.")

if __name__ == "__main__":
    main()