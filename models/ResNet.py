"""
ResNet implementation for diffusion model.
Uses the proper ResNet-18 architecture.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    """Basic residual block for ResNet-18/34."""
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                              stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, 
                              stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = F.relu(out)

        return out


class ResNet18(nn.Module):
    """ResNet-18 architecture adapted for diffusion models."""

    def __init__(self, in_channels=1, out_channels=1):
        super(ResNet18, self).__init__()
        
        self.in_channels = 64
        
        # Initial convolution layer (keep spatial dimensions)
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        
        # ResNet layers (modified to maintain spatial dimensions)
        self.layer1 = self._make_layer(BasicBlock, 64, 2, stride=1)   # 28x28 -> 28x28
        self.layer2 = self._make_layer(BasicBlock, 128, 2, stride=1)  # 28x28 -> 28x28
        self.layer3 = self._make_layer(BasicBlock, 256, 2, stride=1)  # 28x28 -> 28x28
        self.layer4 = self._make_layer(BasicBlock, 512, 2, stride=1)  # 28x28 -> 28x28
        
        # Final output layer to reduce channels back to output size
        self.final_conv = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, out_channels, kernel_size=1)
        )
        
        self._initialize_weights()

    def _make_layer(self, block, out_channels, blocks, stride=1):
        """Create a ResNet layer with specified number of blocks."""
        downsample = None
        
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion,
                         kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion),
            )

        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion
        
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def _initialize_weights(self):
        """Initialize weights following ResNet paper."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """Forward pass."""
        # Initial convolution
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        
        # ResNet layers
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # Final output
        x = self.final_conv(x)
        
        return x


class ResNet(nn.Module):
    """
    ResNet wrapper for diffusion model compatibility.
    Uses the standard ResNet-18 architecture.
    """
    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()
        self.resnet18 = ResNet18(in_channels=in_channels, out_channels=out_channels)
    
    def forward(self, x):
        return self.resnet18(x)


# Legacy implementation kept for backward compatibility
class Residual(nn.Module):
    def __init__(self, input_channels, num_channels, use_1x1conv=False, strides=1):
        super(Residual, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, num_channels, kernel_size=3, stride=strides, padding=1)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(input_channels, num_channels, kernel_size=1, stride=strides, padding=0)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, x):
        Y = F.relu(self.bn1(self.conv1(x)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3 is not None:
            x = self.conv3(x)
        Y += x
        return F.relu(Y)


def resnet_block(input_channels, num_channels, num_residuals, first_block=False):
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(Residual(input_channels, num_channels, use_1x1conv=True))
        else:
            blk.append(Residual(num_channels, num_channels))
    return blk


class LegacyResNet(nn.Module):
    """Legacy ResNet implementation - kept for backward compatibility."""
    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            # Optional: MaxPool2d for downsampling
            nn.Sequential(*resnet_block(16, 16, 2, first_block=True)),
            nn.Sequential(*resnet_block(16, 32, 2)),
            nn.Sequential(*resnet_block(32, 64, 2)),
            # Optional: AdaptiveAvgPool2d for output resizing
            nn.Conv2d(64, out_channels, kernel_size=1)
        )

    def forward(self, x):
        return self.net(x)