import torch

import torch.nn as nn
import torch.nn.functional as F


class SimpleCNN(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, features=16):
        super(SimpleCNN, self).__init__()
        
        # Encoder
        self.conv1 = nn.Conv2d(in_channels, features, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(features, features * 2, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(features * 2, features * 4, kernel_size=3, padding=1)
        
        # Decoder
        self.conv4 = nn.Conv2d(features * 4, features * 2, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(features * 2, features, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(features, out_channels, kernel_size=3, padding=1)
        
        # Batch normalization
        self.bn1 = nn.BatchNorm2d(features)
        self.bn2 = nn.BatchNorm2d(features * 2)
        self.bn3 = nn.BatchNorm2d(features * 4)
        self.bn4 = nn.BatchNorm2d(features * 2)
        self.bn5 = nn.BatchNorm2d(features)
        
    def forward(self, x):
        # Encoder
        x1 = F.relu(self.bn1(self.conv1(x)))
        x2 = F.relu(self.bn2(self.conv2(x1)))
        x3 = F.relu(self.bn3(self.conv3(x2)))
        
        # Decoder with skip connections
        x = F.relu(self.bn4(self.conv4(x3)))
        x = x + x2  # Skip connection
        x = F.relu(self.bn5(self.conv5(x)))
        x = x + x1  # Skip connection
        x = self.conv6(x)
        
        return x