import torch

import torch.nn as nn
import torch.nn.functional as F


class SimpleCNN(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, features=16):
        super(SimpleCNN, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, features, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(features, features, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(features, features * 2, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(features * 2, features * 4, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(features * 4, features * 2, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(features * 2, features, kernel_size=3, padding=1)
        self.conv7 = nn.Conv2d(features, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x1 = F.relu(self.conv1(x))
        x2 = F.relu(self.conv2(x1))
        x3 = F.relu(self.conv3(x2))
        x4 = F.relu(self.conv4(x3))
        x5 = F.relu(self.conv5(x4))
        x6 = F.relu(self.conv6(x5))
        x7 = self.conv7(x6)
        return x7
    
