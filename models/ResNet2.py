import torch

import torch.nn as nn
import torch.nn.functional as F

class Residual(nn.Module):
    def __init__(self,input_channels, num_channels, use_1x1conv = True, strides = 1):
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

def resnet_block(input_channels, num_channels,num_residuals, first_block = False):
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(Residual(input_channels, num_channels, use_1x1conv=True))
        else:
            blk.append(Residual(num_channels, num_channels))
    return blk

class ResNet2(nn.Module):
    def __init__(self,in_channels=1, out_channels=1):
        super().__init__()
        self.net = nn.Sequential(
                                nn.Conv2d(in_channels,out_channels=4,kernel_size=3,padding=1),
                                nn.BatchNorm2d(4),
                                nn.ReLU(),
                                nn.Sequential(*resnet_block(4, 4, 2, first_block=True)),
                                nn.Sequential(*resnet_block(4, 8, 3)),
                                nn.Sequential(*resnet_block(8, 16, 3)),
                                nn.Sequential(*resnet_block(16, 32, 3)),
                                nn.Sequential(*resnet_block(32, 64, 3)),
                                nn.Sequential(*resnet_block(64, 128, 3)),
                                nn.Conv2d(128, out_channels, kernel_size=1)
                                )
    def forward(self, x):
        return self.net(x)

