"""
Convolutional Neural Nets
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .quant_modules import QLinear
from .sram_modules import SRAMConv2d

__all__ = ['cnn_mnist']

class SRAMCNN(nn.Module):
    def __init__(self, num_class=10, drop_rate=0.5, wbit=4, abit=4, subArray=64):
        super(SRAMCNN, self).__init__()
        self.conv1 = SRAMConv2d(1, 128, 3, 1, bias=False, wl_input=abit, wl_weight=wbit, subArray=subArray)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = SRAMConv2d(128, 128, 3, 1, bias=False, wl_input=abit, wl_weight=wbit, subArray=subArray)
        self.relu2 = nn.ReLU(inplace=True)

        self.dropout1 = nn.Dropout(drop_rate)
        self.dropout2 = nn.Dropout(drop_rate)
        self.fc1 = QLinear(4608, 128, wbit=wbit, abit=abit)
        self.fc2 = QLinear(128, num_class, wbit=wbit, abit=abit)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        
        x = self.conv2(x)
        x = self.relu2(x)
        
        x = F.avg_pool2d(x, 4)
        
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return x