import torch.nn as nn
import torch.nn.functional as F
from .layers import ConvBlock, LinearBlock

class CNN1Model(nn.Module):
    """
    A deep learning model for image classification.
    """
    def __init__(self, num_class:int=10):
        super().__init__()
        self.conv1 = ConvBlock(in_channels=3, out_channels=16)
        self.conv2 = ConvBlock(in_channels=16, out_channels=32)
        self.conv3 = ConvBlock(in_channels=32, out_channels=64)
        self.fc1 = LinearBlock(in_features=64 * 4 * 4, out_features=512)
        self.fc2 = nn.Linear(512, num_class)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(-1, 64 * 4 * 4)
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x
    