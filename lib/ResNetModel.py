import pandas as pd
import numpy as np
import torch
from torch import nn

#import hiddenlayer as h

import logging

__author__='Gao Shichen'
__mail__= 'gaoshichend@163.com'
__date__= '2024/04/26'
__update__= '2024/05/10'

# Configure logging and set the log level to INFO.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define the model class
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu = nn.LeakyReLU()

        if in_channels != out_channels or stride != 1:
            self.downsample = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm1d(out_channels)
            )
        else:
            self.downsample = nn.Identity()

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.downsample(residual)
        out = self.relu(out)
        return out


class NeuralNetwork(nn.Module):
    def __init__(self, LinearInputLength):
        super(NeuralNetwork, self).__init__()
        self.LinearInputLength = LinearInputLength

        self.conv_stack = nn.Sequential(
            nn.Conv1d(1, 6, kernel_size=1, stride=1),
            nn.BatchNorm1d(6),
            
            ResidualBlock(6, 6, stride=4),
            nn.MaxPool1d(2),
            
            ResidualBlock(6, 6),
            nn.MaxPool1d(2),
            
            ResidualBlock(6, 6),
            nn.MaxPool1d(2),

            nn.Flatten(start_dim=1),
            nn.Linear(self.LinearInputLength, 16),
            nn.LeakyReLU(),
            nn.BatchNorm1d(16),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.conv_stack(x)
        return y
