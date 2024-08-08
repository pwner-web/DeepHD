import pandas as pd
import numpy as np

import torch
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

#import hiddenlayer as h
import logging

__author__='Gao Shichen'
__mail__= 'gaoshichend@163.com'
__date__= '2024/05/10'

# Configure logging and set the log level to INFO.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define the model class
class NeuralNetwork(nn.Module):
    def __init__(self, input_size):
        super(NeuralNetwork, self).__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.LeakyReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, 64),
            nn.LeakyReLU(),
            nn.BatchNorm1d(64),
            nn.Linear(64, 16),
            nn.LeakyReLU(),
            nn.BatchNorm1d(16),
            nn.Linear(16, 8),
            nn.LeakyReLU(),
            nn.BatchNorm1d(8),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.linear_relu_stack(x)
        return y
