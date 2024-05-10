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
__date__= '2023/05/26'

# Configure logging and set the log level to INFO.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define the model class
class NeuralNetwork(nn.Module):
    def __init__(self, input_size):
        super(NeuralNetwork, self).__init__()
        self.linear_relu_stack = nn.Sequential(
            # nn.Linear：全链接层
            nn.Linear(input_size, 512),
            # LeakyReLU activate function 相比于ReLU，保留了一些负轴的值
            nn.LeakyReLU(),
            # Dropout function
            nn.Dropout(0.5),
            # normalization function
            nn.BatchNorm1d(512),
            
            nn.Linear(512, 256),
            nn.LeakyReLU(),
            nn.Dropout(0.5),
            nn.BatchNorm1d(256),
            
            nn.Linear(256, 128),
            nn.LeakyReLU(),
            nn.Dropout(0.3),
            nn.BatchNorm1d(128),
            
            nn.Linear(128, 64),
            nn.LeakyReLU(),
            nn.BatchNorm1d(64),
            nn.Linear(64, 1),
            # sigmod activation function
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.linear_relu_stack(x)
        return y