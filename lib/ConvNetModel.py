import pandas as pd
import numpy as np
import torch
from torch import nn

#import hiddenlayer as h

import logging

__author__='Gao Shichen'
__mail__= 'gaoshichend@163.com'
__date__= '2023/05/26'

# Configure logging and set the log level to INFO.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define the model class
class NeuralNetwork(nn.Module):
    def __init__(self, LinearInputLength):
        self.LinearInputLength = LinearInputLength
        super(NeuralNetwork, self).__init__()
        self.conv_stack = nn.Sequential(
            nn.Conv1d(1,6,1, stride=1),
            nn.BatchNorm1d(6),

            nn.Conv1d(6,6,4, stride=4),
            nn.BatchNorm1d(6),
            nn.LeakyReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(6,6,4, stride=4),
            nn.BatchNorm1d(6),
            nn.LeakyReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(6,6,4, stride=4),
            nn.BatchNorm1d(6),
            nn.LeakyReLU(),
            nn.MaxPool1d(2),

            nn.Flatten(start_dim=1),
            #nn.Linear(10*1*6, 16),
            # 此处需根据需要进行调整
            nn.Linear(self.LinearInputLength, 16),
            nn.LeakyReLU(),
            nn.BatchNorm1d(16),

            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.conv_stack(x)
        return y