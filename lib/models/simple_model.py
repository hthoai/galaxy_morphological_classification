from torch.nn import Conv2d, MaxPool2d, Linear, ReLU, Module
import torch.nn.functional as F
import numpy as np
import pandas as pd


class SimpleModel(Module):
    def __init__(self):
        self.conv1 = Conv2d(in_channels=3, out_channels=32, kernel_size=3)
        self.maxpool1 = MaxPool2d(kernel_size=3)
        self.dense1 = Linear(in_features=3, out_features=37)
        self.ReLu = ReLU()

    def forward(self, input):
        input = self.conv1(input)
        input = self.maxpool1(input)
        input = self.dense1(input)
        input = self.ReLu(input)
