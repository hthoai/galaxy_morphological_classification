from torch.nn import Linear, Conv2d, MaxPool2d, Module
from torch.nn import ReLU, LeakyReLU, Dropout
import torch

"""
Ensemble model 1 with the following mods: 
1. ReLU -> Leaky ReLU
2. 4 conv nets -> 3 conv nets
3. 3 dense layers -> 2 dense layers
"""

class EnsemModel1(Module):
    def __init__(self):
        super(EnsemModel1,self).__init__()
        self.conv1 = Conv2d(in_channels=3, kernel_size=7, out_channels=32)
        self.conv2 = Conv2d(in_channels=32, kernel_size=5, out_channels=64)
        self.conv3 = Conv2d(in_channels=64, kernel_size=3, out_channels=128)

        self.max_pool = MaxPool2d(kernel_size=2)

        self.dense1 = Linear(in_features=128 * 5 * 5, out_features=2048)
        self.dense2 = Linear(in_features=2048, out_features=37)

        self.leaky = LeakyReLU()
        self.dropout = Dropout(p=0.5, inplace=True)

    def forward(self, input):
        input = self.conv1(input)
        input = self.leaky(input)
        input = self.max_pool(input)
        input = self.dropout(input)
        
        input = self.conv2(input)
        input = self.leaky(input)
        input = self.max_pool(input)
        input = self.dropout(input)

        input = self.conv3(input)
        input = self.leaky(input)
        input = self.max_pool(input)
        input = self.dropout(input)

        input = self.dense1(input)
        input = self.leaky(input)

        input = self.dense2(input)
        input = self.leaky(input)

        return input
        
