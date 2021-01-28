import torch
from torch.nn import Linear, Conv2d, MaxPool2d, Module
from torch.nn import Sigmoid, ReLU

class DielemanModel(Module):
    def __init__(self):
        super().__init__()
        self.dense1 = Linear(in_features=1024, out_features=2048)
        self.dense2 = Linear(in_features=2048, out_features=2048)
        self.dense3 = Linear(in_features=2048, out_features=37)
        self.conv1 = Conv2d(in_channels=3, kernel_size=6, out_channels=32)
        self.conv2 = Conv2d(in_channels=32, kernel_size=5, out_channels=64)
        self.conv3 = Conv2d(in_channels=64, kernel_size=3, out_channels=128)
        self.conv4 = Conv2d(in_channels=128, kernel_size=3, out_channels=128)
        self.maxpool1 = MaxPool2d(kernel_size=2)
        self.maxpool2 = MaxPool2d(kernel_size=2)
        self.maxpool3 = MaxPool2d(kernel_size=2)
        self.ReLU = ReLU()
        self.Sigmoid = Sigmoid()

    def forward(self, input):
        input = self.conv1(input)
        input = self.ReLU(input)
        input = self.maxpool1(input)

        input = self.conv2(input)
        input = self.ReLU(input)
        input = self.maxpool2(input)

        input = self.conv3(input)
        input = self.ReLU(input)
        
        input = self.conv4(input)
        input = self.maxpool3(input)

        input = input.view(-1,1024)
        input = self.dense1(input)
        input = self.dense2(input)
        input = self.ReLU(input)
        input = self.dense3(input)
        input = self.ReLU(input)
        return input

