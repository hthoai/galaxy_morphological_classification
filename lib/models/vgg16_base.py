from torch.nn import Conv2d, MaxPool2d, Linear, Module
from torch.nn import ReLU, Sigmoid
import torch.nn.functional as F
import numpy as np
import pandas as pd
import torchvision.models as models
import torchsummary as summary
import torch

class SimpleModel(Module):
    def __init__(self):
        super().__init__()
        # use pre-trained base
        self.vgg16 = models.vgg16(pretrained=True)
        # self-constructed head
        self.dense1 = Linear(in_features=1000, out_features=37)
        # activations
        self.Sigmoid = Sigmoid()

    def forward(self, input):
        # assemble model
        input = self.vgg16(input)
        input = self.dense1(input)
        input = self.Sigmoid(input)

x = torch.rand(2,3,424,424)
model = models.vgg16(pretrained=True)
y = model(x)
print(y.shape)