from torch.nn import Conv2d, MaxPool2d, Linear, Module
from torch.nn import ReLU, Softmax
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
        self.alexnet = models.alexnet(pretrained=True)
        self.resnet18 = models.resnet18(pretrained=True)
        # self-constructed head
        self.dense1 = Linear(in_features=3, out_features=37)
        # activations
        self.Softmax = Softmax()
        self.ReLu = ReLU()

    def forward(self, input):
        # assemble model
        input = self.vgg16(input)
        input = self.dense1(input)
        input = self.Softmax(input)
