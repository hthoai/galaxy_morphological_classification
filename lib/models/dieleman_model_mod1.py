from torch.nn import Linear, Conv2d, MaxPool2d, Module
from torch.nn import ReLU, Dropout
from maxout import MaxOut
import torch

class DielemanModelMod1(Module):
    def __init__(self, img_w=424, img_h=424):
        super(DielemanModelMod1, self).__init__()
        self.conv1 = Conv2d(in_channels=3, kernel_size=6, out_channels=32)
        self.conv2 = Conv2d(in_channels=32, kernel_size=5, out_channels=64)
        self.conv3 = Conv2d(in_channels=64, kernel_size=3, out_channels=128)
        self.conv4 = Conv2d(in_channels=128, kernel_size=3, out_channels=128)

        self.max_pool = MaxPool2d(kernel_size=2)

        self.dense1 = Linear(in_features=128 * 5 * 5, out_features=4096)
        self.dense2 = Linear(in_features=2048, out_features=2048)
        self.dense3 = Linear(in_features=2048, out_features=37)

        self.relu = ReLU()
        self.maxout = MaxOut(step_size=2)
        self.dropout = Dropout(p=0.5)

    def forward(self, input):
        input = self.conv1(input)
        input = self.relu(input)
        input = self.max_pool(input)

        input = self.conv2(input)
        input = self.relu(input)
        input = self.max_pool(input)

        input = self.conv3(input)
        input = self.relu(input)

        input = self.conv4(input)
        input = self.max_pool(input)

        input = input.view(-1,128 * 5 * 5)
        input = self.dense1(input)
        input = self.maxout(input)
        input = self.dense2(input)
        input = self.relu(input)
        input = self.dense3(input)
        input = self.relu(input)

        return input

#input = torch.rand(1,3,69,69) 
#model = DielemanModelMod1()
#print(model(input))


