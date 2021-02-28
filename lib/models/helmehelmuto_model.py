from torch.nn import Linear, Conv2d, MaxPool2d, Flatten, Module, Dropout
from torch.nn import Sigmoid, ReLU
import torch.nn.functional as F
from lib.models.output_layer import OutputLayer


class HelmehelmutoModel(Module):
    def __init__(self, flatten_size: int, dropout_rate: float):
        super().__init__()
        # Conv layer
        self.conv1 = Conv2d(in_channels=3, kernel_size=3, out_channels=512)
        self.conv2 = Conv2d(in_channels=512, kernel_size=2, out_channels=256)
        self.conv3 = Conv2d(in_channels=256, kernel_size=3, out_channels=256)
        self.conv4 = Conv2d(in_channels=256, kernel_size=3, out_channels=128)
        self.conv5 = Conv2d(in_channels=128, kernel_size=3, out_channels=128)

        # Max pooling layer
        self.max_pool = MaxPool2d(kernel_size=2)

        # FC layer
        self.fc1 = Linear(in_features=flatten_size, out_features=128)
        self.fc2 = Linear(in_features=128, out_features=37)
        
        # Flatten layer
        self.flatten = Flatten()

        # Dropout layer
        self.dropout = Dropout(dropout_rate)

        # Activation function
        self.relu = ReLU()
        self.sigmoid = Sigmoid()

        # Output Layer
        # self.output = OutputLayer(in_features=37, out_features=37)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.max_pool(x)
        
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.relu(x)
        x = self.max_pool(x)

        x = self.conv5(x)
        x = self.conv5(x)
        x = self.relu(x)
        x = self.max_pool(x)
        x = F.max_pool2d(x, kernel_size=x.size()[2:])
        x = self.flatten(x)

        x = self.dropout(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.sigmoid(x)

        # x = self.output(x)
        return x
