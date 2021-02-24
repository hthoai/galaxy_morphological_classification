from torch.nn import Linear, Conv2d, MaxPool2d, Flatten, Module
from torch.nn import Sigmoid, ReLU
from lib.models.output_layer import OutputLayer

# OUT_DENSE_SHAPE = 512

class GalaxyModel(Module):
    def __init__(self, out_dense_shape):
        super().__init__()
        # Conv layer
        self.conv1 = Conv2d(in_channels=3, kernel_size=6, out_channels=32)
        self.conv2 = Conv2d(in_channels=32, kernel_size=5, out_channels=64)
        self.conv3 = Conv2d(in_channels=64, kernel_size=3, out_channels=128)
        self.conv4 = Conv2d(in_channels=128, kernel_size=3, out_channels=128)

        # Max pooling layer
        self.maxpool = MaxPool2d(kernel_size=2)

        # MLP layer
        self.dense1 = Linear(in_features=out_dense_shape, out_features=2048)
        self.dense2 = Linear(in_features=2048, out_features=2048)
        self.dense3 = Linear(in_features=2048, out_features=37)
        
        # Flatten layer
        self.flatten = Flatten()

        # Activation function
        self.ReLU = ReLU()
        self.Sigmoid = Sigmoid()

        # Output Layer
        self.output = OutputLayer(in_features=37, out_features=37)

    def forward(self, x):
        x = self.conv1(x)
        x = self.ReLU(x)
        x = self.maxpool(x)

        x = self.conv2(x)
        x = self.ReLU(x)
        x = self.maxpool(x)

        x = self.conv3(x)
        x = self.ReLU(x)
        
        x = self.conv4(x)
        x = self.maxpool(x)
        x = self.ReLU(x)
        
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.ReLU(x)
        x = self.dense2(x)
        x = self.ReLU(x)
        x = self.dense3(x)

        x = self.output(x)
        return x
