from torch import nn
import torch.nn.functional as F
from torch.tensor import Tensor
from torchsummary import summary
import torch


class BaselineCNN(nn.Module):
    """Baseline CNN model with structure from Kaggle discussion.

    https://www.kaggle.com/helmehelmuto/keras-cnn
    """

    def __init__(self, img_w=424, img_h=424) -> None:
        super(BaselineCNN, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=3, out_channels=512, kernel_size=3, padding=1
        )
        self.conv2 = nn.Conv2d(
            in_channels=512, out_channels=256, kernel_size=3, padding=1
        )
        self.conv3 = nn.Conv2d(
            in_channels=256, out_channels=256, kernel_size=3, padding=1
        )
        self.conv4 = nn.Conv2d(
            in_channels=256, out_channels=128, kernel_size=3, padding=1
        )
        self.conv5 = nn.Conv2d(
            in_channels=128, out_channels=128, kernel_size=3, padding=1
        )
        self.max_pool = nn.MaxPool2d(kernel_size=2)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(p=0.25)
        self.fc1 = nn.Linear(in_features=128 * 53 * 53, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=128)
        self.out = nn.Linear(in_features=128, out_features=37)

    def forward(self, x) -> Tensor:
        """Defines the network structure."""
        x = self.conv1(x)
        x = self.relu(self.conv2(x))
        x = self.max_pool(x)

        x = self.conv3(x)
        x = self.relu(self.conv4(x))
        x = self.max_pool(x)

        x = self.conv5(x)
        x = self.relu(self.conv5(x))
        x = self.max_pool(x)
        x = x.view(-1, 128 * 53 * 53)

        x = self.dropout(x)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.sigmoid(self.out(x))

        return x


# model = BaselineCNN()
# input = torch.rand(1, 3, 424, 424)
# output = model(input)
# print(output.shape)
# summary(model, (3, 424, 424))
