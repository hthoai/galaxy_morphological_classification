import torch
from torch import nn
import torch.nn.functional as F

# https://www.kaggle.com/helmehelmuto/keras-cnn
class BaselineCNN(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=512, kernel_size=3)
        self.conv2 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3)
        self.conv3 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3)
        self.conv4 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3)
        self.conv5 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3)
        self.max_pool = nn.MaxPool2d(kernel_size=2)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(p=0.25)
        self.fc = nn.Linear(in_features=1024, out_features=128)
        self.out = nn.Linear(37)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(self.conv2(x))
        x = self.max_pool(x)

        x = self.conv3(x)
        x = self.relu(self.conv4(x))
        x = self.max_pool(x)

        x = self.conv4(x)
        x = self.relu(self.conv4(x))
        x = self.max_pool(x)

        x = self.dropout(x)
        x = self.relu(self.fc(x))
        x = self.dropout(x)
        x = self.relu(self.fc(x))
        x = self.dropout(x)
        x = self.relu(self.fc(x))
        x = self.dropout(x)
        x = self.sigmoid(self.out(x))

        return x

    def training_step(self, batch, batch_idx):
        # training_step defined the train loop
        # it is independent of forward
        x, y = batch
        logits = self.forward(x)
        loss = F.binary_cross_entropy(logits, y)
        # Logging to TensorBoard by default
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = F.binary_cross_entropy(logits, y)
        metric = F.mse_loss()
        rmse = torch.sqrt(metric(logits, y))
        # Logging to TensorBoard by default
        self.log('val_loss', loss)
        self.log('val_rmse', rmse)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=0.01, momentum=0.9)
        return optimizer

    