import torch
from torch import optim
from torch.nn import MSELoss
from torch.utils.data import TensorDataset, DataLoader 
import numpy as np
from lib.models.galaxy_model import GalaxyModel

lr = 0.5
epochs = 20
bs = 64

def get_model():
    model = GalaxyModel()
    criterion = MSELoss()
    opt = optim.Adam(model.parameters(), lr=lr)
    return model, opt, criterion

def get_data(train_ds, valid_ds, bs):
    return (
        DataLoader(train_ds, batch_size=bs, shuffle=True),
        DataLoader(valid_ds, batch_size=bs * 2),
    )

def fit(epochs, model, criterion, opt, train_dl, valid_dl):
    for epoch in range(epochs):
        for xb, yb in train_dl:
            pred = model(xb)
            loss = criterion(pred, yb)
            if opt is not None:
                loss.backward(retain_graph=True)
                opt.step()
                opt.zero_grad()

        model.eval()
        with torch.no_grad():
            losses, nums = zip(
                *[(criterion(model(xb), yb), len(xb)) for xb, yb in valid_dl]
            )
        val_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)
        print('epoch {}, val_loss {}'.format(epoch, val_loss))

if __name__ == '__main__':
    net, opt, criterion = get_model()
    print(net)
    x_train = torch.randn(1000, 3, 45, 45)
    y_train = torch.randn(1000, 37)
    x_val = torch.randn(100, 3, 45, 45)
    y_val = torch.randn(100, 37)
    train_ds = TensorDataset(x_train, y_train)
    valid_ds = TensorDataset(x_val, y_val)
    train_dl, valid_dl = get_data(train_ds, valid_ds, bs)
    fit(epochs, net, criterion, opt, train_dl, valid_dl)