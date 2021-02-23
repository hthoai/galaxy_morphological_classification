# import torch
# from torch import optim
# from torch.nn import MSELoss
# from torch.utils.data import TensorDataset, DataLoader 
# import numpy as np
# from lib.models.galaxy_model import GalaxyModel

# lr = 0.01
# epochs = 20
# bs = 64

# def get_model():
#     model = GalaxyModel()
#     criterion = MSELoss()
#     opt = optim.Adam(model.parameters(), lr=lr)
#     return model, opt, criterion

# def get_data(train_ds, valid_ds, bs):
#     return (
#         DataLoader(train_ds, batch_size=bs, shuffle=True),
#         DataLoader(valid_ds, batch_size=bs * 2),
#     )

# def fit(epochs, model, criterion, opt, train_dl, valid_dl):
#     for epoch in range(epochs):
#         for xb, yb in train_dl:
#             pred = model(xb)
#             loss = criterion(pred, yb)
#             if opt is not None:
#                 loss.backward(retain_graph=True)
#                 opt.step()
#                 opt.zero_grad()

#         model.eval()
#         with torch.no_grad():
#             losses, nums = zip(
#                 *[(criterion(model(xb), yb), len(xb)) for xb, yb in valid_dl]
#             )
#         val_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)
#         print('epoch {}, val_loss {}'.format(epoch, val_loss))

# if __name__ == '__main__':
#     net, opt, criterion = get_model()
#     print(net)
#     x_train = torch.randn(1000, 3, 45, 45)
#     y_train = torch.randn(1000, 37)
#     x_val = torch.randn(100, 3, 45, 45)
#     y_val = torch.randn(100, 37)
#     train_ds = TensorDataset(x_train, y_train)
#     valid_ds = TensorDataset(x_val, y_val)
#     train_dl, valid_dl = get_data(train_ds, valid_ds, bs)
#     fit(epochs, net, criterion, opt, train_dl, valid_dl)

import logging
import argparse

import torch

from lib.config import Config
from lib.runner import Runner
from lib.experiment import Experiment


def parse_args():
    parser = argparse.ArgumentParser(description="Galaxy Morphological Classification")
    parser.add_argument(
        "mode", choices=["train", "test", "pred"], help="Train, eval or predict?"
    )
    parser.add_argument("--exp_name", help="Experiment name", required=True)
    parser.add_argument("--cfg", help="Config file")
    parser.add_argument("--model_nb", help="Model number (epoch number)")
    parser.add_argument("--resume", action="store_true", help="Resume training")
    parser.add_argument("--epoch", type=int, help="Epoch to test the model on")
    parser.add_argument(
        "--cpu", action="store_true", help="(Unsupported) Use CPU instead of GPU"
    )
    args = parser.parse_args()
    if args.cfg is None and args.mode == "train":
        raise Exception(
            "If you are training, you have to set a config file using --cfg /path/to/your/config.yaml"
        )
    if args.resume and args.mode == "test":
        raise Exception("args.resume is set on `test` mode: can't resume testing")
    if args.epoch is not None and args.mode == "train":
        raise Exception("The `epoch` parameter should not be set when training")
    if args.cpu:
        raise Exception(
            "CPU training/testing is not supported: the NMS procedure is only implemented for CUDA"
        )

    return args


def main():
    args = parse_args()
    exp = Experiment(args.exp_name, args, mode=args.mode)
    if args.cfg is None:
        cfg_path = exp.cfg_path
    else:
        cfg_path = args.cfg
    cfg = Config(cfg_path)
    exp.set_cfg(cfg, override=False)
    device = (
        torch.device("cpu")
        if not torch.cuda.is_available() or args.cpu
        else torch.device("cuda")
    )
    runner = Runner(cfg, exp, device, resume=args.resume)
    if args.mode == "train":
        try:
            runner.train()
        except KeyboardInterrupt:
            logging.info("Training interrupted.")
    elif args.mode == "test":
        runner.eval(epoch=args.epoch or exp.get_last_checkpoint_epoch())
    else:
        runner.predict(args.model_nb)


if __name__ == "__main__":
    main()
