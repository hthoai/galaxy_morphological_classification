import os
import pickle
import random
import logging
from tqdm import tqdm, trange

import cv2
import numpy as np
import pandas as pd

import torch
import torch.nn.functional as F


class Runner:
    def __init__(self, cfg, exp, device, resume=False) -> None:
        self.cfg = cfg
        self.exp = exp
        self.device = device
        self.resume = resume
        self.logger = logging.getLogger(__name__)
        torch.manual_seed(cfg["seed"])
        np.random.seed(cfg["seed"])
        random.seed(cfg["seed"])

    def train(self) -> None:
        """Training loop."""
        starting_epoch = 1
        model = self.cfg.get_model()
        model = model.to(self.device)
        optimizer = self.cfg.get_optimizer(model.parameters())
        scheduler = self.cfg.get_lr_scheduler(optimizer)
        if self.resume:
            last_epoch, model, optimizer, scheduler = self.exp.load_last_train_state(
                model, optimizer, scheduler
            )
            starting_epoch = last_epoch + 1
        max_epochs = self.cfg["epochs"]
        train_loader = self.get_train_dataloader()
        loss_parameters = self.cfg.get_loss_parameters()

        for epoch in trange(
            starting_epoch, max_epochs + 1, initial=starting_epoch - 1, total=max_epochs
        ):
            self.exp.epoch_start_callback(epoch, max_epochs)
            model.train()
            pbar = tqdm(train_loader)
            for idx, (images, targets) in enumerate(pbar):
                images = images.to(self.device)
                targets = targets.to(self.device)

                # Forward pass
                outputs = model(images)
                loss = F.binary_cross_entropy(outputs, targets)
                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # Scheduler step (iteration based)
                scheduler.step()

                # Log
                # postfix_dict = {key: float(value) for key, value in loss.item()}
                postfix_dict = {}
                postfix_dict["lr"] = optimizer.param_groups[0]["lr"]
                self.exp.iter_end_callback(
                    epoch, max_epochs, idx, len(train_loader), loss.item(), postfix_dict
                )
                postfix_dict["loss"] = loss.item()
                pbar.set_postfix(ordered_dict=postfix_dict)
            self.exp.epoch_end_callback(
                epoch + 1, max_epochs, model, optimizer, scheduler
            )

            # Validate
            if (epoch + 1) % self.cfg["val_every"] == 0:
                self.eval(epoch + 1, checkpoint=model)

    def eval(self, epoch, checkpoint=None):
        if checkpoint is None:
            model = self.cfg.get_model()
            model_path = self.exp.get_checkpoint_path(epoch)
            self.logger.info("Loading model %s", model_path)
            model.load_state_dict(self.exp.get_epoch_model(epoch))
            dataloader = self.get_test_dataloader()
            on_val = False
        else:
            model = checkpoint
            dataloader = self.get_val_dataloader()
            on_val = True
        model = model.to(self.device)
        model.eval()
        self.exp.eval_start_callback(self.cfg)
        with torch.no_grad():
            eval_loss = 0
            eval_rmse = 0
            nb_eval_steps = len(dataloader.dataset) // self.cfg["batch_size"]
            for idx, (images, targets) in enumerate(tqdm(dataloader)):
                images = images.to(self.device)
                targets = targets.to(self.device)
                prediction = model(images)
                batch_eval_loss = F.binary_cross_entropy(prediction, targets)
                eval_loss += batch_eval_loss.mean().item()
                batch_rmse = torch.sqrt(F.mse_loss(prediction, targets))
                eval_rmse += batch_rmse.mean().item()
            eval_loss = eval_loss / nb_eval_steps
            eval_rmse = eval_rmse / nb_eval_steps
            results = {"eval_loss": eval_loss, "eval_rmse": eval_rmse}
        self.exp.eval_end_callback(dataloader.dataset, epoch + 1, results, on_val)

    def predict(self, model_name):
        model = self.cfg.get_model()
        self.logger.info("Loading model %s", model_name)
        model.load_state_dict(
            torch.load(os.path.join(self.exp.models_dirpath, model_name))["model"]
        )
        model = model.to(self.device)
        model.eval()
        dataloader = self.get_pred_dataloader()
        self.exp.pred_start_callback(self.cfg)
        df = pd.DataFrame(
            columns=[
                "GalaxyID",
                "Class1.1",
                "Class1.2",
                "Class1.3",
                "Class2.1",
                "Class2.2",
                "Class3.1",
                "Class3.2",
                "Class4.1",
                "Class4.2",
                "Class5.1",
                "Class5.2",
                "Class5.3",
                "Class5.4",
                "Class6.1",
                "Class6.2",
                "Class7.1",
                "Class7.2",
                "Class7.3",
                "Class8.1",
                "Class8.2",
                "Class8.3",
                "Class8.4",
                "Class8.5",
                "Class8.6",
                "Class8.7",
                "Class9.1",
                "Class9.2",
                "Class9.3",
                "Class10.1",
                "Class10.2",
                "Class10.3",
                "Class11.1",
                "Class11.2",
                "Class11.3",
                "Class11.4",
                "Class11.5",
                "Class11.6",
            ]
        )
        # Predict and write to csv file
        with torch.no_grad():
            for idx, (img_ids, images) in enumerate(tqdm(dataloader)):
                images = images.to(self.device)
                predictions = model(images)
                for preddiction in predictions:
                    df = df.append(
                        dict(zip(df.columns, torch.flatten(preddiction).tolist())),
                        ignore_index=True,
                    )
            self.logger.info(f"Writing result to {self.exp.results_path}")
            df.to_csv(self.exp.results_path, index=False)
        self.exp.pred_end_callback()

    def get_train_dataloader(self):
        train_dataset = self.cfg.get_dataset("train")
        train_loader = torch.utils.data.DataLoader(
            dataset=train_dataset,
            batch_size=self.cfg["batch_size"],
            shuffle=True,
            num_workers=1,
            worker_init_fn=self._worker_init_fn_,
        )
        return train_loader

    def get_test_dataloader(self):
        test_dataset = self.cfg.get_dataset("test")
        test_loader = torch.utils.data.DataLoader(
            dataset=test_dataset,
            batch_size=self.cfg["batch_size"],
            shuffle=False,
            num_workers=1,
            worker_init_fn=self._worker_init_fn_,
        )
        return test_loader

    def get_val_dataloader(self):
        val_dataset = self.cfg.get_dataset("val")
        val_loader = torch.utils.data.DataLoader(
            dataset=val_dataset,
            batch_size=self.cfg["batch_size"],
            shuffle=False,
            num_workers=1,
            worker_init_fn=self._worker_init_fn_,
        )
        return val_loader

    def get_pred_dataloader(self):
        pred_dataset = self.cfg.get_dataset("pred")
        pred_loader = torch.utils.data.DataLoader(
            dataset=pred_dataset,
            batch_size=self.cfg["batch_size"],
            shuffle=False,
            num_workers=1,
            worker_init_fn=self._worker_init_fn_,
        )
        return pred_loader

    @staticmethod
    def _worker_init_fn_(_):
        torch_seed = torch.initial_seed()
        np_seed = torch_seed // 2 ** 32 - 1
        random.seed(torch_seed)
        np.random.seed(np_seed)
