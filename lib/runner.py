import pickle
import random
import logging
from tqdm import tqdm, trange

import cv2
import numpy as np

import torch


class Runner:

    def __init__(self, cfg, exp, device, resume=False) -> None:
        self.cfg = cfg
        self.exp = exp
        self.device = device
        self.resume = resume
        torch.manual_seed(cfg['seed'])
        np.random.seed(cfg['seed'])
        random.seed(cfg['seed'])

    def train(self) -> None:
        """Training loop.
        """
        starting_epoch = 1
        model = self.cfg.get_model()
        model = model.to(self.device)
        optimizer = self.cfg.get_optimizer(model.parameters())
        scheduler = self.cfg.get_lr_scheduler(optimizer)
        if self.resume:
            last_epoch, model, optimizer, scheduler = self.exp.load_last_train_state(model, optimizer, scheduler)
            starting_epoch = last_epoch + 1
        max_epochs = self.cfg['epochs']
        train_loader = self.get_train_dataloader()
        loss_parameters = self.cfg.get_loss_parameters()

        for epoch in trange(starting_epoch, max_epochs + 1, initial=starting_epoch - 1, total=max_epochs):
            model.train()
            pbar = tqdm(train_loader)
            for idx, (images, targets) in enumerate(pbar):
                images = images.to(self.device)
                targets = targets.to(self.device)

                # Forward pass
                outputs = model(images, **self.cfg.get_train_parameters())
                loss = model.loss(outputs, targets, **loss_parameters)

                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Scheduler step (iteration based)
                scheduler.step()

                # Log
                postfix_dict = {key: float(value) for key, value in loss.items()}
                postfix_dict['lr'] = optimizer.param_groups[0]["lr"]
                self.exp.iter_end_callback(epoch, max_epochs, idx, len(train_loader), loss.item(), postfix_dict)
                postfix_dict['loss'] = loss.item()
                pbar.set_postfix(ordered_dict=postfix_dict)

            # Validate
            if (epoch + 1) % self.cfg['val_every'] == 0:
                self.eval(epoch, on_val=True)

    def eval(self, epoch, on_val=False):
        model = self.cfg.get_model()
        model_path = self.exp.get_checkpoint_path(epoch)
        self.logger.info('Loading model %s', model_path)
        model.load_state_dict(self.exp.get_epoch_model(epoch))
        model = model.to(self.device)
        model.eval()
        if on_val:
            dataloader = self.get_val_dataloader()
        else:
            dataloader = self.get_test_dataloader()
        test_parameters = self.cfg.get_test_parameters()
        predictions = []
        self.exp.eval_start_callback(self.cfg)
        with torch.no_grad():
            for idx, (images, _) in enumerate(tqdm(dataloader)):
                images = images.to(self.device)
                output = model(images, **test_parameters)
                predictions.extend(output)
        self.exp.eval_end_callback(dataloader.dataset.dataset, predictions, epoch)

    def get_train_dataloader(self):
        train_dataset = self.cfg.get_dataset('train')
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                   batch_size=self.cfg['batch_size'],
                                                   shuffle=True,
                                                   num_workers=8,
                                                   worker_init_fn=self._worker_init_fn_)
        return train_loader

    def get_test_dataloader(self):
        test_dataset = self.cfg.get_dataset('test')
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                  batch_size=self.cfg['batch_size'] if not self.view else 1,
                                                  shuffle=False,
                                                  num_workers=8,
                                                  worker_init_fn=self._worker_init_fn_)
        return test_loader

    def get_val_dataloader(self):
        val_dataset = self.cfg.get_dataset('val')
        val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                                 batch_size=self.cfg['batch_size'],
                                                 shuffle=False,
                                                 num_workers=8,
                                                 worker_init_fn=self._worker_init_fn_)
        return val_loader
