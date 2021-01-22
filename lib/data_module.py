import os
import logging

import pandas as pd
import numpy as np
from PIL import Image

import imgaug.augmenters as iaa
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from torchvision import transforms


SPLIT_FILES = {
    'train': 'train.csv',
    'val': 'val.csv'
}

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])

class GalaxyDataset(object):
    
    def __init__(self,
                 split='train',
                 root=None,
                 augmentations=None,
                 transforms=None,
                 aug_chance=1.0,
                 img_size=(424, 424)):
        super(GalaxyDataset, self).__init__()
        self.transforms = transforms
        self.img_h, self.img_w = img_size
        self.logger = logging.getLogger(__name__)
        if root is None:
            raise Exception('Please specify the root directory.')
        if split not in SPLIT_FILES:
            raise Exception(f'Split `{split}` does not exist.')
        df = pd.read_csv(root + SPLIT_FILES[split])
        self.img_paths = df['GalaxyID'].apply(lambda idx: os.path.join(root, str(idx)))
        self.targets = df[df.columns.difference(['GalaxyID'])]
        if augmentations is not None:
            # add augmentations
            augmentations = [getattr(iaa, aug['name'])(**aug['parameters'])
                             for aug in augmentations]  # add augmentation
        else:
            augmentations = []
        transformations = iaa.Sequential([
            iaa.Resize({'height': self.img_h, 'width': self.img_w})])
        self.transform = iaa.Sequential([
            iaa.Sometimes(then_list=augmentations, p=aug_chance),
            transformations])
        
    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img = Image.open(img_path).convert('RGB')
        if self.normalize:
            img = (img - IMAGENET_MEAN) / IMAGENET_STD
        img = self.transform(image=img.copy())
        img = self.to_tensor(img.astype(np.float32))
        target = torch.tensor(self.targets[idx])
        return img, target
    
    def __len__(self):
        return len(self.img_paths)
    
class GalaxyDataModule(pl.LightningDataModule):
    def setup(self, stage):
        # transforms for images
        transform = transforms.Compose([transforms.ToTensor, 
                                      transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
        self.train = GalaxyDataset(split=SPLIT_FILES['train'], root='data/', transforms=transform)
        self.val = GalaxyDataset(split=SPLIT_FILES['val'], root='data/', transforms=transform)
        self.test = GalaxyDataset(split=SPLIT_FILES['test'], root='data/', transforms=transform)
        
    def train_dataloader(self):
        return DataLoader(self.train, batch_size=8)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=8)
    
    def test_dataloader(self):
        return DataLoader(self.test, batch_size=8)
