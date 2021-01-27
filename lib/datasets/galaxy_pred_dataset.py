import os
from glob import glob
import logging
import pandas as pd
import numpy as np

from PIL import Image
import imgaug.augmenters as iaa

import torch
from torch.utils.data.dataset import Dataset
from torchvision.transforms import ToTensor


IMAGE_EXT = '.jpg'
IMAGE_TEST_FOLDER = 'images_test_rev1'
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])

class GalaxyPredictDataset(Dataset):
    
    def __init__(self,
                 root=None,
                 normalize=False,
                 img_size=(424, 424)):
        super(GalaxyPredictDataset, self).__init__()
        self.normalize = normalize
        self.split = 'pred'
        self.img_h, self.img_w = img_size
        self.logger = logging.getLogger(__name__)
        if root is None:
            raise Exception('Please specify the root directory.')
        self.img_paths = glob(root + '/' + IMAGE_TEST_FOLDER + '/*' + IMAGE_EXT)
        self.img_paths = self.img_paths[:160]
        self.to_tensor = ToTensor()
        # self.transform = iaa.Sequential([iaa.Sometimes(then_list=augmentations, p=aug_chance), transformations])
        
    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img = Image.open(img_path).convert('RGB')
        if self.normalize:
            img = (img - IMAGENET_MEAN) / IMAGENET_STD
        # img = self.transform(image=img.copy())
        img = self.to_tensor(img)
        img_id = img_path.split('/')[-1].split('.')[0]
        return img_id, img
    
    def __len__(self):
        return len(self.img_paths)
    
# galaxy = GalaxyPredictDataset(root='datasets')
# print(galaxy.__getitem__(0)[0])
# train_loader = torch.utils.data.DataLoader(dataset=galaxy,
#                                            batch_size=8,
#                                            shuffle=True,
#                                            num_workers=1)
# print(train_loader.dataset.split)       
