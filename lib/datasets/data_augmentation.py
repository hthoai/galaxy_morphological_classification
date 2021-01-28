import numpy as np
import torch
import os
from torchvision.transforms import Compose, RandomPerspective, RandomAffine
from torchvision.transforms import RandomHorizontalFlip, RandomVerticalFlip

def data_augmentation(image_tensor):
    transform = Compose([RandomAffine(degrees=(-180,180)),
                        RandomPerspective(distortion_scale=0.5,p=0.8),
                        RandomVerticalFlip(p=0.5), 
                        RandomHorizontalFlip(p=0.5)])
    return transform(image_tensor)

