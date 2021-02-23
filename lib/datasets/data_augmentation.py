import numpy as np
import torch
import imgaug as ia
import cv2
from imgaug import augmenters as iaa
from imgaug import parameters as iap

class ImageAug:
    def __init__(self):
        self.aug = iaa.Sequential([
            iaa.Affine(
                rotate=iap.Normal(0.0,30), 
                translate_px={"x": (-4, 4), "y": (-4, 4)}
            )
            iaa.Fliplr(0.5), 
            iaa.imgcorruptlike.Pixelate(severity=2)
        ])
    def __call__(self, img):
        img = np.array(img)
    return self.aug.augment_image(img)

class GetPerspective:
    def __init__(self):
        self.trans_matx1 = np.float32([[0,0],[212,0],[0,212],[212,212]])
        self.trans_matx2 = np.float32([[0,0],[45,0],[0,45],[45,45]])
    def __call__(self, img):
        M = cv2.getPerspectiveTransform(self.trans_matx1, self.trans_matx2)
    return cv2.warpPerspective(img,M,(45,45))