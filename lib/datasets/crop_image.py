import cv2
import os
import numpy as np

# data directory 
data_path = "./data_dir/"

# crop each image in data directory
for filename in os.listdir(data_path):
    print(filename)
    image = cv2.imread(data_path + filename)
    width = image.shape[0]
    height = image.shape[1]
    crop_image = image[int(width*0.25):int(width*0.75), int(height*0.25):int(height*0.75), 0:3]
    # put cropped images into new directory
    cv2.imwrite('./data_crop/' + filename, crop_image)
    