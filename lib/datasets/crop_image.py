from PIL import Image
import os
import numpy as np

def crop_image(data_path):
    for filename in os.listdir(data_path):
        image = Image.open(data_path + filename)
        return image.resize((70,70))
