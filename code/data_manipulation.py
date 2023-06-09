import os
from PIL import Image
import numpy as np
data = []
labels = []

# dog = 0, cat = 1, wild = 2
LABELS = ['dog', 'cat', 'wild']

# train data


def get_data(image_type, label_order):
    BASE_PATH = '../data/train/'

    images = os.listdir('../data/train/' + image_type)
    for image in images:
        img = Image.open(BASE_PATH + image_type +  '/' + image)
        arr_img = Image.fromarray(img, 'RGB')
        resized_img = arr_img.resize((50, 50))
        data.append(np.array(resized_img))
        labels.append(label_order)


get_data('dog', 0)
get_data('cat', 1)
get_data('wild', 2)


