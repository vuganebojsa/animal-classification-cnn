import os
from PIL import Image
import numpy as np
import cv2
import pandas as pd
data = []
labels = []
# dog = 0, cat = 1, wild = 2
LABELS = ['dog', 'cat', 'wild']

# train data


def convert_images_to_array(folder_path):
    image_list = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            img_path = os.path.join(folder_path, filename)
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
            img = cv2.resize(img, (32, 32))  # Resize the image to desired dimensions
            image_list.append(img)

    image_array = np.array(image_list)
    return image_array


def save_array_to_csv(array, csv_path):
    df = pd.DataFrame(array.reshape(array.shape[0], -1))  # Reshape array to 2D
    df.to_csv(csv_path, index=False)

# dogs = convert_images_to_array('data/train/dog')
# cats = convert_images_to_array('data/train/cat')
# wild = convert_images_to_array('data/train/wild')
# save_array_to_csv(dogs, 'data/csv/train/dogs.csv')
# save_array_to_csv(cats, 'data/csv/train/cats.csv')
# save_array_to_csv(wild, 'data/csv/train/wild.csv')


# dogs = convert_images_to_array('data/val/dog')
# cats = convert_images_to_array('data/val/cat')
# wild = convert_images_to_array('data/val/wild')
# save_array_to_csv(dogs, 'data/csv/val/dogs.csv')
# save_array_to_csv(cats, 'data/csv/val/cats.csv')
# save_array_to_csv(wild, 'data/csv/val/wild.csv')




