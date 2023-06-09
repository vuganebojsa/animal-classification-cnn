
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def setup_dataset(data_set):
    data_set = data_set.reshape(len(data_set), 32, 32, 3)
    data_set = data_set/255.0
    return data_set


def load_datasets():
    array_shape = (0, 32, 32, 3)

    X_train = np.empty(array_shape, dtype=np.uint8)
    X_val = np.empty(array_shape, dtype=np.uint8)
    Y_train = np.empty(0)
    Y_val = np.empty(0)


    x_dogs_train = np.loadtxt('data/csv/train/dog.csv', delimiter=',')
    x_dogs_train = setup_dataset(x_dogs_train)
    x_cats_train = np.loadtxt('data/csv/train/cats.csv', delimiter=',')
    x_cats_train = setup_dataset(x_cats_train)

    x_wild_train = np.loadtxt('data/csv/train/wild.csv', delimiter=',')
    x_wild_train = setup_dataset(x_wild_train)

    x_dogs_val = np.loadtxt('data/csv/val/dogs.csv', delimiter=',')
    x_dogs_val = setup_dataset(x_dogs_val)

    x_cats_val = np.loadtxt('data/csv/val/cats.csv', delimiter=',')
    x_cats_val = setup_dataset(x_cats_val)

    x_wild_val = np.loadtxt('data/csv/val/wild.csv', delimiter=',')
    x_wild_val = setup_dataset(x_wild_val)

    X_train = np.concatenate((X_train, x_dogs_train), axis=0)
    X_train = np.concatenate((X_train, x_cats_train), axis=0)
    X_train = np.concatenate((X_train, x_wild_train), axis=0)

    Y_train = np.append(Y_train, np.full(len(x_dogs_train), 0))
    Y_train = np.append(Y_train, np.full(len(x_cats_train), 1))
    Y_train = np.append(Y_train, np.full(len(x_wild_train), 2))

    Y_val = np.append(Y_val, np.full(len(x_dogs_val), 0))
    Y_val = np.append(Y_val, np.full(len(x_cats_val), 1))
    Y_val = np.append(Y_val, np.full(len(x_wild_val), 2))


    X_val = np.concatenate((X_val, x_dogs_val), axis=0)
    X_val = np.concatenate((X_val, x_cats_val), axis=0)
    X_val = np.concatenate((X_val, x_wild_val), axis=0)

    return X_train, Y_train, X_val, Y_val


if __name__ == '__main__':
    X_train, Y_train, X_val, Y_val = load_datasets()

    print(Y_train[2])
    print(Y_train[5000])
