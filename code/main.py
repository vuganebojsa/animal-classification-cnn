
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import keras
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D,Dense,Flatten,Dropout
import cv2
import random
from keras.models import load_model


NUM_OF_EPOCHS = 10
BATCH_SIZE = 50
IMAGE_SIZE = 32
NUM_OF_CLASSES = 3


def setup_dataset(data_set):
    '''
               Reshapes the dataset to be in format (length_of_set, size, size, chanels) and divides by 255.0
               so that the values are in between 0 and 1

           :return: Returns modified dataset
           '''
    data_set = data_set.reshape(len(data_set), IMAGE_SIZE, IMAGE_SIZE, 3)
    data_set = data_set/255.0
    return data_set


def load_datasets():
    '''
            Loads the datasets from the previously created csv files

        :return: Returns X_train, Y_train, X_val, Y_val
        '''
    array_shape = (0, IMAGE_SIZE, IMAGE_SIZE, 3)

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


def shuffled_loaded_dataset():
    '''
        Shuffles the dataset so that the dogs are not all next to dogs, cats not next to cats, etc.

    :return: Returns shuffled values for X_train, Y_train, X_validation, Y_validation
    '''
    X_train, Y_train, X_val, Y_val = load_datasets()
    np.random.seed(42)
    shuffled_indices_train = np.random.permutation(len(X_train))
    shuffled_indices_val = np.random.permutation(len(X_val))

    shuf_x_train = X_train[shuffled_indices_train]
    shuf_y_train = Y_train[shuffled_indices_train]

    shuf_x_val = X_val[shuffled_indices_val]
    shuf_y_val = Y_val[shuffled_indices_val]
    return shuf_x_train, shuf_y_train, shuf_x_val, shuf_y_val


def get_final_dataset():
    X_train, Y_train, X_val, Y_val = shuffled_loaded_dataset()
    # get total different classes
    num_classes = len(np.unique(Y_train))
    # get total ammount of data
    data_length = len(X_train)

    # split the train to train-test 90%-10%
    (x_train, x_test) = X_train[(int)(0.1 * data_length):], X_train[:(int)(0.1 * data_length)]
    (y_train, y_test) = Y_train[(int)(0.1 * data_length):], Y_train[:(int)(0.1 * data_length)]

    # reshape y classes to be like [0 0 1] instead of [2]
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)
    Y_val = keras.utils.to_categorical(Y_val, num_classes)

    return x_train, y_train, x_test, y_test, X_val, Y_val


def convert_to_array(img):
    im = cv2.imread(img)
    img = Image.fromarray(im, 'RGB')
    image = img.resize((IMAGE_SIZE, IMAGE_SIZE))
    return np.array(image)


def get_animal_name(label):
    if label == 0:
        return "dog"
    if label == 1:
        return "cat"
    if label == 2:
        return "wild"
    return -1


def predict_animal(file, model):
    # ar=convert_to_array(file)
    # ar=ar/255.0
    #label=1
    # a=[]
    # a.append(ar)
    # a=np.array(a)
    file = np.expand_dims(file, axis=0)
    score=model.predict(file, verbose=1)
    label_index=np.argmax(score)
    acc=np.max(score)
    animal=get_animal_name(label_index)
    print('-----------------------------')
    print(label_index)
    print(score)
    print(animal)
    print("The predicted Animal is a "+animal+" with accuracy =    "+str(acc))


def get_model():
    model = Sequential()
    model.add(
        Conv2D(filters=16, kernel_size=2, padding="same", activation="relu", input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3)))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Conv2D(filters=32, kernel_size=2, padding="same", activation="relu"))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Conv2D(filters=64, kernel_size=2, padding="same", activation="relu"))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(500, activation="relu"))
    model.add(Dropout(0.2))
    model.add(Dense(NUM_OF_CLASSES, activation="softmax"))
    model.summary()
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(x_train, y_train, validation_data=(X_val, Y_val), batch_size=BATCH_SIZE, epochs=NUM_OF_EPOCHS)
    model.save('model.h5')
    return model


if __name__ == '__main__':

    x_train, y_train, x_test, y_test, X_val, Y_val = get_final_dataset()
    #
    # model = get_model()
    model = load_model('model.h5')
    score = model.evaluate(x_test, y_test, verbose=1)
    print('\n', 'Test accuracy:', score[1])

    for i in range(10):
        idx = random.randint(0, 1000)
        plt.imshow(x_test[idx], interpolation='nearest')
        plt.show()
        predict_animal(x_test[idx], model)
