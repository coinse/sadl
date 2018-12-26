import argparse

import numpy as np
import tensorflow as tf
import keras.backend as K

from keras.datasets import mnist, cifar10
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from keras.regularizers import l2


CLIP_MIN = -0.5
CLIP_MAX = 0.5


def train(args):
    if args.d == "mnist":
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train = x_train.reshape(-1, 28, 28, 1)
        x_test = x_test.reshape(-1, 28, 28, 1)

        layers = [
            Conv2D(64, (3, 3), padding="valid", input_shape=(28, 28, 1)),
            Activation("relu"),
            Conv2D(64, (3, 3)),
            Activation("relu"),
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(0.5),
            Flatten(),
            Dense(128),
            Activation("relu"),
            Dropout(0.5),
            Dense(10),
        ]

    elif args.d == "cifar":
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()

        layers = [
            Conv2D(32, (3, 3), padding="same", input_shape=(32, 32, 3)),
            Activation("relu"),
            Conv2D(32, (3, 3), padding="same"),
            Activation("relu"),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(64, (3, 3), padding="same"),
            Activation("relu"),
            Conv2D(64, (3, 3), padding="same"),
            Activation("relu"),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(128, (3, 3), padding="same"),
            Activation("relu"),
            Conv2D(128, (3, 3), padding="same"),
            Activation("relu"),
            MaxPooling2D(pool_size=(2, 2)),
            Flatten(),
            Dropout(0.5),
            Dense(1024, kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)),
            Activation("relu"),
            Dropout(0.5),
            Dense(512, kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)),
            Activation("relu"),
            Dropout(0.5),
            Dense(10),
        ]

    x_train = x_train.astype("float32")
    x_test = x_test.astype("float32")
    x_train = (x_train / 255.0) - (1.0 - CLIP_MAX)
    x_test = (x_test / 255.0) - (1.0 - CLIP_MAX)

    y_train = np_utils.to_categorical(y_train, 10)
    y_test = np_utils.to_categorical(y_test, 10)

    model = Sequential()
    for layer in layers:
        model.add(layer)
    model.add(Activation("softmax"))

    print(model.summary())
    model.compile(
        loss="categorical_crossentropy", optimizer="adadelta", metrics=["accuracy"]
    )

    model.fit(
        x_train,
        y_train,
        epochs=50,
        batch_size=128,
        shuffle=True,
        verbose=1,
        validation_data=(x_test, y_test),
    )

    model.save("./model/model_{}.h5".format(args.d))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", required=True, type=str)
    args = parser.parse_args()
    assert args.d in ["mnist", "cifar"], "Dataset should be either 'mnist' or 'cifar'"

    train(args)
