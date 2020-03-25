import cv2
from imutils import paths
from os import path
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

from keras.models import Sequential
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers import Dense, Activation, Flatten
from keras import backend as K
from keras.optimizers import SGD
from sklearn.metrics import classification_report
from keras.callbacks import ModelCheckpoint
import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import tensorflow as tf
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.InteractiveSession(config=config)


def load_dataset(size):
    (w, h, d) = size
    data_dir = './dataset/annotation/text/'
    images = list(paths.list_images(data_dir))
    X = []
    Y = []
    for img_path in images:
        x = cv2.imread(img_path)
        if d == 1:
            x = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)
            x = cv2.resize(x, (w, h), interpolation=cv2.INTER_CUBIC)
            x = x.reshape((h, w, 1))
        else:
            x = cv2.resize(x, (w, h), interpolation=cv2.INTER_CUBIC)

        x = x.astype('float') / 255.0
        y = path.basename(path.dirname(img_path))

        X.append(x)
        Y.append(y)

    return np.array(X), np.array(Y)


def build_network(width, height, depth, classes):
    model = Sequential()
    inputShape = (height, width, depth)

    # LeNet
    # first set of CONV => RELU => POOL layers
    model.add(Conv2D(20, (5, 5), padding="same",
                     input_shape=inputShape))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # second set of CONV => RELU => POOL layers
    model.add(Conv2D(50, (5, 5), padding="same"))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # first (and only) set of FC => RELU layers
    model.add(Flatten())
    model.add(Dense(500))
    model.add(Activation("relu"))

    # softmax classifier
    model.add(Dense(classes))
    model.add(Activation("softmax"))

    return model


def train_text():
    width = height = 32
    depth = 1
    classes = 80

    # load dataset
    X, Y = load_dataset((width, height, depth))

    # label category
    l = LabelBinarizer()
    Y = l.fit_transform(Y)
    labels = l.classes_

    with open('model/text-label.pkl', 'wb') as f:
        pickle.dump(l, f)

    # split train and test
    (trainX, testX, trainY, testY) = train_test_split(X, Y, test_size=0.2)

    # compose network
    model = build_network(width, height, depth, classes)

    # train network
    opt = SGD(lr=0.05)
    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

    # save the min val loss model
    checkpoint = ModelCheckpoint('model/text-classifier.hdf5', monitor="val_loss", mode="min",
                                 save_best_only=True, save_weights_only=False, verbose=1)
    callbacks = [checkpoint]

    # fit
    epoch = 20
    batch_size = 32
    H = model.fit(trainX, trainY, validation_data=(testX, testY),
                  batch_size=batch_size, epochs=epoch, callbacks=callbacks, verbose=1)

    predictions = model.predict(testX, batch_size=batch_size)
    print(classification_report(testY.argmax(axis=1),
                                predictions.argmax(axis=1),
                                target_names=labels))


    # plot train result
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(np.arange(0, epoch), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, epoch), H.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, epoch), H.history["accuracy"], label="train_acc")
    plt.plot(np.arange(0, epoch), H.history["val_accuracy"], label="val_acc")
    plt.title("Training Loss and Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend()
    plt.savefig('model/text.png')


if __name__ == '__main__':
    train_text()
