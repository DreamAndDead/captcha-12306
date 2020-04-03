import argparse
import pickle
import cv2
import json
import os
import numpy as np
from imutils import paths
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from keras import optimizers
from keras.models import Sequential
from keras.models import load_model
from keras.callbacks import Callback
from keras.applications import VGG16
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers import Dense, Activation, Flatten, BatchNormalization, GlobalAveragePooling2D, Dropout

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import tensorflow as tf
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.InteractiveSession(config=config)


def create_model(model_type, shape, classes):
    width, height, depth = shape
    inputShape = (height, width, depth)

    if model_type == 'text':
        model = Sequential([
            Conv2D(20, (5, 5), padding="same", activation='relu', input_shape=inputShape),
            MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),

            Conv2D(20, (3, 3), padding="same", activation='relu'),
            MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),

            Flatten(),
            BatchNormalization(),

            Dense(256, activation='relu'),
            Dropout(0.20),

            Dense(classes, activation='softmax'),
        ])

        model.compile(loss="categorical_crossentropy",
                      optimizer=optimizers.SGD(lr=1e-3, momentum=0.9),
                      metrics=["accuracy"])

    elif model_type == 'image':
        base = VGG16(weights='imagenet', include_top=False, input_shape=(height, width, depth))

        for layer in base.layers:
            layer.trainable = False
    
        model = Sequential([
            base,
            BatchNormalization(),
            
            GlobalAveragePooling2D(),
            BatchNormalization(),

            Dense(1024, activation='relu'),
            Dropout(0.50),
            BatchNormalization(),

            Dense(classes, activation='softmax')
        ])
    
        model.compile(loss="categorical_crossentropy",
                      optimizer=optimizers.SGD(lr=1e-2, momentum=0.9),
                      metrics=["accuracy"])

    return model


class DataLoader:
    def __init__(self, data_type):
        self.data_type = data_type

    def load_dataset(self, data_dir):
        file_paths = list(paths.list_images(data_dir))

        X = []
        Y = []
        for file_path in file_paths:
            x = cv2.imread(file_path)
            x = self.preprocess(x)
            y = os.path.basename(os.path.dirname(file_path))

            X.append(x)
            Y.append(y)

        X = np.array(X)
        Y = np.array(Y)
        
        self.le = LabelBinarizer()
        Y = self.le.fit_transform(Y)

        return X, Y, file_paths
    
    def get_label_classes(self):
        return self.le.classes_
    
    def get_data_shape(self):
        if self.data_type == 'text':
            shape = (32, 32, 1)
        elif self.data_type == 'image':
            shape = (67, 67, 3)

        return shape

    def preprocess(self, x):
        width, height, depth = self.get_data_shape()

        if self.data_type == 'text':
            x = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)
            x = cv2.resize(x, (width, height), interpolation=cv2.INTER_CUBIC)
            x = x.reshape((height, width, 1))
            x = x.astype('float32') / 255.0
        elif self.data_type == 'image':
            mean = [103.939, 116.779, 123.68]
            x = cv2.resize(x, (width, height), interpolation=cv2.INTER_CUBIC)
            x = x.astype('float32') - mean

        return x

    def save_label_encoder(self, le_file):
        pickle.dump(self.le, open(le_file, 'wb'))

    def load_label_encoder(self, le_file):
        return pickle.load(open(le_file, 'rb'))


class TrainingState(Callback):
    def __init__(self, state_dir):
        super(TrainingState, self).__init__()

        self.state_dir = state_dir

        self.json_log_file = os.path.join(self.state_dir, 'logs.json')
        self.png_log_file = os.path.join(self.state_dir, 'logs.png')
        self.last_model_file = os.path.join(self.state_dir, 'last_model.h5')
        self.best_model_file = os.path.join(self.state_dir, 'best_model.h5')

        if os.path.exists(self.json_log_file):
            with open(self.json_log_file, 'r') as f:
                self.history = json.load(f)
        else:
            self.history = {
                'epoch': -1,
                'best': 0
            }

    def get_initial_epoch(self):
        return self.history['epoch'] + 1

    def load_last_model(self):
        if os.path.exists(self.last_model_file):
            return load_model(self.last_model_file)
        else:
            return None

    def load_best_model(self):
        if os.path.exists(self.best_model_file):
            return load_model(self.best_model_file)
        else:
            return None

    def on_epoch_end(self, epoch, logs=None):
        self.history['epoch'] = epoch
        
        logs = logs or None
        for k, v in logs.items():
            self.history.setdefault(k, []).append(float(v))

        self.save_json_log()
        self.save_png_log()
        self.save_last_model()
        self.save_best_model()

    def save_json_log(self):
        with open(self.json_log_file, 'w') as f:
            json.dump(self.history, f)

        print('save json log to {}'.format(self.json_log_file))

    def save_png_log(self):
        history = self.history
        size = history['epoch'] + 1
        
        plt.style.use("ggplot")
        plt.figure()
        plt.plot(np.arange(0, size), history["loss"], label="train_loss")
        plt.plot(np.arange(0, size), history["val_loss"], label="val_loss")
        plt.plot(np.arange(0, size), history["accuracy"], label="train_acc")
        plt.plot(np.arange(0, size), history["val_accuracy"], label="val_acc")
        plt.title("Training Loss and Accuracy")
        plt.xlabel("Epoch #")
        plt.ylabel("Loss/Accuracy")
        plt.legend()
        plt.savefig(self.png_log_file)
        plt.close('all')

        print('save png log to {}'.format(self.png_log_file))

    def save_last_model(self):
        self.model.save(self.last_model_file)
        print('save last model to {}'.format(self.last_model_file))

    def save_best_model(self):
        epoch = self.history['epoch']
        best = self.history['best']
        val_acc = self.history['val_accuracy']
        
        if val_acc[-1] > best:
            self.history['best'] = val_acc[-1]
            self.model.save(self.best_model_file)
            print('val_acc inc from {} to {}, save best model to {}'.format(best, val_acc[-1], self.best_model_file))
        else:
            print('no inc in val_acc ...')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--epoch", type=int, default=100, help="how many epoch to train")
    parser.add_argument("-t", "--type", required=True, choices=['text', 'image'], help="text or image dataset")
    parser.add_argument("-d", "--dataset", required=True, help="dataset dir")
    parser.add_argument("-o", "--output", required=True, help="output dir to save training state")
    args = vars(parser.parse_args())

    epoch = args['epoch']
    dataset_type = args['type']
    dataset_dir = args['dataset']
    output_dir = args['output']

    data_loader = DataLoader(dataset_type)
    shape = data_loader.get_data_shape()
    
    batch_size=32
    classes = 80

    os.makedirs(output_dir, exist_ok=True)
    label_encoder_file = os.path.join(output_dir, 'label_encoder.pkl')
    report_file = os.path.join(output_dir, 'classification_report.txt')

    trainingState = TrainingState(output_dir)
    
    model = trainingState.load_last_model()
    if not model:
        model = create_model(dataset_type, shape, classes)
        
    model.summary()
    
    X, Y, _ = data_loader.load_dataset(dataset_dir)
    trainX, testX, trainY, testY = train_test_split(X, Y, test_size=0.2, random_state=42)

    data_loader.save_label_encoder(label_encoder_file)

    if dataset_type == 'text':
        model.fit(trainX, trainY,
                  validation_data=(testX, testY),
                  batch_size=batch_size,
                  epochs=epoch,
                  initial_epoch=trainingState.get_initial_epoch(),
                  callbacks=[trainingState],
                  verbose=1)
    elif dataset_type == 'image':
        aug = ImageDataGenerator(horizontal_flip=True, vertical_flip=True)
        model.fit_generator(aug.flow(trainX, trainY),
                            validation_data=(testX, testY),
                            steps_per_epoch=len(trainX) // batch_size,
                            epochs=epoch,
                            initial_epoch=trainingState.get_initial_epoch(),
                            callbacks=[trainingState],
                            verbose=1)

    predictions = model.predict(testX, batch_size=64)
    report = classification_report(testY.argmax(axis=1),
                                   predictions.argmax(axis=1),
                                   target_names=data_loader.get_label_classes())
    
    with open(report_file, 'w') as f:
        f.write(report)
    
