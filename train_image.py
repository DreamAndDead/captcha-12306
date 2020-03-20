import argparse
import pickle
import cv2
from imutils import paths
import os
import json
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from keras.models import Sequential
from keras import layers
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers import Dense, Activation, Flatten
from keras import backend as K
from keras import optimizers
from keras.callbacks import Callback
from keras.callbacks import ReduceLROnPlateau
from keras.applications import VGG16
from keras.preprocessing.image import ImageDataGenerator
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def create_model(width, height, depth, classes):
    base = VGG16(weights='imagenet', include_top=False, input_shape=(height, width, depth))

    for layer in base.layers[:-4]:
        layer.trainable = False
    
    model = Sequential([
        base,
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.GlobalAveragePooling2D(),
        layers.BatchNormalization(),
        layers.Dense(64, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.20),
        layers.Dense(classes, activation='softmax')
    ])

    model.compile(loss="categorical_crossentropy", optimizer=optimizers.RMSprop(lr=1e-5), metrics=["accuracy"])

    return model


def load_image():
    data_dir = './dataset/annotation/image/'
    images = list(paths.list_images(data_dir))

    X = []
    Y = []
    for img_path in images:
        x = cv2.imread(img_path)
        mean = [103.939, 116.779, 123.68]
        x = x.astype('float32') - mean
        y = os.path.basename(os.path.dirname(img_path))

        X.append(x)
        Y.append(y)

    return np.array(X), np.array(Y)


class DataLoader:
    def load_dataset(self):
        X, Y = load_image()

        X = self.preprocess(X)

        self.le = LabelBinarizer()
        Y = self.le.fit_transform(Y)

        return train_test_split(X, Y, test_size=0.2, random_state=1)

    def preprocess(self, X):
        return X

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

    def on_epoch_end(self, epoch, logs=None):
        self.history['epoch'] = epoch
        
        logs = logs or None
        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)

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
        plt.plot(np.arange(0, size), history["acc"], label="train_acc")
        plt.plot(np.arange(0, size), history["val_acc"], label="val_acc")
        plt.title("Training Loss and Accuracy")
        plt.xlabel("Epoch #")
        plt.ylabel("Loss/Accuracy")
        plt.legend()
        plt.savefig(self.png_log_file)

        print('save png log to {}'.format(self.png_log_file))

    def save_last_model(self):
        self.model.save(self.last_model_file)
        print('save last model to {}'.format(self.last_model_file))

    def save_best_model(self):
        epoch = self.history['epoch']
        best = self.history['best']
        val_acc = self.history['val_acc']
        
        if val_acc[-1] > best:
            self.history['best'] = val_acc[-1]
            self.model.save(self.best_model_file)
            print('val_acc inc from {} to {}, save best model to {}'.format(best, val_acc[-1], self.best_model_file))
        else:
            print('no inc in val_acc ...')
            

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--output", required=True,
                    help="output dir to save training state")
    args = vars(ap.parse_args())

    output_dir = args['output']
    os.makedirs(output_dir, exist_ok=True)

    label_encoder_file = os.path.join(output_dir, 'label_encoder.pkl')
    report_file = os.path.join(output_dir, 'classification_report.txt')

    data_loader = DataLoader()
    trainX, testX, trainY, testY = data_loader.load_dataset()
    data_loader.save_label_encoder(label_encoder_file)


    trainingState = TrainingState(output_dir)
    
    model = trainingState.load_last_model()
    if not model:
        model = create_model(67, 67, 3, 80)
        
    model.summary()


    aug = ImageDataGenerator(horizontal_flip=True, vertical_flip=True)
    
    model.fit_generator(aug.flow(trainX, trainY),
                        validation_data=(testX, testY),
                        epochs=100,
                        steps_per_epoch=len(trainX) // 32,
                        callbacks=[trainingState],
                        initial_epoch=trainingState.get_initial_epoch(),
                        verbose=1)


    predictions = model.predict(testX, batch_size=64)
    report = classification_report(testY.argmax(axis=1),
                                   predictions.argmax(axis=1))
    
    with open(report_file, 'w') as f:
        f.write(report)
    
