from keras.models import load_model
from download import fetch_captcha
import cv2
import numpy as np
from clip import clip_text, clip_image
import pickle
from imutils import paths
import shutil
import os

import tensorflow as tf
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.InteractiveSession(config=config)


def eval_text(captcha):
    # split text part
    text = clip_text(captcha)

    # load model
    model = load_model('model/text-classifier.hdf5')

    # predict result
    w = h = 32
    x = cv2.cvtColor(text, cv2.COLOR_BGR2GRAY)
    x = cv2.resize(x, (w, h), interpolation=cv2.INTER_CUBIC)
    x = x.reshape((h, w, 1))
    res = model.predict(np.array([x]))

    with open('model/text-label.pkl', 'rb') as f:
        l = pickle.load(f)

    print(l.inverse_transform(res))


def eval_image(captcha):
    # split text part
    images = clip_image(captcha)

    # load model
    model = load_model('temp/4/best_model.h5')

    # predict result
    w = h = 67
    X = np.array(images)
    res = model.predict_proba(X)

    print(res)

    with open('temp/4/label_encoder.pkl', 'rb') as f:
        l = pickle.load(f)

    print(l.inverse_transform(res))


def load_unlabel_image(image_dir):
    data_dir = image_dir
    images = list(paths.list_images(data_dir))

    X = []
    p = []

    for img_path in images:
        x = cv2.imread(img_path)
        mean = [103.939, 116.779, 123.68]
        x = x.astype('float32') - mean
        X.append(x)
        p.append(img_path)

    return np.array(X), p


def old():
    X, p = load_unlabel_image('./dataset/raw/image/')
    model = load_model('./temp/3/best_model.h5')
    res = model.predict_proba(X, batch_size=32)

    with open('temp/3/label_encoder.pkl', 'rb') as f:
        l = pickle.load(f)

    for p, l in zip(p, l.inverse_transform(res)):
        dst = './dataset/annotation/prob-image-4/{}'.format(l)
        os.makedirs(dst, exist_ok=True)
        shutil.copy(p, dst)

if __name__ == '__main__':
    # download a new captcha
    #captcha = fetch_captcha()
    #captcha = cv2.imdecode(np.frombuffer(captcha, np.uint8), cv2.IMREAD_COLOR)

    X, p = load_unlabel_image('./dataset/raw/image/')
    model = load_model('./temp/12306.image.model.h5')
    res = model.predict_classes(X, batch_size=32)

    with open('./temp/texts.txt', 'r') as f:
        labels = f.read().split('\n')

    for p, l in zip(p, res):
        dst = './dataset/annotation/image/{}'.format(labels[l])
        os.makedirs(dst, exist_ok=True)
        shutil.copy(p, dst)
        
