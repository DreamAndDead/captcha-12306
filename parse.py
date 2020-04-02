from keras.models import load_model
from download import fetch_captcha
import cv2
import numpy as np
from crop import crop_text, crop_image
import pickle
from imutils import paths
import shutil
import os
import argparse

import tensorflow as tf
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.InteractiveSession(config=config)


def eval_text(model, captcha):
    # split text part
    text = crop_text(captcha)

    # predict result
    w = h = 32
    x = cv2.cvtColor(text, cv2.COLOR_BGR2GRAY)
    x = cv2.resize(x, (w, h), interpolation=cv2.INTER_CUBIC)
    x = x.reshape((h, w, 1))
    res = model.predict(np.array([x]))

    with open('model/text-label.pkl', 'rb') as f:
        l = pickle.load(f)

    print(l.inverse_transform(res))



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='model in reality')
#    parser.add_argument('-t', '--text_model_dir', type=str, required=True, help='')
    parser.add_argument('-i', '--image_model_dir', type=str, required=True, help='')
    args = vars(parser.parse_args())

#    text_model_path = os.path.join(args['text_model_dir'], 'best_model.h5')
#    text_label_encoder_path = os.path.join(args['text_model_dir'], 'label_encoder.pkl')

    image_model_path = os.path.join(args['image_model_dir'], 'best_model.h5')
    image_label_encoder_path = os.path.join(args['image_model_dir'], 'label_encoder.pkl')

    # download a new captcha
    captcha = fetch_captcha()
    captcha = cv2.imdecode(np.frombuffer(captcha, np.uint8), cv2.IMREAD_COLOR)

    # split text part
    images = crop_image(captcha)

    # predict result
    w = h = 67
    X = np.array(images)

    model = load_model(image_model_path)
    res = model.predict_proba(X)

    with open(image_label_encoder_path, 'rb') as f:
        l = pickle.load(f)

    print(l.inverse_transform(res))
