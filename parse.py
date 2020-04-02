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
from train import TrainingState, DataLoader

import tensorflow as tf
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.InteractiveSession(config=config)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='model in reality')
    parser.add_argument('-t', '--text_model_dir', type=str, required=True, help='')
    parser.add_argument('-i', '--image_model_dir', type=str, required=True, help='')
    parser.add_argument('-c', '--captcha', type=str, help='')
    args = vars(parser.parse_args())

    if not args['captcha']:
        captcha = fetch_captcha()
        captcha = cv2.imdecode(np.frombuffer(captcha, np.uint8), cv2.IMREAD_COLOR)
    else:
        captcha = cv2.imread(args['captcha'])

    state = TrainingState(args['text_model_dir'])
    loader = DataLoader('text')

    text = crop_text(captcha)
    text = loader.preprocess(text)

    model = state.load_best_model()
    res = model.predict(np.array([text]))

    le_path = os.path.join(args['text_model_dir'], 'label_encoder.pkl')
    le = loader.load_label_encoder(le_path)

    print(le.inverse_transform(res))


    state = TrainingState(args['image_model_dir'])
    loader = DataLoader('image')

    images = []
    for i in crop_image(captcha):
        images.append(loader.preprocess(i))

    model = state.load_best_model()
    res = model.predict(np.array(images))

    le_path = os.path.join(args['image_model_dir'], 'label_encoder.pkl')
    le = loader.load_label_encoder(le_path)

    print(le.inverse_transform(res))

    cv2.imshow('captcha', captcha)
    cv2.waitKey(0)
