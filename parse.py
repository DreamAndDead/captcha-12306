import cv2
import pickle
import shutil
import os
import argparse
import numpy as np
from imutils import paths
from train import TrainingState, DataLoader

import tensorflow as tf
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.InteractiveSession(config=config)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='parse captcha in reality')
    parser.add_argument('-t', '--text_model', type=str, required=True, help='text model dir')
    parser.add_argument('-i', '--image_model', type=str, required=True, help='image model dir')
    parser.add_argument('-c', '--captcha', type=str, help='captcha file path')
    args = vars(parser.parse_args())

    if not args['captcha']:
        captcha = fetch_captcha()
        captcha = cv2.imdecode(np.frombuffer(captcha, np.uint8), cv2.IMREAD_COLOR)
    else:
        captcha = cv2.imread(args['captcha'])

    # text
    state = TrainingState(args['text_model'])
    loader = DataLoader('text')

    text = crop_text(captcha)
    text = loader.preprocess(text)

    model = state.load_best_model()
    res = model.predict(np.array([text]))

    le_path = os.path.join(args['text_model'], 'label_encoder.pkl')
    le = loader.load_label_encoder(le_path)

    print("text label:")
    print(le.inverse_transform(res))

    # image
    state = TrainingState(args['image_model'])
    loader = DataLoader('image')

    images = []
    for i in crop_image(captcha):
        images.append(loader.preprocess(i))

    model = state.load_best_model()
    res = model.predict(np.array(images))

    le_path = os.path.join(args['image_model'], 'label_encoder.pkl')
    le = loader.load_label_encoder(le_path)

    print("image labels:")
    print(le.inverse_transform(res))

    
    cv2.imshow('press any key to escape', captcha)
    cv2.waitKey(0)
