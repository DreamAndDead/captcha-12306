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
    parser.add_argument('-m', '--model_dir', type=str, required=True, help='')
    parser.add_argument("-t", "--type", choices=['text', 'image'], required=True, help="text or image dataset")
    parser.add_argument('-d', '--dataset_dir', type=str, required=True, help='')
    parser.add_argument('-o', '--output_dir', type=str, required=True, help='')
    args = vars(parser.parse_args())

    state = TrainingState(args['model_dir'])
    loader = DataLoader(args['type'])

    X, Y, file_paths = loader.load_dataset(args['dataset_dir'])

    model = state.load_best_model()
    res = model.predict_proba(X[:10])

    le_path = os.path.join(args['model_dir'], 'label_encoder.pkl')
    le = loader.load_label_encoder(le_path)
    
    for file_path, label in zip(file_paths, le.inverse_transform(res)):
        output_path = os.path.join(args['output_dir'], label)
        os.makedirs(output_path, exist_ok=True)
        shutil.copy(file_path, output_path)
