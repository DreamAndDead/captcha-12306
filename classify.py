import cv2
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
    parser = argparse.ArgumentParser(description='classify raw data using model')
    parser.add_argument("-t", "--type", choices=['text', 'image'], required=True, help="dataset type")
    parser.add_argument('-m', '--model', type=str, required=True, help='model directory')
    parser.add_argument('-r', '--raw', type=str, required=True, help='raw data directory')
    parser.add_argument('-o', '--output', type=str, required=True, help='which directory to save classified data')
    args = vars(parser.parse_args())

    state = TrainingState(args['model'])
    loader = DataLoader(args['type'])

    X, Y, file_paths = loader.load_dataset(args['raw'])

    model = state.load_best_model()
    res = model.predict_proba(X)

    le_path = os.path.join(args['model'], 'label_encoder.pkl')
    le = loader.load_label_encoder(le_path)
    
    for file_path, label in zip(file_paths, le.inverse_transform(res)):
        output_path = os.path.join(args['output'], label)
        os.makedirs(output_path, exist_ok=True)
        shutil.copy(file_path, output_path)

        print("label %s as %s." % (os.path.basename(file_path), label))
