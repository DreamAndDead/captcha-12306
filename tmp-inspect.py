import cv2
import numpy as np


if __name__ == '__main__':
    f = './data/datasets/captcha.npz'
    f = './data/datasets/dataset.npz'
    f = './data/datasets/captcha.test.npz'

    data = np.load(f)
    trainX, trainY = data['images'], data['labels']

    print(trainX.shape, trainY.shape)

    cv2.imshow('i', trainX[0])
    cv2.waitKey(0)

    exit()
    sample_weight = trainY.max(axis=1) / np.sqrt(trainY.sum(axis=1))
    
    sample_weight /= sample_weight.mean()
    print(trainY.argmax(axis=1))

