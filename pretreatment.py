# coding: utf-8
import hashlib
import os
import pathlib

import cv2
import numpy as np
import requests
import scipy.fftpack
import json
import base64

PATH = 'datasets/captcha'

def get_text(img, offset=0):
    # 3, 22, 120, 177 are magic numbers
    return img[3:22, 120 + offset:177 + offset]

def _get_imgs(img):
    # 8 images
    interval = 5
    length = 67
    for x in range(40, img.shape[0] - length, interval + length):
        for y in range(interval, img.shape[1] - length, interval + length):
            yield img[x:x + length, y:y + length]

def get_imgs(img):
    imgs = []
    for img in _get_imgs(img):
        imgs.append(img)
    return imgs


def pretreat():
    texts, imgs = [], []
    for img in os.listdir(PATH):
        img = os.path.join(PATH, img)
        img = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
        texts.append(get_text(img))
        imgs.append(get_imgs(img))
    return texts, imgs

 
def load_data():
    texts, images = pretreat()
    return texts, images


if __name__ == '__main__':
    texts, imgs = load_data()
    for t in texts:
        print(t.shape)
        cv2.imshow('t', t)
    for i in imgs:
        for j in i:
            print(j.shape)

