import base64
import cv2
import numpy as np
import requests
import json
import time
import os
from imutils import paths
from skimage.metrics import structural_similarity
import argparse
import shutil
from shutil import copyfile

import hash

# https://ai.baidu.com/forum/topic/show/867951
def get_token(ak, sk):
    # https://ai.baidu.com/ai-doc/REFERENCE/Ck3dwjhhu
    url = 'https://aip.baidubce.com/oauth/2.0/token'
    params = {
        'grant_type': 'client_credentials',
        'client_id': ak,
        'client_secret': sk,
    }
    r = requests.post(url, params=params)
    return r.json()['access_token']


# https://ai.baidu.com/tech/ocr/general
def ocr(img):
    # https://ai.baidu.com/ai-doc/OCR/zk3h7xz52
    url = 'https://aip.baidubce.com/rest/2.0/ocr/v1/general_basic'
    params = {'access_token': TOKEN}
    headers = {'Content-Type': 'application/x-www-form-urlencoded'}

    _, img = cv2.imencode('.jpg', img)
    img = base64.b64encode(img)
    data = {'image': img}

    r = requests.post(url, data=data, params=params, headers=headers)
    return r.json()


# https://cloud.baidu.com/doc/IMAGERECOGNITION/s/Xk3bcxe21
def reco_img(img):
    url = "https://aip.baidubce.com/rest/2.0/image-classify/v2/advanced_general"
    with open('./baidu_id.json', 'r') as f:
        k = json.load(f)

    params = {'access_token': k['image']['token']}
    headers = {'Content-Type': 'application/x-www-form-urlencoded'}

    _, img = cv2.imencode('.jpg', img)
    img = base64.b64encode(img)
    data = {"image": img}

    r = requests.post(url, data=data, params=params, headers=headers)
    return r.json()


"""
kind: 'text' or 'image'
"""
def annotate(kind):
    idx_file = './dataset/annotation/baidu_{}.json'.format(kind)
    idx = load_json_file(idx_file)

    image_paths = list(paths.list_images('./dataset/raw/{}/'.format(kind)))
    for image_path in image_paths:
        basename = os.path.basename(image_path)
        (filename, ext) = os.path.splitext(basename)
        
        key = filename
        if key in idx:
            value = idx[key]
            print('image: {}, label: {}'.format(key, value))
            continue
        
        img = cv2.imread(image_path)
        # https://cloud.baidu.com/doc/OCR/s/zk3h7xz52#%E8%AF%B7%E6%B1%82%E8%AF%B4%E6%98%8E
        # width and height are 15px at least
        (h, w, _) = img.shape
        if h < 15:
            size = (int(w * 15 / h), 15)
            img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)

        if kind == 'text':
            res = ocr(img)
            if res.get('error_code', None):
                # qps limit
                continue
            elif res.get('words_result_num') == 0:
                # can't recoganize
                value = 'unknown'
            else:
                # most possible value
                value = res['words_result'][0]['words']
        elif kind == 'image':
            res = reco_img(img)
            if res.get('error_code', None):
                # qps limit
                continue
            elif res.get('result_num') == 0:
                # can't recoganize
                value = 'unknown'
            else:
                # most possible value
                value = res['result'][0]['keyword']

        idx[key] = value

        print('image: {}, label: {}'.format(key, value))

        dump_json_file(idx, idx_file)

        time.sleep(0.1)

def similarity(one, two):
    (h1, w1, _) = one.shape
    (h2, w2, _) = two.shape

    h = max(h1, h2)
    w = max(w1, w2)

    one = cv2.copyMakeBorder(one,
                             (h - h1) // 2,
                             h - h1 - (h - h1) // 2,
                             (w - w1) // 2,
                             w - w1 - (w - w1) // 2,
                             cv2.BORDER_CONSTANT,
                             value=[255, 255, 255])

    two = cv2.copyMakeBorder(two,
                             (h - h2) // 2,
                             h - h2 - (h - h2) // 2,
                             (w - w2) // 2,
                             w - w2 - (w - w2) // 2,
                             cv2.BORDER_CONSTANT,
                             value=[255, 255, 255])

    one = cv2.cvtColor(one, cv2.COLOR_BGR2GRAY)
    two = cv2.cvtColor(two, cv2.COLOR_BGR2GRAY)

    (score, diff) = structural_similarity(one, two, full=True)
    return score

def load_json_file(json_file):
    if not os.path.exists(json_file):
        return dict()
    
    with open(json_file, 'r') as f:
        o = json.load(f)
    return o

def dump_json_file(o, json_file):
    with open(json_file, 'w') as f:
        json.dump(o, f)
    
def load_label_file(label_file):
    if not os.path.exists(label_file):
        return list()

    with open(label_file, 'r') as f:
        l = f.read().splitlines()
    return l

def cluster():
    baidu_mark = load_json_file('./dataset/annotation/text_baidu.json')
    labels = load_label_file('./dataset/annotation/label.txt')

    sim_mark_file = './dataset/annotation/text_similarity.json'
    sim_mark = load_json_file(sim_mark_file)

    correct_mark = dict()
    for k, v in baidu_mark.items():
        if v in labels:
            correct_mark[k] = v
            
    for k, v in baidu_mark.items():
        if k in sim_mark:
            print('{} is already marked'.format(k, v))
            continue

        if k in correct_mark:
            sim_mark[k] = v
            dump_json_file(sim_mark, sim_mark_file)
            print('{} is marked as {}'.format(k, v))
            continue

        score = 0
        label = None
        for ck, cv in correct_mark.items():
            one = './dataset/raw/text/{}.jpg'.format(k)
            two = './dataset/raw/text/{}.jpg'.format(ck)

            one = cv2.imread(one)
            two = cv2.imread(two)
            
            s = similarity(one, two)
            if s > score:
                score = s
                label = cv

        if score > 0.5:
            sim_mark[k] = label
            print('{} is marked as {} with score {}'.format(k, label, score))
        else:
            sim_mark[k] = 'unknown'
            print('{} is marked as unknown'.format(k))

        dump_json_file(sim_mark, sim_mark_file)
        
def anno_text():
    idx = load_json_file('./dataset/annotation/text_similarity.json')

    total = 0
    text_anno_dir = './dataset/annotation/text/'
    for k, v in idx.items():
        text_src = './dataset/raw/text/{}.jpg'.format(k)
        text_dst = './dataset/annotation/text/{}/{}.jpg'.format(v, k)

        text_dst_dir = os.path.dirname(text_dst)
        if not os.path.exists(text_dst_dir):
            os.mkdir(text_dst_dir)

        copyfile(text_src, text_dst)
        total += 1

    print(total)


def anno_image():
    # load text label
    text_dir = './dataset/annotation/text/'
    text_class = dict()

    for image_path in list(paths.list_images(text_dir)):
        basename = os.path.basename(image_path)
        (filename, ext) = os.path.splitext(basename)
        label = os.path.basename(os.path.dirname(image_path))
        text_class[filename] = label

    # every image get the same label
    image_dir = './dataset/raw/image/'
    anno_dir = './dataset/annotation/image/'
    for image_path in list(paths.list_images(image_dir)):
        basename = os.path.basename(image_path)
        (filename, ext) = os.path.splitext(basename)
        key = filename.split('_')[0]
        label = text_class[key]

        print('{} has label {}'.format(key, label))

        target_dir = anno_dir + label
        if not os.path.exists(target_dir):
            os.mkdir(target_dir)

        shutil.copy(image_path, target_dir)

if __name__ == '__main__':
    anno_image()
