import base64
import cv2
import numpy as np
import requests
import json
import time
import os
from imutils import paths
import argparse
import shutil
from shutil import copyfile
from key import get_token

TOKEN = get_token()

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


def annotate(kind):
    image_paths = list(paths.list_images('./dataset/raw/text/'))

    for image_path in image_paths:
        basename = os.path.basename(image_path)
        (filename, ext) = os.path.splitext(basename)
        
        img = cv2.imread(image_path)

        # https://cloud.baidu.com/doc/OCR/s/zk3h7xz52#%E8%AF%B7%E6%B1%82%E8%AF%B4%E6%98%8E
        # width and height are 15px at least
        (h, w, _) = img.shape
        if h < 15:
            size = (int(w * 15 / h), 15)
            img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)

        res = ocr(img)
        if res.get('error_code', None):
            # such as qps limit
            continue
        elif res.get('words_result_num') == 0:
            # can't recoganize
            label = 'unknown'
        else:
            # most possible value
            label = res['words_result'][0]['words']



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='annotate text with baidu ocr api')
    parser.add_argument('-t', '--text', type=str, required=True, help='text directory to annotate')
    parser.add_argument('-o', '--ocr', type=str, required=True, help='target directory to save ocr result')
    args = vars(parser.parse_args())


