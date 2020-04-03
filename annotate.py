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

def get_captcha_text_label(anno_text_dir):
    serial_labels = {}
    
    for file_path in list(paths.list_images(anno_text_dir)):
        basename = os.path.basename(file_path)
        (serial, ext) = os.path.splitext(basename)
        label = os.path.basename(os.path.dirname(file_path))

        serial_labels[serial] = label

    return serial_labels
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='annotate text/image raw data')
    parser.add_argument('-t', '--type', choices=['text', 'image'], required=True, help='data type')
    parser.add_argument('-r', '--raw', type=str, required=True, help='raw data directory to annotate')
    parser.add_argument('-a', '--annotated_text', type=str, help='annotated text dir, only need when type is image')
    parser.add_argument('-o', '--output', type=str, required=True, help='target directory to save annotated result')
    args = vars(parser.parse_args())

    raw_type = args['type']
    raw_dir = args['raw']
    anno_text_dir = args['annotated_text']
    output_dir = args['output']

    if raw_type == 'image' and not anno_text_dir:
        print("-a option is needed when type is image")
        exit()

    if raw_type == 'image':
        serial_labels = get_captcha_text_label(anno_text_dir)
    
    raw_paths = list(paths.list_images(raw_dir))
    output_paths = list(paths.list_images(output_dir))
    output_files = list(map(os.path.basename, output_paths))

    for raw_path in raw_paths:
        basename = os.path.basename(raw_path)
        if basename in output_files:
            print("%s is already annotated." % basename)
            continue

        raw_data = cv2.imread(raw_path)

        if raw_type == 'text':
            # https://cloud.baidu.com/doc/OCR/s/zk3h7xz52#%E8%AF%B7%E6%B1%82%E8%AF%B4%E6%98%8E
            # width and height are 15px at least
            (h, w, _) = raw_data.shape
            if h < 15:
                size = (int(w * 15 / h), 15)
                raw_data = cv2.resize(raw_data, size, interpolation=cv2.INTER_AREA)

            res = ocr(raw_data)
            time.sleep(0.5)

            if res.get('error_code', None):
                # such as qps limit, etc.
                print("some error returns from ocr api.")
                break
            elif res.get('words_result_num') == 0:
                # can't recoganize
                label = 'unknown'
            else:
                # most possible value
                label = res['words_result'][0]['words']
        elif raw_type == 'image':
            serial, _ = basename.split('_')
            label = serial_labels[serial]

        print("recoganize %s as label %s" % (basename, label))
        
        output_path = os.path.join(output_dir, label, basename)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        shutil.copy(raw_path, output_path)


