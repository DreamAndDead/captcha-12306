import numpy as np
import cv2
from imutils import paths
import os
import argparse


def clip_text(captcha):
    text = captcha[0:28, 116:]
    gray = cv2.cvtColor(text, cv2.COLOR_BGR2GRAY)

    (T, thresh) = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    (x, y, w, h) = cv2.boundingRect(thresh)
    padding = 1
    tl = (x - padding, y - padding)
    br = (x + w + padding, y + h + padding)
    roi = text[tl[1]:br[1], tl[0]:br[0]]

    return roi

def clip_image(captcha):
    x_offset = 5
    y_offset = 41
    x_margin = 5
    y_margin = 5
    img_width = img_height = 67

    images = list()
    for i in range(8):
        row, col = divmod(i, 4)
        top = y_offset + (img_height + y_margin) * row
        left = x_offset + (img_width + x_margin) * col
        bottom = top + img_height
        right = left + img_width
        images.append(captcha[top:bottom, left:right])

    return images

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='parse and save text part of captcha')
    parser.add_argument('-c', '--captcha', type=str, required=True, help='captcha directory to parse')
    parser.add_argument('-t', '--text', type=str, required=True, help='target directory to save text part')
    parser.add_argument('-i', '--image', type=str, required=True, help='target directory to save image part')
    args = vars(parser.parse_args())

    image_paths = list(paths.list_images(args["captcha"]))
    for captcha_path in image_paths:
        basename = os.path.basename(captcha_path)
        (filename, ext) = os.path.splitext(basename)

        captcha = cv2.imread(captcha_path)
        text = clip_text(captcha)
        text_path = os.path.join(args["text"], basename)
        cv2.imwrite(text_path, text)

        images = clip_image(captcha)
        for i, img in enumerate(images):
            img_name = '{}_{}{}'.format(filename, str(i), ext)
            img_path = os.path.join(args["image"], img_name)
            cv2.imwrite(img_path, img)
