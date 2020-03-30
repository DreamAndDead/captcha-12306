import requests
import base64
import argparse
import json
import os
import time


def fetch_captcha():
    url = 'https://kyfw.12306.cn/passport/captcha/captcha-image64'
    r = requests.get(url)
    img_b64 = json.loads(r.content)['image']
    return base64.b64decode(img_b64)

def download_captcha(directory, serial):
    filename = str(serial).zfill(5) + '.jpg'
    filepath = os.path.join(directory, filename)

    if os.path.exists(filepath):
        print("have downloaded captcha %d." % serial)
    else:
        print("downloading captcha %d ... " % serial, end='')
        with open(filepath, 'wb') as f:
            f.write(fetch_captcha())
            print("end.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='download 12306 captchas')
    parser.add_argument('-d', '--dir', type=str, required=True, help='directory to save captchas')
    parser.add_argument('-n', '--num', type=int, required=True, help='total captcha numbers to download')
    args = vars(parser.parse_args())

    os.makedirs(args['dir'], exist_ok=True)
    
    serial = 0
    while serial < args['num']:
        try:
            download_captcha(args['dir'], serial)
            serial += 1
        except Exception as e:
            print(e)
            print('download captcha %d failed!' % serial)
        
        time.sleep(0.1)

        if serial % 10 == 0:
            time.sleep(1)
        if serial % 100 == 0:
            time.sleep(2)
        if serial % 1000 == 0:
            time.sleep(4)            
