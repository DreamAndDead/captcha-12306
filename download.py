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

    with open(filepath, 'wb') as f:
        f.write(fetch_captcha())

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='download 12306 captcha images')
    parser.add_argument('-d', '--dir', type=str, required=True, help='directory to save the original captchas')
    parser.add_argument('-s', '--start', type=int, required=True, help='first captcha number to save')
    parser.add_argument('-n', '--num', type=int, required=True, help='how many captchas to download')
    args = vars(parser.parse_args())
    
    offset = args['start']
    total = 0
    while total < args['num']:
        print('begin to download captcha %d ...' % offset)

        try:
            download_captcha(args['dir'], offset)
            print('downloaded captcha %d' % offset)
            offset += 1
            total += 1
        except:
            print('captcha %d failed ...' % offset)
        
        time.sleep(1)

        if total % 10 == 0:
            time.sleep(1)
        if total % 100 == 0:
            time.sleep(2)
        if total % 1000 == 0:
            time.sleep(4)            
