import os
import json
import argparse
import requests


key_file = 'key.json'
    
def get_token():
    keys = load_json_file(key_file)
    token = keys['token']

    if not token:
        raise Exception("token not exists, run python token.py to gen it.")

    return token

# https://ai.baidu.com/forum/topic/show/867951
def gen_token(ak, sk):
    # https://ai.baidu.com/ai-doc/REFERENCE/Ck3dwjhhu
    url = 'https://aip.baidubce.com/oauth/2.0/token'
    params = {
        'grant_type': 'client_credentials',
        'client_id': ak,
        'client_secret': sk,
    }
    r = requests.post(url, params=params)
    return r.json()['access_token']

def load_json_file(json_file):
    if not os.path.exists(json_file):
        return dict()
    
    with open(json_file, 'r') as f:
        o = json.load(f)
    return o

def dump_json_file(o, json_file):
    with open(json_file, 'w') as f:
        json.dump(o, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='generate token of baidu ocr api')
    parser.add_argument('-a', '--ak', type=str, required=True, help='api key')
    parser.add_argument('-s', '--sk', type=str, required=True, help='secret key')
    args = vars(parser.parse_args())

    ak = args['ak']
    sk = args['sk']
    keys = load_json_file(key_file)

    token = gen_token(ak, sk)
    keys['ak'] = ak
    keys['sk'] = sk
    keys['token'] = token

    dump_json_file(keys, key_file)

    print("generating token ... done.")
