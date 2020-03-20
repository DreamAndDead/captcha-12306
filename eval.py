from keras.models import load_model
from download import fetch_captcha
import cv2
import numpy as np
from clip import clip_text, clip_image
import pickle


def eval_text(captcha):
    # split text part
    text = clip_text(captcha)

    # load model
    model = load_model('model/text-classifier.hdf5')

    # predict result
    w = h = 32
    x = cv2.cvtColor(text, cv2.COLOR_BGR2GRAY)
    x = cv2.resize(x, (w, h), interpolation=cv2.INTER_CUBIC)
    x = x.reshape((h, w, 1))
    res = model.predict(np.array([x]))

    with open('model/text-label.pkl', 'rb') as f:
        l = pickle.load(f)

    print(l.inverse_transform(res))


def eval_image(captcha):
    # split text part
    images = clip_image(captcha)

    # load model
    model = load_model('model/best_model.h5')
    #model = load_model('model/last_model.h5')

    # predict result
    w = h = 67
    X = np.array(images)
    res = model.predict_proba(X)

    print(res)

    with open('model/label_encoder.pkl', 'rb') as f:
        l = pickle.load(f)

    print(l.inverse_transform(res))
    
if __name__ == '__main__':
    # download a new captcha
    captcha = fetch_captcha()
    captcha = cv2.imdecode(np.frombuffer(captcha, np.uint8), cv2.IMREAD_COLOR)

#    captcha = cv2.imread('./download/20200214/00001.jpg')

#    eval_text(captcha)
    eval_image(captcha)

    # show the image
    cv2.imshow('captcha', captcha)
    cv2.waitKey(0)

