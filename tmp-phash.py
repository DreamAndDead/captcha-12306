from hash import phash
import numpy as np
import cv2


if __name__ == '__main__':
    f = './dataset/annotation/image/中国结/00016_0.jpg'

    i = cv2.imread(f)

    r = phash(i)

    print(r, r.shape)

    r = np.unpackbits(r)
    print(r, r.shape)

