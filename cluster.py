import os
from imutils import paths
import cv2
from sklearn.cluster import KMeans
import shutil
import numpy as np


sift = cv2.xfeatures2d.SIFT_create()

def get_feature(img):
    (h1, w1, _) = img.shape
    (w, h) = (60, 30)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#    img = cv2.resize(img, (60, 30), interpolation=cv2.INTER_CUBIC)

 #   kp, des = sift.detectAndCompute(img, None)


    img = cv2.copyMakeBorder(img,
                             (h - h1) // 2,
                             h - h1 - (h - h1) // 2,
                             (w - w1) // 2,
                             w - w1 - (w - w1) // 2,
                             cv2.BORDER_CONSTANT,
                             value=[255, 255, 255])

    return img.reshape(1, -1).tolist()[0]


if __name__ == '__main__':
    unlabel_dir = './dataset/annotation/text/unknown/'
    # unlabel_dir = './dataset/raw/text/'
    cluster_dir = './dataset/annotation/text/cluster/'

    os.mkdir(cluster_dir)

    images = list(paths.list_images(unlabel_dir))
    features = []
    for i in images:
        img = cv2.imread(i)
        f = get_feature(img)
        features.append(f)

    k = KMeans(n_clusters=80).fit(np.array(features))

    for i, l in zip(images, k.labels_):
        c_dir = os.path.join(cluster_dir, str(l))
        if not os.path.exists(c_dir):
            os.mkdir(c_dir)
        shutil.copy(i, c_dir)
        
