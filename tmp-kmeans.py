from hash import phash
import numpy as np
import cv2
from imutils import paths
import shutil
import os
from sklearn.cluster import KMeans

if __name__ == '__main__':
    in_dir = './dataset/annotation/image/'
    out_dir = './dataset/annotation/image-cluster/'

    shutil.rmtree(out_dir)
    os.mkdir(out_dir)

    images = list(paths.list_images(in_dir))
    files = []
    X = []
    for i in images:
        img = cv2.imread(i)
        r = phash(img)
        r = np.unpackbits(r)
        X.append(r)
        files.append(i)

    X = np.array(X)
    kmeans = KMeans(n_clusters=80, verbose=1, max_iter=10000, n_init=100).fit(X)

    for l, f in zip(kmeans.labels_, files):
        dest_dir = os.path.join(out_dir, str(l))
        os.makedirs(dest_dir, exist_ok=True)

        shutil.copy(f, dest_dir)
