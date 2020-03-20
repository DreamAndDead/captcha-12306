# coding: utf-8
import cv2
import numpy as np
import scipy.fftpack


def avhash(im):
    im = cv2.resize(im, (8, 8), interpolation=cv2.INTER_CUBIC)
    avg = im.mean()
    im = im > avg
    im = np.packbits(im)
    return im


def phash(im):
    im = cv2.resize(im, (32, 32), interpolation=cv2.INTER_CUBIC)
    im = scipy.fftpack.dct(scipy.fftpack.dct(im, axis=0), axis=1)
    im = im[:8, :8]
    med = np.median(im)
    im = im > med
    im = np.packbits(im)
    return im


def phash_simple(im):
    im = cv2.resize(im, (32, 32), interpolation=cv2.INTER_CUBIC)
    im = scipy.fftpack.dct(im)
    im = im[:8, 1:8 + 1]
    avg = im.mean()
    im = im > avg
    im = np.packbits(im)
    return im


def dhash(im):
    im = cv2.resize(im, (8 + 1, 8), interpolation=cv2.INTER_CUBIC)
    im = im[:, 1:] > im[:, :-1]
    im = np.packbits(im)
    return im


def dhash_vertical(im):
    im = cv2.resize(im, (8, 8 + 1), interpolation=cv2.INTER_CUBIC)
    im = im[1:, :] > im[:-1, :]
    im = np.packbits(im)
    return im


def whash(image):
    """
    Wavelet Hash computation.

    based on https://www.kaggle.com/c/avito-duplicate-ads-detection/
    @image must be a PIL instance.
    """
    ll_max_level = int(np.log2(min(image.shape)))
    image_scale = 2**ll_max_level

    level = 3
    dwt_level = ll_max_level - level

    image = cv2.resize(image, (image_scale, image_scale))
    pixels = image / 255

    # Remove low level frequency LL(max_ll) if @remove_max_haar_ll using haar filter
    coeffs = pywt.wavedec2(pixels, 'haar', level = ll_max_level)
    coeffs[0][:] = 0
    pixels = pywt.waverec2(coeffs, 'haar')

    # Use LL(K) as freq, where K is log2(@hash_size)
    coeffs = pywt.wavedec2(pixels, 'haar', level = dwt_level)
    dwt_low = coeffs[0]

    # Substract median and compute hash
    med = np.median(dwt_low)
    diff = dwt_low > med

    diff = np.packbits(diff)
    return diff


def hamming(hash1, hash2):
    b1 = np.unpackbits(hash1)
    b2 = np.unpackbits(hash2)

    xor = b1 ^ b2
    return np.sum(xor)

