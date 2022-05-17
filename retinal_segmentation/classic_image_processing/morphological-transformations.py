import numpy as np
from numpy import asarray
from skimage.exposure import equalize_hist
from skimage.morphology import disk, square, dilation, closing
from skimage.filters import unsharp_mask
from skimage.filters.edges import convolve
from skimage.feature import canny
from skimage.color import rgb2gray
import cv2 as cv
from PIL import ImageChops


def extract_veins_sensitive(image: np.ndarray) -> np.ndarray:
    """

    :param image:
    :return:
    """
    inv = ImageChops.invert(image)
    image = rgb2gray(asarray(inv))
    # Normalize histogram
    image = equalize_hist(image)
    # Sharpen the image
    image = unsharp_mask(image, radius=6, amount=3)
    # Find vein contours
    image = dilation(canny(image, sigma=3.5))
    # Fill contours
    image = closing(image, disk(3))
    return image


def extract_veins_specific(image: np.ndarray) -> np.ndarray:
    """

    :param image:
    :return:
    """
    image = asarray(image)
    image = rgb2gray(image)

    image = equalize_hist(image)  # normalize histogram

    x = 4
    kernel = np.ones((x, x), np.uint8)
    image = cv.morphologyEx(image, cv.MORPH_OPEN, kernel)

    image = filters.sobel(image)  # sobel

    x = 8
    K = np.ones([x, x])
    K = K / sum(K)
    image = convolve(image, K)

    x = 4
    kernel = np.ones((x, x), np.uint8)
    image = cv.morphologyEx(image, cv.MORPH_OPEN, kernel)

    image = unsharp_mask(image, radius=64, amount=8)  # sharpen
    return image
