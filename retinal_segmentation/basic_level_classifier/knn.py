import numpy as np
import cv2
from typing import Tuple


from retinal_segmentation.basic_level_classifier import \
    feature_extraction as ft_ext


def slice_image(image: np.ndarray, size: int) -> list:
    width, height = image.shape

    slices = []
    for i in range(height - size + 1):
        for j in range(width - size + 1):
            slices.append(image[i:i+size, j:j+size])

    return slices


def get_slice_features(image_slice: np.ndarray) -> list:

    img_gray = cv2.cvtColor(image_slice, cv2.COLOR_BGR2GRAY)

    contrast = ft_ext.get_contrast(img_gray)

    hu_moments = ft_ext.get_hu_moments(img_gray)

    pass




# matrix = np.array(np.random.randint(0, 100 + 1, 100)).reshape((10, 10))
# print(matrix)
# slice = slice_image(matrix, 5)
# print(slice)