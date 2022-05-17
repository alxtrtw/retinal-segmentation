import numpy as np
import cv2
from PIL import Image
from typing import Tuple
from skimage.feature import hog

def image_to_array(image: Image) -> np.ndarray:
    return np.ndarray(image)


def threshold_image(image: np.ndarray) -> np.ndarray:
    _, result = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY)
    return result


def get_hu_moments(image: np.ndarray) -> cv2.moments:
    moments = cv2.moments(image)
    hu_moments = cv2.HuMoments(moments)
    return hu_moments


def get_color_variance(image: np.ndarray) -> float:
    pass


def get_contrast(image: np.ndarray) -> float:
    contrast = image.std()
    return contrast


def split_rgb_image(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    (B, G, R) = cv2.split(image.astype("float"))
    return R, G, B


def get_center_value(image: np.ndarray) -> float:  # TODO
    if image.shape[0] != image.shape[1] or image.shape[0]%2 == 0:
        raise Exception("Can't get image center value!")

    x = np.ceil(image.shape[0]/2)
    val = image[x, x]
    return val


def get_hog(image: np.ndarray) -> np.ndarray:
    fd, hog_image = hog(image, orientations=8, pixels_per_cell=(16, 16),
                        cells_per_block=(1, 1), visualize=True, channel_axis=-1)
    return hog_image


def get_features(image: np.ndarray) -> list:  # TODO

    features = get_hog(image)

    # img_gray = cv2.cvtColor(image_slice, cv2.COLOR_BGR2GRAY)
    #
    # contrast = ft_ext.get_contrast(img_gray)
    #
    # hu_moments = ft_ext.get_hu_moments(img_gray).tolist()
    #
    # features.append(contrast)
    # for moment in hu_moments:
    #     features.append(moment)
    # to raczej nic nie da w obecnej formie w jakiej jest

    return features