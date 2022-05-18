import numpy as np
import cv2
import os
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from typing import Tuple

from retinal_segmentation.utils.data_loader import get_image_paths, load_image
from retinal_segmentation.basic_level_classifier import \
    feature_extraction as ft_ext


def slice_image(image: np.ndarray, slice_size: int) -> list:
    width, height = image.shape[0], image.shape[1]

    slices = []
    for i in range(height - slice_size + 1):
        for j in range(width - slice_size + 1):
            slices.append(image[i:i+slice_size, j:j+slice_size])

    return slices


def slice_image_2(image: np.ndarray, slice_size: int) -> list:
    width, height = image.shape[0], image.shape[1]

    new_width, new_height = width + slice_size * 2, height + slice_size * 2

    if len(image.shape) == 3:
        new_img = np.zeros((new_width, new_height, 3))
    else:
        new_img = np.zeros((new_width, new_height))

    new_img[slice_size:-slice_size, slice_size:-slice_size] = image

    slices = []
    for i in range(slice_size - 1, height + 1):
        for j in range(slice_size - 1, width + 1):
            slices.append(image[i:i + slice_size, j:j + slice_size])

    return slices


def get_list_slices(image_list: list, slice_size) -> list:
    slices = []
    for image in image_list:
        image_slices = slice_image(image, slice_size)
        # image_slices = slice_image_2(image, slice_size)
        for img in image_slices:
            slices.append(img)
    return slices


def load_sliced_images(
        data_dir: str,
        target_dir: str,
        img_shape: Tuple[int, int] = (256, 256),
        slice_size: int = 9)\
        -> Tuple[list, list]:

    data_paths, target_paths = get_image_paths(data_dir), get_image_paths(target_dir)

    data_images = \
        [cv2.resize(load_image(path, as_arr=True), img_shape) for path in data_paths]
    data_slices = get_list_slices(data_images, slice_size)

    target_images = \
        [cv2.resize(load_image(path, as_arr=True), img_shape) for path in target_paths]
    target_slices = get_list_slices(target_images, slice_size)

    return data_slices, target_slices


def get_input_features(data_images: list) -> np.ndarray:
    features = [ft_ext.get_features(image) for image in data_images]
    # features = np.stack(np.asarray(features), axis=1)[0]
    features = np.stack(np.asarray(features), axis=0)
    # features = np.array(features)
    # features.reshape(len(data_images), features.shape[2])
    return features


def get_target_vals(target_images: list) -> list:
    vals = [ft_ext.get_center_value(image) for image in target_images]
    return vals


def init_knn_classifier(train_data, train_target) \
        -> KNeighborsClassifier:

    classifier = KNeighborsClassifier(n_neighbors=4)
    classifier.fit(train_data, train_target)

    return classifier



# matrix = np.array(np.random.randint(0, 100 + 1, 100)).reshape((10, 10))
# print(matrix)
# slice = slice_image(matrix, 5)
# print(slice)