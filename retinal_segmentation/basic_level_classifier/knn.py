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
    width, height = image.shape

    slices = []
    for i in range(height - slice_size + 1):
        for j in range(width - slice_size + 1):
            slices.append(image[i:i+slice_size, j:j+slice_size])

    return slices


def get_list_slices(image_list: list, slice_size) -> list:
    slices = []
    for image in image_list:
        image_slices = slice_image(image, slice_size)
        for img in image_slices:
            slices.append(img)
    return slices


def load_sliced_images(
        data_dir: str,
        target_dir: str,
        img_shape: Tuple[int, int] = (200, 200),
        slice_size: int = 9)\
        -> Tuple[list, list]:

    data_paths, target_paths = get_image_paths(data_dir), get_image_paths(target_dir)

    data_images = [cv2.resize(load_image(path), img_shape) for path in data_paths]
    data_slices = get_list_slices(data_images, slice_size)

    target_images = [cv2.resize(load_image(path), img_shape) for path in target_paths]
    target_slices = get_list_slices(target_images, slice_size)

    return data_slices, target_slices


def get_input_features(data_images: list) -> list:  # TODO
    features = [ft_ext.get_features(image) for image in data_images]
    return features


def get_target_vals(target_images: list) -> list:  # TODO
    vals = [ft_ext.get_center_value(image) for image in target_images]
    return vals


def init_knn_classifier(train_data, train_target) \
        -> KNeighborsClassifier:  # TODO

    classifier = KNeighborsClassifier(n_neighbors=2)
    classifier.fit(train_data, train_target)

    return classifier



# matrix = np.array(np.random.randint(0, 100 + 1, 100)).reshape((10, 10))
# print(matrix)
# slice = slice_image(matrix, 5)
# print(slice)