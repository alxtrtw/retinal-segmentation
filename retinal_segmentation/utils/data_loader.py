import os
import numpy as np
from typing import List
from PIL import Image


def get_image_paths(directory_path: str) -> List:
    paths = []
    for filename in os.listdir(directory_path):
        if filename.endswith(".ppm"):
            paths.append(os.path.join(directory_path, filename))

    return sorted(paths)


def load_image(path: str, as_arr: bool = False) -> Image:
    img = Image.open(path)
    if as_arr:
        return np.asarray(img)
    else:
        return img


