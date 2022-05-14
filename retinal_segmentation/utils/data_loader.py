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


def load_image(path: str) -> Image:
    return Image.open(path)

