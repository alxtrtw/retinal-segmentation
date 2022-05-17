from scipy.stats import gmean
import numpy as np
from typing import Tuple


def get_statistics(image: np.ndarrray, mask: np.ndarray) -> Tuple[int, int, int, int]:
    """

    :param image:
    :param mask:
    :return:
    """
    (width, height) = image.shape
    tp, fp, fn, tn = 0, 0, 0, 0

    for x in range(width):
        for y in range(height):
            if image[x, y]:
                if mask[x, y]:
                    tp += 1
                else:
                    fp += 1
            else:
                if mask[x, y]:
                    fn += 1
                else:
                    tn += 1

    return tp, fp, fn, tn


def get_metrics(image: np.ndarrray, mask: np.ndarrray, thresh: float = 0.1) -> Tuple[float, float,
                                                                                     float, float]:
    """

    :param image:
    :param mask:
    :param thresh:
    :return:
    """
    img_bool = image > thresh
    mask_bool = mask > thresh

    tp, fp, fn, tn = get_statistics(img_bool, mask_bool)

    accuracy = (tp + tn) / (tn + fn + tp + fp)
    sensitivity = tp / (tp + fn)
    specificity = tn / (fp + tn)
    geometric = float(gmean([sensitivity, specificity])) # TODO check

    return accuracy, sensitivity, specificity, geometric


def show_statistics(stats: Tuple[float, float, float, float], fname: str):
    """

    :param stats:
    :param fname:
    :return:
    """
    stat_names = ['Accuracy', 'Sensitivity', 'Specificity', 'Geometric']
    print(f"File: {fname}")
    for i, name in enumerate(stat_names):
        print(f"{name}: {stats[i]}")