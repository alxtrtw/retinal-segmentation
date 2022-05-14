def image_to_array(image: PIL.Image) -> np.ndarray:
    return np.ndarray(image)


def threshold_image(image: np.ndarray) -> np.ndarray:
    _, result = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY)
    return result


def get_hu_moments(image: np.ndarray) -> cv2.Moments:
    moments = cv2.moments(image)
    hu_moments = cv2.HuMoments(moments)
    return hu_moments


def get_color_variance(image: np.ndarray) -> float:
    pass


def get_contrast(image: np.ndarray) -> float:
    contrast = img_grey.std()
    return contrast


def split_rgb_image(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    (B, G, R) = cv2.split(image.astype("float"))
    return R, G, B
