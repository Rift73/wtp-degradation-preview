import numpy as np
from pepeline import cvt_color, CVTColor
import cv2 as cv


def probability(prob: float) -> bool:
    if prob > np.random.uniform(0, 1):
        return False
    else:
        return True


def __max_min(img: np.ndarray) -> (int, int):
    maximum = np.max(img)
    minimal = np.min(img)
    return maximum, minimal


def normalize(img: np.ndarray) -> np.ndarray:
    maximum, minimum = __max_min(img)
    return (img - minimum) * 1 / (maximum - minimum)


def normalize_noise(img: np.ndarray) -> np.ndarray:
    maximum, minimum = __max_min(img)
    return (img - minimum) * (1 + 1) / (maximum - minimum) - 1


def gray_or_color(img: np.ndarray, threshold: float) -> bool:
    """True when the means of the R, G and B channels all lie within `threshold` of each
    other (port of dataset_support 0.1.4's gray_or_color)."""
    means = img[:, :, :3].mean(axis=(0, 1))
    return float(means.max() - means.min()) <= threshold


def color_or_gray(img: np.ndarray) -> np.ndarray:
    if gray_or_color(img, 0.0003):
        return img2gray(img)
    return img


def img2gray(img: np.ndarray) -> np.ndarray:
    if img.ndim != 2 and img.shape[2] != 1:
        return cvt_color(img, CVTColor.RGB2Gray_2020)
    else:
        return img


def lq_hq2grays(lq: np.ndarray, hq: np.ndarray) -> (np.ndarray, np.ndarray):
    return img2gray(lq), img2gray(hq)


def laplace_filter(img: np.ndarray, mean_min: float) -> bool:
    gray_img = img2gray(img)
    laplace_img = cv.Laplacian(gray_img, -1)
    return np.mean(np.abs(laplace_img)) < mean_min
