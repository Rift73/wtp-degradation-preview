from .utils import probability
import random
import numpy as np
from ..utils.registry import register_class
from ..utils.random import safe_uniform
import logging


def sin_patern(
    img: np.ndarray, shape_sin: int, alpha: float, vertical: bool, bias: float
) -> np.ndarray:
    """Adds a sine wave of period `shape_sin` px and amplitude `alpha` across the image,
    shifted by `bias` px per line, clipped to [0, 1] (port of dataset_support 0.1.4's
    sin_patern; vertical=True runs the wave along each row)."""
    h, w = img.shape[:2]
    steps = np.arange(shape_sin, dtype=np.float32)
    tile = np.sin(np.float32(3.14) * steps / np.float32(shape_sin / 2.0)) * np.float32(alpha)
    if vertical:
        shift = np.trunc(np.arange(h, dtype=np.float32) * np.float32(bias)).astype(np.int64)
        index = (np.arange(w)[None, :] - shift[:, None]) % shape_sin
    else:
        shift = np.trunc(np.arange(w, dtype=np.float32) * np.float32(bias)).astype(np.int64)
        index = (shift[None, :] - np.arange(h)[:, None]) % shape_sin
    pattern = tile[index]
    if img.ndim == 3:
        pattern = pattern[:, :, None]
    return np.clip(img + pattern, 0.0, 1.0).astype(np.float32)


@register_class("sin")
class Sin:
    """Class for applying sinusoidal patterns to images.

    Args:
        sin_loss_dict (dict): A dictionary containing sinusoidal pattern settings.
            It should include the following keys:
                - "shape" (list of int, optional): Range of shape values for the sinusoidal pattern.
                    Defaults to [100, 1000, 100].
                - "alpha" (list of float, optional): Range of alpha values for the sinusoidal pattern.
                    Defaults to [0.1, 0.5].
                - "bias" (list of float, optional): Range of bias values for the sinusoidal pattern.
                    Defaults to [0.8, 1.2].
                - "vertical" (float, optional): Probability of applying vertical sinusoidal patterns.
                    Defaults to 0.5.
                - "probability" (float, optional): Probability of applying sinusoidal patterns. Defaults to 1.0.
    """

    def __init__(self, sin_loss_dict: dict):
        self.shape = sin_loss_dict.get("shape", [100, 1000, 100])
        self.alpha = sin_loss_dict.get("alpha", [0.1, 0.5])
        self.bias = sin_loss_dict.get("bias", [0.8, 1.2])
        self.vertical_prob = sin_loss_dict.get("vertical", 0.5)
        self.probability = sin_loss_dict.get("probability", 1.0)

    def run(self, lq: np.ndarray, hq: np.ndarray) -> (np.ndarray, np.ndarray):
        """Applies sinusoidal patterns to the input image.

        Args:
            lq (numpy.ndarray): The low-quality image.
            hq (numpy.ndarray): The corresponding high-quality image.

        Returns:
            tuple: A tuple containing the image with sinusoidal patterns applied and the corresponding high-quality image.
        """
        if probability(self.probability):
            return lq, hq
        shape = random.randrange(*self.shape)
        alpha = safe_uniform(self.alpha)
        vertical = not probability(self.vertical_prob)
        bias = safe_uniform(self.bias)
        logging.debug(
            f"Sin - shape: {shape} alpha: {alpha:.4f} vertical: {vertical} bias: {bias:.4f}"
        )
        lq = sin_patern(
            lq, shape_sin=shape, alpha=alpha, vertical=vertical, bias=bias
        )
        return lq.clip(0, 1), hq
