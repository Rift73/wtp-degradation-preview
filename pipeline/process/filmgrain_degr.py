import numpy as np
from .utils import probability
from ..utils.registry import register_class
import logging

try:
    import torch
    from optimized.gpu_degradations import film_grain_pt, image_to_tensor, tensor_to_image

    _HAS_GPU = True
except ImportError:
    _HAS_GPU = False


@register_class("filmgrain")
class FilmGrain:
    """Luminance-dependent bandpass-filtered film grain.

    Args:
        filmgrain_dict (dict): Configuration dictionary with keys:
            - "intensity" (list[float]): Grain strength range.
            - "grain_size" (list[float]): Grain spatial scale range.
            - "midtone_bias" (list[float]): Luminance modulation range.
            - "probability" (float): Probability of applying the effect.
    """

    def __init__(self, filmgrain_dict: dict):
        self.intensity = filmgrain_dict.get("intensity", [0.02, 0.08])
        self.grain_size = filmgrain_dict.get("grain_size", [1.0, 3.0])
        self.midtone_bias = filmgrain_dict.get("midtone_bias", [0.5, 1.0])
        self.probability = filmgrain_dict.get("probability", 1.0)

    def run(self, lq: np.ndarray, hq: np.ndarray) -> tuple:
        if probability(self.probability):
            return lq, hq

        if not (_HAS_GPU and torch.cuda.is_available()):
            logging.warning("FilmGrain requires CUDA — skipping")
            return lq, hq

        intensity = float(np.random.uniform(*self.intensity))
        grain_size = float(np.random.uniform(*self.grain_size))
        midtone = float(np.random.uniform(*self.midtone_bias))

        logging.debug(
            f"FilmGrain - intensity: {intensity:.3f} size: {grain_size:.1f} "
            f"midtone: {midtone:.2f}"
        )

        # film_grain_pt reads luma from channels 0-2, so grayscale goes in as
        # three equal channels (luma == gray) and channel 0 comes back out.
        tensor = image_to_tensor(lq)
        if lq.ndim == 2:
            tensor = tensor.repeat(1, 3, 1, 1)

        result = film_grain_pt(tensor, intensity, grain_size, midtone)
        return tensor_to_image(result, lq.ndim), hq
