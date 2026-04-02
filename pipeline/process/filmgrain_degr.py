import numpy as np
from .utils import probability
from ..utils.registry import register_class
import logging

try:
    import torch
    from optimized.gpu_degradations import film_grain_pt

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

        if lq.ndim == 2:
            tensor = torch.from_numpy(lq[None, None]).cuda()
        else:
            tensor = torch.from_numpy(lq.transpose(2, 0, 1)[None]).cuda()

        result = film_grain_pt(tensor, intensity, grain_size, midtone)

        out = result.squeeze(0).cpu().numpy()
        if lq.ndim == 2:
            lq = out.squeeze(0).astype(np.float32)
        else:
            lq = out.transpose(1, 2, 0).astype(np.float32)
        return np.clip(lq, 0, 1), hq
