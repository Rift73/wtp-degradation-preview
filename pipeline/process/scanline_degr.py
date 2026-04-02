import numpy as np
from .utils import probability
from ..utils.registry import register_class
import logging

try:
    import torch
    from optimized.gpu_degradations import scanline_pt

    _HAS_GPU = True
except ImportError:
    _HAS_GPU = False


@register_class("scanline")
class Scanline:
    """CRT scanline darkening.

    Args:
        scanline_dict (dict): Configuration dictionary with keys:
            - "strength" (list[float]): Darkening strength range.
            - "even_lines" (bool): Whether to darken even or odd rows.
            - "probability" (float): Probability of applying the effect.
    """

    def __init__(self, scanline_dict: dict):
        self.strength = scanline_dict.get("strength", [0.1, 0.5])
        self.even_lines = scanline_dict.get("even_lines", True)
        self.probability = scanline_dict.get("probability", 1.0)

    def run(self, lq: np.ndarray, hq: np.ndarray) -> tuple:
        if probability(self.probability):
            return lq, hq

        if not (_HAS_GPU and torch.cuda.is_available()):
            logging.warning("Scanline requires CUDA — skipping")
            return lq, hq

        strength = float(np.random.uniform(*self.strength))

        logging.debug(
            f"Scanline - strength: {strength:.2f} even: {self.even_lines}"
        )

        if lq.ndim == 2:
            tensor = torch.from_numpy(lq[None, None]).cuda()
        else:
            tensor = torch.from_numpy(lq.transpose(2, 0, 1)[None]).cuda()

        result = scanline_pt(tensor, strength, self.even_lines)

        out = result.squeeze(0).cpu().numpy()
        if lq.ndim == 2:
            lq = out.squeeze(0).astype(np.float32)
        else:
            lq = out.transpose(1, 2, 0).astype(np.float32)
        return np.clip(lq, 0, 1), hq
