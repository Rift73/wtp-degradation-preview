import numpy as np
from .utils import probability
from ..utils.registry import register_class
import logging

try:
    import torch
    from optimized.gpu_degradations import scanline_pt, image_to_tensor, result_image

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

    def _degrade(self, tensor: torch.Tensor) -> torch.Tensor:
        strength = float(np.random.uniform(*self.strength))

        logging.debug(
            f"Scanline - strength: {strength:.2f} even: {self.even_lines}"
        )

        return scanline_pt(tensor, strength, self.even_lines)

    def run_tensor(self, lq: torch.Tensor, hq: torch.Tensor) -> tuple:
        """run on 1x3xHxW CUDA tensors (the engine's GPU hand-off)."""
        if probability(self.probability):
            return lq, hq
        return self._degrade(lq), hq

    def run(self, lq: np.ndarray, hq: np.ndarray) -> tuple:
        if probability(self.probability):
            return lq, hq

        if not (_HAS_GPU and torch.cuda.is_available()):
            logging.warning("Scanline requires CUDA — skipping")
            return lq, hq

        tensor = image_to_tensor(lq)
        return result_image(lq, tensor, self._degrade(tensor)), hq
