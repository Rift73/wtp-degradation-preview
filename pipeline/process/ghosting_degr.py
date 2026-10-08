import numpy as np
from .utils import probability
from ..utils.registry import register_class
import logging

try:
    import torch
    from optimized.gpu_degradations import (
        temporal_ghosting_pt, image_to_tensor, result_image,
    )

    _HAS_GPU = True
except ImportError:
    _HAS_GPU = False


@register_class("ghosting")
class Ghosting:
    """Temporal ghosting from residual frame blending.

    Args:
        ghosting_dict (dict): Configuration dictionary with keys:
            - "shift_x" (list[int]): Horizontal ghost displacement range.
            - "shift_y" (list[int]): Vertical ghost displacement range.
            - "opacity" (list[float]): Ghost opacity range.
            - "probability" (float): Probability of applying the effect.
    """

    def __init__(self, ghosting_dict: dict):
        self.shift_x = ghosting_dict.get("shift_x", [1, 8])
        self.shift_y = ghosting_dict.get("shift_y", [0, 2])
        self.opacity = ghosting_dict.get("opacity", [0.05, 0.25])
        self.probability = ghosting_dict.get("probability", 1.0)

    def _degrade(self, tensor: torch.Tensor) -> torch.Tensor:
        sx = int(np.random.randint(self.shift_x[0], self.shift_x[1] + 1))
        sy = int(np.random.randint(self.shift_y[0], self.shift_y[1] + 1))
        opacity = float(np.random.uniform(*self.opacity))

        logging.debug(
            f"Ghosting - shift: ({sx}, {sy}) opacity: {opacity:.2f}"
        )

        return temporal_ghosting_pt(tensor, sx, sy, opacity)

    def run_tensor(self, lq: torch.Tensor, hq: torch.Tensor) -> tuple:
        """run on 1x3xHxW CUDA tensors (the engine's GPU hand-off)."""
        if probability(self.probability):
            return lq, hq
        return self._degrade(lq), hq

    def run(self, lq: np.ndarray, hq: np.ndarray) -> tuple:
        if probability(self.probability):
            return lq, hq
        if lq.ndim == 2:
            return lq, hq

        if not (_HAS_GPU and torch.cuda.is_available()):
            logging.warning("Ghosting requires CUDA — skipping")
            return lq, hq

        tensor = image_to_tensor(lq)
        return result_image(lq, tensor, self._degrade(tensor)), hq
