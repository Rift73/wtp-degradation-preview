import numpy as np
from .utils import probability
from ..utils.registry import register_class
import logging

try:
    import torch
    from optimized.gpu_degradations import overshoot_pt, image_to_tensor, result_image

    _HAS_GPU = True
except ImportError:
    _HAS_GPU = False


@register_class("overshoot")
class Overshoot:
    """Edge overshoot/undershoot from aggressive sharpening (warp sharp).

    Args:
        overshoot_dict (dict): Configuration dictionary with keys:
            - "amount" (list[float]): Sharpening strength range.
            - "cutoff" (list[float]): Butterworth cutoff range (fraction of Nyquist).
            - "order" (list[int]): Filter order range.
            - "probability" (float): Probability of applying the effect.
    """

    def __init__(self, overshoot_dict: dict):
        self.amount = overshoot_dict.get("amount", [0.5, 2.0])
        self.cutoff = overshoot_dict.get("cutoff", [0.2, 0.5])
        self.order = overshoot_dict.get("order", [1, 3])
        self.probability = overshoot_dict.get("probability", 1.0)

    def _degrade(self, tensor: torch.Tensor) -> torch.Tensor:
        amount = float(np.random.uniform(*self.amount))
        cutoff = float(np.random.uniform(*self.cutoff))
        order = int(np.random.randint(self.order[0], self.order[1] + 1))

        logging.debug(
            f"Overshoot - amount: {amount:.2f} cutoff: {cutoff:.2f} order: {order}"
        )

        return overshoot_pt(tensor, amount, cutoff, order)

    def run_tensor(self, lq: torch.Tensor, hq: torch.Tensor) -> tuple:
        """run on 1x3xHxW CUDA tensors (the engine's GPU hand-off)."""
        if probability(self.probability):
            return lq, hq
        return self._degrade(lq), hq

    def run(self, lq: np.ndarray, hq: np.ndarray) -> tuple:
        if probability(self.probability):
            return lq, hq

        if not (_HAS_GPU and torch.cuda.is_available()):
            logging.warning("Overshoot requires CUDA — skipping")
            return lq, hq

        tensor = image_to_tensor(lq)
        return result_image(lq, tensor, self._degrade(tensor)), hq
