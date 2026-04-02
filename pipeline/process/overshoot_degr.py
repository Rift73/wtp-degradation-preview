import numpy as np
from .utils import probability
from ..utils.registry import register_class
import logging

try:
    import torch
    from optimized.gpu_degradations import overshoot_pt

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

    def run(self, lq: np.ndarray, hq: np.ndarray) -> tuple:
        if probability(self.probability):
            return lq, hq

        if not (_HAS_GPU and torch.cuda.is_available()):
            logging.warning("Overshoot requires CUDA — skipping")
            return lq, hq

        amount = float(np.random.uniform(*self.amount))
        cutoff = float(np.random.uniform(*self.cutoff))
        order = int(np.random.randint(self.order[0], self.order[1] + 1))

        logging.debug(
            f"Overshoot - amount: {amount:.2f} cutoff: {cutoff:.2f} order: {order}"
        )

        if lq.ndim == 2:
            tensor = torch.from_numpy(lq[None, None]).cuda()
        else:
            tensor = torch.from_numpy(lq.transpose(2, 0, 1)[None]).cuda()

        result = overshoot_pt(tensor, amount, cutoff, order)

        out = result.squeeze(0).cpu().numpy()
        if lq.ndim == 2:
            lq = out.squeeze(0).astype(np.float32)
        else:
            lq = out.transpose(1, 2, 0).astype(np.float32)
        return np.clip(lq, 0, 1), hq
