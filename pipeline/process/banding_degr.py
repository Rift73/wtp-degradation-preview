import numpy as np
from .utils import probability
from ..utils.registry import register_class
import logging

try:
    import torch
    from optimized.gpu_degradations import color_banding_pt

    _HAS_GPU = True
except ImportError:
    _HAS_GPU = False


@register_class("banding")
class Banding:
    """Color banding / bit depth reduction.

    Args:
        banding_dict (dict): Configuration dictionary with keys:
            - "bits" (list[int]): Bit depth range.
            - "broadcast_range" (bool): Apply broadcast limited range.
            - "probability" (float): Probability of applying the effect.
    """

    def __init__(self, banding_dict: dict):
        self.bits = banding_dict.get("bits", [4, 7])
        self.broadcast_range = banding_dict.get("broadcast_range", False)
        self.probability = banding_dict.get("probability", 1.0)

    def run(self, lq: np.ndarray, hq: np.ndarray) -> tuple:
        try:
            if probability(self.probability):
                return lq, hq

            if not (_HAS_GPU and torch.cuda.is_available()):
                logging.warning("Banding requires CUDA — skipping")
                return lq, hq

            bits = int(np.random.randint(self.bits[0], self.bits[1] + 1))

            logging.debug(
                f"Banding - bits: {bits} broadcast: {self.broadcast_range}"
            )

            if lq.ndim == 2:
                tensor = torch.from_numpy(lq[None, None]).cuda()
            else:
                tensor = torch.from_numpy(lq.transpose(2, 0, 1)[None]).cuda()

            result = color_banding_pt(tensor, bits, self.broadcast_range)

            out = result.squeeze(0).cpu().numpy()
            if lq.ndim == 2:
                lq = out.squeeze(0).astype(np.float32)
            else:
                lq = out.transpose(1, 2, 0).astype(np.float32)
            return np.clip(lq, 0, 1), hq
        except Exception as e:
            logging.error(f"Banding error: {e}")
            return lq, hq
