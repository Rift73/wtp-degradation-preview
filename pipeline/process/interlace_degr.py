import numpy as np
from .utils import probability
from ..utils.registry import register_class
import logging

try:
    import torch
    from optimized.gpu_degradations import interlace_pt

    _HAS_GPU = True
except ImportError:
    _HAS_GPU = False


@register_class("interlace")
class Interlace:
    """Interlaced video combing artifact (480i/576i).

    Args:
        interlace_dict (dict): Configuration dictionary with keys:
            - "field_shift" (list[int]): Pixel shift range between fields.
            - "dominant_field" (list[str]): Field dominance options.
            - "probability" (float): Probability of applying the effect.
    """

    def __init__(self, interlace_dict: dict):
        self.field_shift = interlace_dict.get("field_shift", [1, 5])
        self.dominant_field = interlace_dict.get("dominant_field", ["top", "bottom"])
        self.probability = interlace_dict.get("probability", 1.0)

    def run(self, lq: np.ndarray, hq: np.ndarray) -> tuple:
        try:
            if probability(self.probability):
                return lq, hq
            if lq.ndim == 2:
                return lq, hq

            if not (_HAS_GPU and torch.cuda.is_available()):
                logging.warning("Interlace requires CUDA — skipping")
                return lq, hq

            shift = int(np.random.randint(self.field_shift[0], self.field_shift[1] + 1))
            field = np.random.choice(self.dominant_field)

            logging.debug(f"Interlace - shift: {shift} field: {field}")

            tensor = torch.from_numpy(lq.transpose(2, 0, 1)[None]).cuda()
            result = interlace_pt(tensor, shift, field)
            lq = result.squeeze(0).cpu().numpy().transpose(1, 2, 0)
            return np.clip(lq, 0, 1).astype(np.float32), hq
        except Exception as e:
            logging.error(f"Interlace error: {e}")
            return lq, hq
