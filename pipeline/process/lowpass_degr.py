import numpy as np
from .utils import probability
from ..utils.registry import register_class
import logging

try:
    import torch
    from optimized.gpu_degradations import lowpass_filter_pt, detail_mask_neo_pt

    _HAS_GPU_LOWPASS = True
except ImportError:
    _HAS_GPU_LOWPASS = False


@register_class("lowpass")
class Lowpass:
    """Butterworth lowpass filter (frequency-domain bandwidth limitation).

    Simulates production/mastering lowpass filtering common in anime sources
    and broadcast video. Produces detail loss with ringing (Gibbs phenomenon)
    at edges when the filter order is high.

    Args:
        lowpass_dict (dict): Configuration dictionary with keys:
            - "cutoff" (list[float]): Cutoff frequency range as fraction of Nyquist (0-1).
            - "order" (list[int]): Butterworth filter order range.
            - "probability" (float): Probability of applying the effect.
    """

    def __init__(self, lowpass_dict: dict):
        self.cutoff = lowpass_dict.get("cutoff", [0.3, 0.8])
        self.order = lowpass_dict.get("order", [1, 5])
        self.probability = lowpass_dict.get("probability", 1.0)

        # Detail mask settings (protect edges/detail from lowpass)
        self.detail_mask = lowpass_dict.get("detail_mask", False)
        self.mask_lines_brz = lowpass_dict.get("mask_lines_brz", 0.08)

    def run(self, lq: np.ndarray, hq: np.ndarray) -> tuple:
        try:
            if probability(self.probability):
                return lq, hq

            if not (_HAS_GPU_LOWPASS and torch.cuda.is_available()):
                logging.warning("Lowpass requires CUDA — skipping")
                return lq, hq

            cutoff = float(np.random.uniform(*self.cutoff))
            order = int(np.random.randint(self.order[0], self.order[1] + 1))

            logging.debug(f"Lowpass - cutoff: {cutoff:.3f} order: {order}")

            # Keep original for detail mask blending
            original_lq = lq.copy() if self.detail_mask else None

            if lq.ndim == 2:
                tensor = torch.from_numpy(lq[None, None]).cuda()
            else:
                tensor = torch.from_numpy(lq.transpose(2, 0, 1)[None]).cuda()

            result = lowpass_filter_pt(tensor, cutoff, order)

            out = result.squeeze(0).cpu().numpy()
            if lq.ndim == 2:
                filtered = out.squeeze(0).astype(np.float32)
            else:
                filtered = out.transpose(1, 2, 0).astype(np.float32)
            filtered = np.clip(filtered, 0, 1)

            # Detail mask: protect edges/detail, only lowpass flat areas
            if self.detail_mask and original_lq is not None:
                # Compute mask from HQ using PyTorch detail_mask_neo
                if hq.ndim == 2:
                    hq_tensor = torch.from_numpy(np.repeat(hq[None, None], 3, axis=1)).cuda()
                else:
                    hq_tensor = torch.from_numpy(hq.transpose(2, 0, 1)[None]).cuda()

                mask_tensor = detail_mask_neo_pt(
                    hq_tensor, lines_brz=self.mask_lines_brz,
                )  # (1, 1, H, W)

                mask = mask_tensor.squeeze(0).squeeze(0).cpu().numpy()  # (H, W)
                if original_lq.ndim == 3:
                    mask = mask[:, :, None]
                filtered = mask * original_lq + (1.0 - mask) * filtered

                logging.debug(
                    "Lowpass detail mask - coverage: %.1f%%",
                    float(np.mean(mask > 0)) * 100,
                )

            return filtered, hq
        except Exception as e:
            logging.error(f"Lowpass error: {e}")
            return lq, hq
