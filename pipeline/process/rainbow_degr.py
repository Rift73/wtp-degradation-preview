import numpy as np
from .utils import probability
from ..utils.registry import register_class
import logging

try:
    import torch
    from optimized.gpu_degradations import composite_rainbow_pt

    _HAS_GPU_RAINBOW = True
except ImportError:
    _HAS_GPU_RAINBOW = False


@register_class("rainbow")
class Rainbow:
    """Composite video rainbow artifact (NTSC chroma dot crawl).

    Simulates the encode/decode path of composite video (VHS, DVD composite
    output) where imperfect luma/chroma separation causes rainbow-colored
    fringes along sharp edges.

    Args:
        rainbow_dict (dict): Configuration dictionary with keys:
            - "subcarrier_freq" (list[float]): Subcarrier frequency range in cycles/pixel.
            - "chroma_bandwidth" (list[float]): Chroma demod bandwidth range in cycles/pixel.
            - "intensity" (list[float]): Effect intensity range (0-1 blend).
            - "phase_alternation" (bool): NTSC-style per-line phase flip.
            - "probability" (float): Probability of applying the effect.
    """

    def __init__(self, rainbow_dict: dict):
        self.subcarrier_freq = rainbow_dict.get("subcarrier_freq", [0.20, 0.30])
        self.chroma_bandwidth = rainbow_dict.get("chroma_bandwidth", [0.04, 0.12])
        self.intensity = rainbow_dict.get("intensity", [0.3, 1.0])
        self.phase_alternation = rainbow_dict.get("phase_alternation", True)
        self.probability = rainbow_dict.get("probability", 1.0)

    def run(self, lq: np.ndarray, hq: np.ndarray) -> tuple:
        if probability(self.probability):
            return lq, hq
        if lq.ndim == 2:
            return lq, hq

        if not (_HAS_GPU_RAINBOW and torch.cuda.is_available()):
            logging.warning("Rainbow requires CUDA — skipping")
            return lq, hq

        freq = float(np.random.uniform(*self.subcarrier_freq))
        bw = float(np.random.uniform(*self.chroma_bandwidth))
        intensity = float(np.random.uniform(*self.intensity))
        phase_offset = float(np.random.uniform(0, 2 * np.pi))

        logging.debug(
            f"Rainbow - freq: {freq:.3f} bw: {bw:.3f} "
            f"intensity: {intensity:.2f} phase: {phase_offset:.2f}"
        )

        tensor = torch.from_numpy(lq.transpose(2, 0, 1)[None]).cuda()
        result = composite_rainbow_pt(
            tensor, freq, bw, intensity,
            self.phase_alternation, phase_offset,
        )
        lq = result.squeeze(0).cpu().numpy().transpose(1, 2, 0)
        return np.clip(lq, 0, 1).astype(np.float32), hq
