import numpy as np
from chainner_ext import (
    UniformQuantization as UQ,
    quantize,
    error_diffusion_dither,
    ordered_dither,
    riemersma_dither,
)
from ..constants import DITHERING_MAP
from .utils import probability
from numpy import random
from ..utils.registry import register_class
from ..utils.random import safe_uniform, safe_randint
import logging

try:
    import torch
    from optimized.gpu_degradations import (
        quantize_pt, ordered_dither_pt, image_to_tensor, tensor_to_image,
    )
    _HAS_GPU_DITHER = True
except ImportError:
    _HAS_GPU_DITHER = False

# Exact error-diffusion kernel; absent when there is neither a cached build nor cl.exe
try:
    from optimized.dither_cuda import error_diffusion_dither_cuda, riemersma_dither_cuda
    _HAS_CUDA_DIFFUSION = True
except ImportError:
    _HAS_CUDA_DIFFUSION = False

# Dithering types that benefit from GPU (simple element-wise ops)
_GPU_DITHER_TYPES = {"quantize", "order"}

_DIFFUSION_FALLBACK_LOGGED = False


@register_class("dithering")
class Dithering:
    """Class for applying dithering algorithms to images.

    Args:
        dithering_dict (dict): A dictionary containing dithering settings.
            It should include the following keys:
                - "dithering_type" (list of str, optional): List of dithering algorithms to be used.
                    Defaults to ["quantize"].
                - "color_ch" (list of int, optional): Range of color channels for quantization.
                    Defaults to [2, 10].
                - "map_size" (list of int, optional): Range of map sizes for ordered dithering.
                    Defaults to [4, 8].
                - "history" (list of int, optional): Range of history values for Riemersma dithering.
                    Defaults to [10, 15].
                - "ratio" (list of float, optional): Range of decay ratio values for Riemersma dithering.
                    Defaults to [0.1, 0.9].
                - "probability" (float, optional): Probability of applying dithering. Defaults to 1.0.
    """

    def __init__(self, dithering_dict: dict):
        self.dithering_type_list = dithering_dict.get("dithering_type", ["quantize"])
        self.quantize = dithering_dict.get("color_ch", [2, 10])
        self.map_size = dithering_dict.get("map_size", [4, 8])
        self.history = dithering_dict.get("history", [10, 15])
        self.ratio = dithering_dict.get("ratio", [0.1, 0.9])
        self.probability = dithering_dict.get("probability", 1.0)
        self.dithering_type = "Burkes"
        self.unif_quantiz = 8

    def __error(self, lq: np.ndarray, quantization: UQ) -> np.ndarray:
        logging.debug(
            f"Dithering - type: {self.dithering_type} quantization {self.unif_quantiz}"
        )
        algorithm = DITHERING_MAP[self.dithering_type]
        # The CUDA kernel is bit-identical to chainner_ext's error diffusion
        if _HAS_GPU_DITHER and _HAS_CUDA_DIFFUSION and torch.cuda.is_available():
            global _DIFFUSION_FALLBACK_LOGGED
            try:
                result = error_diffusion_dither_cuda(
                    image_to_tensor(lq), self.unif_quantiz, int(algorithm)
                )
                return tensor_to_image(result, lq.ndim)
            except Exception as exc:
                if not _DIFFUSION_FALLBACK_LOGGED:
                    logging.warning(
                        "Error-diffusion CUDA kernel unavailable, using chainner_ext: %s",
                        exc,
                    )
                    _DIFFUSION_FALLBACK_LOGGED = True
        return error_diffusion_dither(lq, quantization, algorithm)

    def __quantize(self, lq: np.ndarray, quantization: UQ) -> np.ndarray:
        logging.debug(
            f"Dithering - type: {self.dithering_type} quantization {self.unif_quantiz}"
        )
        return quantize(lq, quantization)

    def __order(self, lq: np.ndarray, quantization: UQ) -> np.ndarray:
        map_size = random.choice(self.map_size)
        logging.debug(
            f"Dithering - type: {self.dithering_type} map_size: {map_size} quantization {self.unif_quantiz}"
        )
        return ordered_dither(lq, quantization, map_size)

    def __riemersma(self, lq: np.ndarray, quantization: UQ) -> np.ndarray:
        history = safe_randint(self.history)
        decay_ratio = safe_uniform(self.ratio)
        logging.debug(
            f"Dithering - type: {self.dithering_type} history: {history} decay_ratio: {decay_ratio:.4f} "
            f"quantization {self.unif_quantiz}"
        )
        # Same kernel as error diffusion; on Windows it is 1.1-1.6x faster than
        # chainner_ext even for one image (2048^2: 239 -> 213 ms), bit-identical.
        if _HAS_GPU_DITHER and _HAS_CUDA_DIFFUSION and torch.cuda.is_available():
            global _DIFFUSION_FALLBACK_LOGGED
            try:
                result = riemersma_dither_cuda(
                    image_to_tensor(lq), self.unif_quantiz, history, decay_ratio
                )
                return tensor_to_image(result, lq.ndim)
            except Exception as exc:
                if not _DIFFUSION_FALLBACK_LOGGED:
                    logging.warning(
                        "Riemersma CUDA kernel unavailable, using chainner_ext: %s", exc,
                    )
                    _DIFFUSION_FALLBACK_LOGGED = True
        return riemersma_dither(lq, quantization, history, decay_ratio)

    def run(self, lq: np.ndarray, hq: np.ndarray) -> (np.ndarray, np.ndarray):
        """Applies the selected dithering algorithm to the input image.

        Args:
            lq (numpy.ndarray): The low-quality image.
            hq (numpy.ndarray): The corresponding high-quality image.

        Returns:
            tuple: A tuple containing the dithered low-quality image and the corresponding high-quality image.
        """
        DITHERING_TYPE_MAP = {
            "floydsteinberg": self.__error,
            "jarvisjudiceninke": self.__error,
            "stucki": self.__error,
            "atkinson": self.__error,
            "burkes": self.__error,
            "sierra": self.__error,
            "tworowsierra": self.__error,
            "sierraLite": self.__error,
            "order": self.__order,
            "riemersma": self.__riemersma,
            "quantize": self.__quantize,
        }
        if probability(self.probability):
            return lq, hq
        self.dithering_type = random.choice(self.dithering_type_list)
        self.unif_quantiz = safe_randint(self.quantize)

        # GPU path for quantize and ordered dither
        if _HAS_GPU_DITHER and torch.cuda.is_available() and self.dithering_type in _GPU_DITHER_TYPES:
            tensor = image_to_tensor(lq)
            if self.dithering_type == "quantize":
                result = quantize_pt(tensor, self.unif_quantiz)
            else:  # "order"
                map_sz = random.choice(self.map_size) if isinstance(self.map_size, list) else self.map_size
                result = ordered_dither_pt(tensor, self.unif_quantiz, int(map_sz))

            return np.squeeze(tensor_to_image(result, lq.ndim)), hq

        # Error diffusion (CUDA kernel when available, else chainner_ext) and riemersma,
        # which stays on chainner_ext: one Hilbert walk is slower on a GPU thread
        lq = DITHERING_TYPE_MAP[self.dithering_type](lq, UQ(self.unif_quantiz))
        return np.squeeze(lq), hq
