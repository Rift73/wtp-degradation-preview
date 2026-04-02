import random
import numpy as np
from pepeline.pepeline import fast_color_level

from .utils import probability
from ..constants import INTERPOLATION_MAP, SUBSAMPLING_MAP, YUV_MAP
from numpy.random import choice
from ..utils.registry import register_class
from chainner_ext import resize, ResizeFilter
import cv2 as cv

try:
    import torch
    from optimized.gpu_degradations import chroma_subsample_pt
    _HAS_GPU_SUBSAMPLE = True
except ImportError:
    _HAS_GPU_SUBSAMPLE = False
import logging

from ..utils.random import safe_uniform
import colour


@register_class("subsampling")
class Subsampling:
    """
    A class to perform subsampling on images with various downscaling and upscaling algorithms,
    different subsampling formats, and optional blurring.

    Attributes:
        down_alg (list): List of algorithms for downscaling.
        up_alg (list): List of algorithms for upscaling.
        format_list (list): List of subsampling formats.
        blur_kernels (list): List of blur kernel sizes for optional blurring.
        ycbcr_type (list): List of YUV types.
        probability (float): Probability of applying subsampling.
    """

    def __init__(self, sub: dict):
        """
        Initializes the Subsampling class with the provided configuration.

        Args:
            sub (dict): Configuration dictionary containing options for downscaling,
                        upscaling, subsampling format, blur kernels, YUV type, and
                        probability.
        """
        self.down_alg = sub.get("down", ["nearest"])
        self.up_alg = sub.get("up", ["nearest"])
        self.format_list = sub.get("sampling", ["4:4:4"])
        self.blur_kernels = sub.get("blur")
        self.ycbcr_type = sub.get("yuv", ["601"])
        self.probability = sub.get("probability", 1.0)

    @staticmethod
    def __down_up(
        lq: np.ndarray,
        shape: [int, int],
        scale: [float, float],
        down_alg: ResizeFilter,
        up_alg: ResizeFilter,
    ) -> np.ndarray:
        """
        Applies downscaling followed by upscaling to an image.

        Args:
            lq (np.ndarray): Low-quality input image.
            shape (tuple): Target shape of the image.
            scale (float): Scaling factor.
            down_alg (ResizeFilter): Downscaling algorithm.
            up_alg (ResizeFilter): Upscaling algorithm.

        Returns:
            np.ndarray: Image after applying downscaling and upscaling.
        """
        return fast_color_level(
            resize(
                resize(
                    lq,
                    (int(shape[1] * scale[1]), int(shape[0] * scale[0])),
                    down_alg,
                    False,
                ).squeeze(),
                (shape[1], shape[0]),
                up_alg,
                False,
            ).squeeze(),
            1,
            254,
        )

    def __sample(self, lq: np.ndarray) -> np.ndarray:
        """
        Applies subsampling to the image according to the specified format.

        Args:
            lq (np.ndarray): Low-quality input image.

        Returns:
            np.ndarray: Image after subsampling.
        """
        shape_lq = lq.shape
        down_alg = INTERPOLATION_MAP[choice(self.down_alg)]
        up_alg = INTERPOLATION_MAP[choice(self.up_alg)]
        scale_list = SUBSAMPLING_MAP[random.choice(self.format_list)]
        logging.debug(
            f"Subsampling: format - {scale_list} down_alg - {down_alg} up_alg - {up_alg}"
        )
        if scale_list != [1, 1, 1]:
            lq[..., 1:3] = self.__down_up(
                lq[..., 1:3], shape_lq, scale_list[1:3], down_alg, up_alg
            )
        return lq

    def run(self, lq: np.ndarray, hq: np.ndarray) -> (np.ndarray, np.ndarray):
        """
        Runs the subsampling process and optional blurring on the input image.

        Args:
            lq (np.ndarray): Low-quality input image.
            hq (np.ndarray): High-quality reference image.

        Returns:
            tuple: Modified low-quality image and the original high-quality image.
        """
        if lq.ndim == 2 or lq.shape[2] == 1 or probability(self.probability):
            return lq, hq

        # Sample parameters
        ycbcr_key = random.choice(self.ycbcr_type)
        down_alg_name = choice(self.down_alg)
        up_alg_name = choice(self.up_alg)
        format_str = random.choice(self.format_list)
        blur_sigma = safe_uniform(self.blur_kernels) if self.blur_kernels else None
        if blur_sigma == 0.0:
            blur_sigma = None

        # Map YUV key: "601"→"601", "709"→"709", etc.
        logging.debug(
            f"Subsampling: format={format_str} down={down_alg_name} up={up_alg_name} yuv={ycbcr_key} blur={blur_sigma}"
        )

        # GPU path
        if _HAS_GPU_SUBSAMPLE and torch.cuda.is_available():
            tensor = torch.from_numpy(lq.transpose(2, 0, 1)[None]).cuda()
            result = chroma_subsample_pt(
                tensor, down_alg_name, up_alg_name, format_str, blur_sigma, ycbcr_key
            )
            lq = result.squeeze(0).cpu().numpy().transpose(1, 2, 0)
            return np.clip(lq, 0, 1).astype(np.float32), hq

        # CPU fallback
        yuv = YUV_MAP[ycbcr_key]
        lq = colour.RGB_to_YCbCr(
            lq, in_bits=8, K=colour.models.rgb.ycbcr.WEIGHTS_YCBCR[yuv]
        ).astype(np.float32)

        lq = self.__sample(lq)
        if blur_sigma is not None:
            lq[..., 1] = cv.GaussianBlur(
                lq[..., 1], (0, 0), sigmaX=blur_sigma, sigmaY=blur_sigma,
                borderType=cv.BORDER_REFLECT,
            )
            lq[..., 2] = cv.GaussianBlur(
                lq[..., 2], (0, 0), sigmaX=blur_sigma, sigmaY=blur_sigma,
                borderType=cv.BORDER_REFLECT,
            )
        lq = (
            colour.YCbCr_to_RGB(
                lq, in_bits=8, out_bits=8,
                K=colour.models.rgb.ycbcr.WEIGHTS_YCBCR[yuv],
            )
            .astype(np.float32)
            .clip(0, 1)
        )
        return lq, hq
