import numpy as np
import colour
from concurrent.futures import ThreadPoolExecutor

from .custom_blur import motion_blur
from .procedural_noise import fractal_noise
from .utils import probability, normalize_noise as normalize
from chainner_ext import resize, ResizeFilter
from ..utils.random import safe_uniform, safe_arange, safe_randint
import cv2 as cv
from ..utils.registry import register_class
import logging

# Per-pixel noise is drawn in parallel blocks, each from its own child of one seed taken from
# numpy's global RNG, so the engine's per-step np.random.seed applies. The block count is
# fixed, so the noise does not depend on the number of CPU cores.
_DRAW_BLOCKS = 16
_DRAW_POOL = ThreadPoolExecutor()


def _draw(shape: tuple, fill) -> np.ndarray:
    """float32 noise of the given shape from a numpy Generator method that fills `out`
    (Generator.standard_normal or Generator.random)."""
    noise = np.empty(shape, np.float32)
    seeds = np.random.SeedSequence(int(np.random.randint(0, 2**31 - 1))).spawn(_DRAW_BLOCKS)
    blocks = np.array_split(noise.reshape(-1), _DRAW_BLOCKS)
    list(_DRAW_POOL.map(
        lambda seed, block: fill(np.random.default_rng(seed), out=block, dtype=np.float32),
        seeds, blocks,
    ))
    return noise


def _affine(convert) -> np.ndarray:
    """The 3x4 matrix [A | b] of an affine map of 3-vectors, convert(p) = A p + b."""
    points = convert(np.vstack((np.zeros(3), np.eye(3))))
    return np.hstack(((points[1:] - points[0]).T, points[0][:, None]))


# colour's BT.2020 conversions as used here (RGB in [0, 1] to 8-bit legal-range YCbCr as
# float, and back) are affine, so cv.transform applies them as one matrix each, in float32
# instead of colour's float64 intermediates
_BT2020 = colour.models.rgb.ycbcr.WEIGHTS_YCBCR["ITU-R BT.2020"]
_RGB_TO_YCBCR = _affine(lambda rgb: colour.RGB_to_YCbCr(rgb, in_bits=8, K=_BT2020))
_YCBCR_TO_RGB = _affine(
    lambda yuv: colour.YCbCr_to_RGB(yuv, in_bits=8, out_bits=8, K=_BT2020)
)


@register_class("noise")
class Noise:
    """Class for adding noise to images.

    Args:
        noise_dict (dict): A dictionary containing noise settings.
            It should include the following keys:
                - "probability" (float, optional): Probability of adding noise. Defaults to 1.0.
                - "type_noise" (list of str, optional): List of noise types to choose from.
                    Defaults to ["uniform"].
                - "alpha" (list of int, optional): Range of alpha values for noise intensity.
                    Defaults to [1, 2, 1].
                - "lqhq" (bool, optional): Flag indicating if the low-quality image should be replaced by the noisy one.
                    Defaults to False.
                - "y_noise" (bool, optional): Flag indicating if noise should be added
                to the Y channel only (for YUV images).
                    Defaults to None.
                - "uv_noise" (bool, optional): Flag indicating if noise should be added
                to the UV channels only (for YUV images).
                    Defaults to None.
                - "normalize" (bool, optional): Flag indicating if the generated noise should be normalized.
                Defaults to None.
                - "octaves" (list of int, optional): Range of octaves for procedural noises.
                    Defaults to [1, 2, 1].
                - "frequency" (list of float, optional): Range of frequencies for procedural noises.
                    Defaults to [0.8, 0.9, 0.9].
                - "lacunarity" (list of float, optional): Range of lacunarity values for procedural noises.
                    Defaults to [0.4, 0.5, 0.5].
                - "probability_salt_or_pepper" (list of float, optional): Range of probabilities for salt-and-pepper noise.
                    Defaults to [0, 0.5].
    """

    def __init__(self, noise_dict: dict):
        # common
        self.probability = noise_dict.get("probability", 1.0)
        self.type_noise = noise_dict.get("type_noise", ["uniform"])
        alpha_rand = noise_dict.get("alpha", [1, 2, 1])
        self.alpha_rand = safe_arange(alpha_rand)
        self.lqhq = noise_dict.get("lqhq", False)
        self.y_noise = noise_dict.get("y_noise", 0)
        self.uv_noise = noise_dict.get("uv_noise", 0)
        self.noise_type = "perlin"

        # procedural_noises
        self.normalize_noise = noise_dict.get("normalize")
        octaves_range = noise_dict.get("octaves", [1, 2, 1])
        self.octaves_rand = safe_arange(octaves_range)
        frequency_range = noise_dict.get("frequency", [0.8, 0.9, 0.9])
        self.frequency_rand = safe_arange(frequency_range)
        lacunarity_range = noise_dict.get("lacunarity", [0.4, 0.5, 0.5])
        self.lacunarity_rand = safe_arange(lacunarity_range)
        scale = noise_dict.get("scale")
        self.scale_probability = 0.0
        if scale:
            scale = scale[0]
            self.scale_size = scale.get("size", [1, 2])
            self.scale_sigma = scale.get("sigma", [1, 2])
            self.scale_amount = scale.get("amount", [1, 3])
            self.scale_probability = scale.get("probability", 1)
        self.noise_clip = noise_dict.get("clip")
        if self.noise_clip:
            self.black_clip = self.noise_clip[0]
            self.white_clip = self.noise_clip[1]
        self.scale_cof = 1.0
        motion = noise_dict.get("motion")
        self.motion_probability = 0.0
        if motion:
            motion = motion[0]
            self.size = motion.get("size", [0, 3])
            self.angle = motion.get("angle", [0, 360])
            self.sigma = motion.get("sigma", [0, 1])
            self.amount = motion.get("amount", [0, 1])
            self.motion_probability = motion.get("probability", 1)
        self.bias = noise_dict.get("bias", [0, 0])
        # salt_or_pepper
        self.percentage_salt_or_pepper = noise_dict.get(
            "probability_salt_or_pepper", [0, 0.5]
        )
        self.default_debug = "Noise - color_type: gray"

    def motion(self, noise):
        noise = motion_blur(noise, safe_randint(self.size), safe_randint(self.angle))
        sigma = safe_uniform(self.sigma)
        amount = safe_uniform(self.amount)
        blurred = cv.GaussianBlur(
            noise, (0, 0), sigmaX=sigma, sigmaY=sigma, borderType=cv.BORDER_REFLECT
        )

        return cv.addWeighted(noise, amount + 1, blurred, -amount, 0)

    def noise_scale(self, noise: np.ndarray) -> np.ndarray:
        shape = noise.shape
        self.scale_cof = safe_uniform(self.scale_size)
        noise = normalize(
            resize(
                noise * 0.5 + 0.5,
                (int(shape[1] * self.scale_cof), int(shape[0] * self.scale_cof)),
                ResizeFilter.Lanczos,
                False,
            ).squeeze()[: shape[0], : shape[1]]
        )
        sigma = safe_uniform(self.scale_sigma)
        amount = safe_uniform(self.scale_amount)
        blurred = cv.GaussianBlur(
            noise, (0, 0), sigmaX=sigma, sigmaY=sigma, borderType=cv.BORDER_REFLECT
        )

        return cv.addWeighted(noise, amount + 1, blurred, -amount, 0)

    def __noise_clip(self, noise: np.ndarray, img: np.ndarray) -> np.ndarray:
        black_noise_mask = img > self.black_clip
        white_noise_mask = img < self.white_clip
        noise_mask = black_noise_mask & white_noise_mask
        return np.where(noise_mask, noise, 0)

    def __procedural_noises(self, lq: np.ndarray) -> np.ndarray:
        octaves = int(np.random.choice(self.octaves_rand))
        frequency = float(np.random.choice(self.frequency_rand))
        lacunarity = float(np.random.choice(self.lacunarity_rand))
        # Seeded from numpy's global RNG, so the engine's per-step np.random.seed applies
        noise = fractal_noise(
            lq.shape,
            self.noise_type,
            octaves,
            frequency,
            lacunarity,
            seed=int(np.random.randint(0, 2**31 - 1)),
        )
        if self.normalize_noise:
            noise = normalize(noise)
        if not probability(self.motion_probability):
            noise = self.motion(noise)
        bias = 0
        if self.bias != [0, 0]:
            bias = safe_uniform(self.bias)
            noise += bias
            np.clip(noise, -1, 1, out=noise)
        alpha = np.random.choice(self.alpha_rand)
        noise *= float(alpha)
        logging.debug(
            "%s noise_type: %s alpha: %.4f bias: %.4f octaves: %s frequency: %.4f lacunarity: %.4f",
            self.default_debug,
            self.noise_type,
            alpha,
            bias,
            octaves,
            frequency,
            lacunarity,
        )
        if self.noise_clip:
            noise = self.__noise_clip(noise, lq)
        noise += lq
        return np.clip(noise, 0, 1, out=noise)

    def __gauss(self, lq: np.ndarray) -> np.ndarray:
        noise = _draw(lq.shape, np.random.Generator.standard_normal)
        noise *= 0.25
        if not probability(self.motion_probability):
            noise = self.motion(noise)
        if not probability(self.scale_probability):
            noise = self.noise_scale(noise)
        bias = 0
        if self.bias != [0, 0]:
            bias = safe_uniform(self.bias)
            noise += bias
            np.clip(noise, -1, 1, out=noise)
        alpha = np.random.choice(self.alpha_rand)
        noise *= float(alpha)
        logging.debug(
            "%s noise_type: %s alpha: %.4f bias: %.4f scale: %.4f",
            self.default_debug,
            self.noise_type,
            alpha,
            bias,
            self.scale_cof,
        )
        if self.noise_clip:
            noise = self.__noise_clip(noise, lq)
        noise += lq
        return noise

    def __uniform_noise(self, lq: np.ndarray) -> np.ndarray:
        noise = _draw(lq.shape, np.random.Generator.random)
        noise *= 2
        noise -= 1
        if not probability(self.motion_probability):
            noise = self.motion(noise)
        if not probability(self.scale_probability):
            noise = self.noise_scale(noise)
        bias = 0
        if self.bias != [0, 0]:
            bias = safe_uniform(self.bias)
            noise += bias
            np.clip(noise, -1, 1, out=noise)
        alpha = np.random.choice(self.alpha_rand)
        noise *= float(alpha)
        logging.debug(
            "%s noise_type: %s alpha: %.4f bias: %.4f scale: %.4f",
            self.default_debug,
            self.noise_type,
            alpha,
            bias,
            self.scale_cof,
        )
        if self.noise_clip:
            noise = self.__noise_clip(noise, lq)
        noise += lq
        return noise

    def __salt_and_pepper_core(self, img_shape: tuple) -> (np.ndarray, float):
        noise = np.random.uniform(0, 1, img_shape)
        probability_sp = safe_uniform(self.percentage_salt_or_pepper)
        return noise, probability_sp

    def __salt_and_pepper(self, lq: np.ndarray) -> np.ndarray:
        noise, probability_sp = self.__salt_and_pepper_core(lq.shape)
        logging.debug(
            "%s noise_type: %s probability: %.4f",
            self.default_debug,
            self.noise_type,
            probability_sp,
        )
        lq = np.where(noise > probability_sp / 2, lq, 1)
        return np.where(noise < 1 - probability_sp / 2, lq, 0).astype(np.float32)

    def __salt(self, lq: np.ndarray) -> np.ndarray:
        noise, probability_sp = self.__salt_and_pepper_core(lq.shape)
        logging.debug(
            "%s noise_type: %s probability: %.4f",
            self.default_debug,
            self.noise_type,
            probability_sp,
        )
        if self.noise_clip:
            noise = self.__noise_clip(noise, lq)
        return np.where(noise > probability_sp, lq, 1).astype(np.float32)

    def __pepper(self, lq: np.ndarray) -> np.ndarray:
        noise, probability_sp = self.__salt_and_pepper_core(lq.shape)
        logging.debug(
            "%s noise_type: %s probability: %.4f",
            self.default_debug,
            self.noise_type,
            probability_sp,
        )
        if self.noise_clip:
            noise = self.__noise_clip(noise, lq)
        return np.where(noise < 1 - probability_sp, lq, 0).astype(np.float32)

    def run(self, lq: np.ndarray, hq: np.ndarray) -> (np.ndarray, np.ndarray):
        """Adds noise to the input image.

        Args:
            lq (numpy.ndarray): The low-quality image.
            hq (numpy.ndarray): The corresponding high-quality image.

        Returns:
            tuple: A tuple containing the noisy low-quality image and the corresponding high-quality image.
        """
        NOISE_TYPE_MAP = {
            "perlin": self.__procedural_noises,
            "simplex": self.__procedural_noises,
            "opensimplex": self.__procedural_noises,
            "supersimplex": self.__procedural_noises,
            "uniform": self.__uniform_noise,
            "gauss": self.__gauss,
            "salt": self.__salt,
            "pepper": self.__pepper,
            "salt_and_pepper": self.__salt_and_pepper,
        }
        if probability(self.probability):
            return lq, hq
        # The YCbCr channels that get the noise (Y, or Cb and Cr), None for RGB
        channels = None
        if lq.ndim == 3:
            if not probability(self.y_noise):
                channels = 0
                self.default_debug = "Noise - color_type: y"
            elif not probability(self.uv_noise):
                channels = slice(1, None)
                self.default_debug = "Noise - color_type: uv"
            else:
                self.default_debug = "Noise - color_type: rgb"
        if channels is not None:
            yuv_img = cv.transform(lq, _RGB_TO_YCBCR)
            lq = yuv_img[:, :, channels]

        self.noise_type = np.random.choice(self.type_noise)
        lq = NOISE_TYPE_MAP[self.noise_type](lq)
        if channels is not None:
            yuv_img[:, :, channels] = lq
            lq = cv.transform(yuv_img, _YCBCR_TO_RGB)
        # Every noise type returns a new array, so it is clipped in place
        np.clip(lq, 0, 1, out=lq)

        if self.lqhq:
            hq = lq
        return lq, hq
