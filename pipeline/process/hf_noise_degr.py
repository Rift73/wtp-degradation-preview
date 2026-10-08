import numpy as np
import torch
import torch.nn.functional as F
from .utils import probability
from ..utils.registry import register_class
from optimized.gpu_degradations import image_to_tensor, tensor_to_image
import logging

try:
    from optimized.nlmeans_cuda import nlmeans_denoise_cuda
    _HAS_CUDA_NLMEANS = True
except ImportError:
    nlmeans_denoise_cuda = None
    _HAS_CUDA_NLMEANS = False

_NLMEANS_FALLBACK_LOGGED = False


def _nlmeans_gpu(
    img: np.ndarray,
    h: float = 10.0,
    template_size: int = 7,
    search_size: int = 21,
) -> np.ndarray:
    """GPU Non-Local Means denoising with CUDA kernel fallback."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = image_to_tensor(img, device)

    # The CUDA kernel handles three channels only
    if _HAS_CUDA_NLMEANS and x.is_cuda and x.shape[1] == 3 and nlmeans_denoise_cuda is not None:
        global _NLMEANS_FALLBACK_LOGGED
        try:
            result = nlmeans_denoise_cuda(x, h, template_size, search_size)
        except Exception as exc:
            if not _NLMEANS_FALLBACK_LOGGED:
                logging.warning(
                    "NLMeans CUDA kernel unavailable, using PyTorch fallback: %s",
                    exc,
                )
                _NLMEANS_FALLBACK_LOGGED = True
            result = _nlmeans_core(x, h, template_size, search_size)
    else:
        result = _nlmeans_core(x, h, template_size, search_size)

    return tensor_to_image(result, img.ndim)


@torch.no_grad()
def _nlmeans_core(
    x: torch.Tensor,
    h: float = 10.0,
    template_size: int = 7,
    search_size: int = 21,
) -> torch.Tensor:
    """PyTorch NLMeans fallback. Input: [B, C, H, W] tensor on GPU."""
    B, C, H, W = x.shape
    t_half = template_size // 2
    s_half = search_size // 2
    pad = s_half + t_half

    x_pad = F.pad(x, [pad] * 4, mode="reflect")

    box = torch.ones(1, 1, template_size, template_size, device=x.device, dtype=x.dtype)
    norm_factor = template_size * template_size * C

    h_scaled = h / 255.0
    h_sq = h_scaled * h_scaled

    weights_sum = torch.zeros(B, 1, H, W, device=x.device, dtype=x.dtype)
    output = torch.zeros_like(x)

    for dy in range(-s_half, s_half + 1):
        for dx in range(-s_half, s_half + 1):
            sy = s_half + dy
            sx = s_half + dx
            shifted = x_pad[:, :, sy : sy + H + 2 * t_half, sx : sx + W + 2 * t_half]

            center = x_pad[
                :, :, s_half : s_half + H + 2 * t_half, s_half : s_half + W + 2 * t_half
            ]

            diff_sq = (center - shifted).square().sum(dim=1, keepdim=True)
            patch_dist = F.conv2d(diff_sq, box, padding=0) / norm_factor
            w = torch.exp(-patch_dist / h_sq)

            shifted_pixel = x_pad[
                :, :, pad + dy : pad + dy + H, pad + dx : pad + dx + W
            ]

            weights_sum += w
            output += w * shifted_pixel

    return output / weights_sum


def _beta_noise_gpu(
    hq: np.ndarray, a: float, b: float, alpha: float, noise_channels: int, normalize: bool,
) -> np.ndarray:
    """Add Beta noise to HQ on the GPU.

    The CUDA RNG is seeded from numpy's stream, so the engine's per-step seed
    reproduces the noise on the same GPU; the global CUDA RNG state is restored
    afterwards. torch.distributions takes no generator, hence fork_rng.
    """
    tensor = image_to_tensor(hq)
    seed = int(np.random.randint(0, 2**31 - 1))
    with torch.random.fork_rng(devices=[tensor.device]):
        torch.cuda.manual_seed(seed)
        params = torch.tensor([a, b], dtype=torch.float32, device=tensor.device)
        noise = torch.distributions.Beta(params[0], params[1]).sample(
            (1, noise_channels, *tensor.shape[2:])
        )

    if normalize:
        noise = noise - noise.mean()
        std = noise.std(correction=0)
        if std > 1e-6:
            noise = noise / std
        noise = noise.clamp(-3, 3) * float(alpha)
    else:
        noise = (noise - 0.5) * 2 * float(alpha)

    # Grayscale noise (one channel) broadcasts over the colour channels
    return tensor_to_image(tensor + noise, hq.ndim)


def _beta_noise_numpy(
    hq: np.ndarray, a: float, b: float, alpha: float, noise_channels: int, normalize: bool,
) -> np.ndarray:
    """CPU path of _beta_noise_gpu."""
    h, w = hq.shape[:2]
    channels = hq.shape[2] if hq.ndim == 3 else 1

    # Generate noise from Beta distribution
    noise = np.random.beta(a, b, size=(h, w, noise_channels)).astype(np.float32)

    if normalize:
        noise = noise - noise.mean()
        std = noise.std()
        if std > 1e-6:
            noise = noise / std
        noise = np.clip(noise, -3, 3)
        noise = noise * alpha
    else:
        noise = (noise - 0.5) * 2 * alpha

    # Broadcast grayscale noise to all channels
    if noise_channels < channels:
        noise = np.broadcast_to(noise, (h, w, channels))

    if hq.ndim == 2:
        noise = noise.squeeze(-1)

    return np.clip(hq + noise, 0, 1).astype(np.float32)


@register_class("hf_noise")
class HFNoise:
    """Adds beta-distributed high-frequency noise to HQ, optionally denoises LQ.

    When denoise is enabled, LQ is smoothed with GPU-accelerated Non-Local Means
    so the model input is clean, while HQ gets added HF texture noise so the model
    learns to produce natural high-frequency detail.

    Args:
        config (dict): Dictionary containing:
            - "probability" (float): Probability of applying. Default 1.0.
            - "alpha" (list): [min, max] amplitude range. Default [0.01, 0.05].
            - "beta_shape" (list): [min, max] for Beta 'a' param. Default [2, 5].
            - "beta_offset" (list or None): [min, max] offset for 'b = a + offset'.
              When None, 'b' is sampled independently from beta_shape. Default None.
            - "gray_prob" (float): Probability noise is grayscale. Default 1.0.
            - "normalize" (bool): Zero-center and normalize before scaling. Default True.
            - "denoise" (bool): Whether to denoise LQ. Default False.
            - "denoise_strength" (float): NLMeans filter strength h (0-255 scale). Default 30.0.
    """

    def __init__(self, config: dict):
        self.probability = config.get("probability", 1.0)
        self.alpha_range = config.get("alpha", [0.01, 0.05])
        self.beta_shape_range = config.get("beta_shape", [2, 5])
        self.beta_offset_range = config.get("beta_offset", None)
        self.gray_prob = config.get("gray_prob", 1.0)
        self.normalize = config.get("normalize", True)
        self.denoise = config.get("denoise", False)
        self.denoise_strength = config.get("denoise_strength", 30.0)

    def run(self, lq: np.ndarray, hq: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if probability(self.probability):
            return lq, hq

        # Denoise LQ with GPU NLMeans (clean input for model)
        if self.denoise:
            lq = _nlmeans_gpu(lq, h=self.denoise_strength)

        channels = hq.shape[2] if hq.ndim == 3 else 1
        is_gray_noise = np.random.uniform() < self.gray_prob
        noise_channels = 1 if is_gray_noise else channels

        # Sample Beta distribution parameters
        a = np.random.uniform(*self.beta_shape_range)
        if self.beta_offset_range is not None:
            b = a + np.random.uniform(*self.beta_offset_range)
        else:
            b = np.random.uniform(*self.beta_shape_range)

        alpha = np.random.uniform(*self.alpha_range)

        if torch.cuda.is_available():
            hq = _beta_noise_gpu(hq, a, b, alpha, noise_channels, self.normalize)
        else:
            hq = _beta_noise_numpy(hq, a, b, alpha, noise_channels, self.normalize)

        logging.debug(
            "HF Noise - alpha: %.4f beta_a: %.2f beta_b: %.2f gray: %s normalize: %s denoise: %s strength: %.1f",
            alpha, a, b, is_gray_noise, self.normalize, self.denoise, self.denoise_strength,
        )

        return lq, hq
