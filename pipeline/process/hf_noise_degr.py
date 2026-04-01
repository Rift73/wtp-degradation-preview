import numpy as np
import torch
import torch.nn.functional as F
from .utils import probability
from ..utils.registry import register_class
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

    # [H, W, C] -> [1, C, H, W]
    if img.ndim == 2:
        x = torch.from_numpy(img[None, None]).to(device=device, dtype=torch.float32)
    else:
        x = torch.from_numpy(img.transpose(2, 0, 1)[None]).to(device=device, dtype=torch.float32)

    if _HAS_CUDA_NLMEANS and x.is_cuda and nlmeans_denoise_cuda is not None:
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

    # Back to numpy [H, W, C]
    result_np = result.squeeze(0).cpu().numpy()
    if img.ndim == 2:
        return result_np.squeeze(0).astype(np.float32)
    return result_np.transpose(1, 2, 0).astype(np.float32)


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

        h, w = hq.shape[:2]
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

        # Generate noise from Beta distribution
        noise = np.random.beta(a, b, size=(h, w, noise_channels)).astype(np.float32)

        if self.normalize:
            noise = noise - noise.mean()
            std = noise.std()
            if std > 1e-6:
                noise = noise / std
            noise = np.clip(noise, -3, 3)
            noise = noise * alpha
        else:
            noise = (noise - 0.5) * 2 * alpha

        # Broadcast grayscale noise to all channels
        if is_gray_noise and channels > 1:
            noise = np.broadcast_to(noise, (h, w, channels))

        if hq.ndim == 2:
            noise = noise.squeeze(-1)

        hq = np.clip(hq + noise, 0, 1).astype(np.float32)

        logging.debug(
            "HF Noise - alpha: %.4f beta_a: %.2f beta_b: %.2f gray: %s normalize: %s denoise: %s strength: %.1f",
            alpha, a, b, is_gray_noise, self.normalize, self.denoise, self.denoise_strength,
        )

        return lq, hq
