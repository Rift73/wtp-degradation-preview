"""Exact CUDA error-diffusion dithering with lazy extension build.

The sources are the owner's traiNNer fork's (the .cu adds a Windows-only
#undef): bit-identical to chainner_ext's error_diffusion_dither with
UniformQuantization, one block per channel plane. The extension also carries the fork's Riemersma kernel, which
the GUI leaves to chainner_ext: it dithers one image at a time, and a single
Hilbert walk is slower on one GPU thread than on a CPU core.

The build is cached per user (see cuda_ext); without a cached build and without
cl.exe the import fails, and callers keep chainner_ext.
"""

from __future__ import annotations

import os
import threading

import torch
from torch import Tensor

from optimized.cuda_ext import check_buildable, load_extension

_EXT_NAME = "dither_cuda_ext_wtp_gui"
_csrc_dir = os.path.join(os.path.dirname(__file__), "csrc")
_SOURCES = [
    os.path.join(_csrc_dir, "dither.cpp"),
    os.path.join(_csrc_dir, "dither_kernel.cu"),
]
# No --use_fast_math: flushing denormal errors to zero would break exactness
_CUDA_CFLAGS: list[str] = []
_ext_lock = threading.Lock()
_ext = None
_ext_error = None

check_buildable(_EXT_NAME, _SOURCES, _CUDA_CFLAGS, "chainner_ext's error diffusion")


def _load_ext():
    global _ext, _ext_error

    if _ext is not None:
        return _ext
    if _ext_error is not None:
        raise RuntimeError("Dithering CUDA extension is unavailable") from _ext_error

    with _ext_lock:
        if _ext is not None:
            return _ext
        if _ext_error is not None:
            raise RuntimeError("Dithering CUDA extension is unavailable") from _ext_error

        try:
            _ext = load_extension(_EXT_NAME, _SOURCES, _CUDA_CFLAGS)
        except Exception as exc:
            _ext_error = exc
            raise

    return _ext


def error_diffusion_dither_cuda(x: Tensor, levels: int, algorithm: int) -> Tensor:
    """Error-diffusion dither a BCHW float32 CUDA tensor with 1, 3 or 4 channels.

    Args:
        x: BCHW float32 CUDA tensor; made contiguous, as the kernel reads planes.
        levels: Quantization levels per channel (at least 2), for every image.
        algorithm: A chainner_ext.DiffusionAlgorithm value as an int; the
            kernel's tables follow that enum's order.

    Returns:
        New dithered tensor of the same shape, values in [0, 1].
    """
    if levels < 2:
        raise ValueError("The quantization level count must be at least 2.")
    ext = _load_ext()
    per_image = torch.full((x.shape[0],), levels, dtype=torch.int64, device=x.device)
    return ext.error_diffusion(x.contiguous(), per_image, algorithm)
