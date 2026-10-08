"""Lazy-loading CUDA IIR trailing filter (tape trailing effect).

JIT-compiles on first use into the per-user build cache (see cuda_ext).
Follows the same pattern as nlmeans_cuda.py.
"""

from __future__ import annotations

import os
import threading

import torch

from optimized.cuda_ext import check_buildable, load_extension

_EXT_NAME = "iir_trailing_cuda_ext_wtp_gui"
_csrc_dir = os.path.join(os.path.dirname(__file__), "csrc")
_SOURCES = [
    os.path.join(_csrc_dir, "iir_trailing.cpp"),
    os.path.join(_csrc_dir, "iir_trailing_kernel.cu"),
]
_CUDA_CFLAGS = ["--use_fast_math", "-O3"]
_ext_lock = threading.Lock()
_ext = None
_ext_error = None

check_buildable(_EXT_NAME, _SOURCES, _CUDA_CFLAGS, "the Python IIR loop")


def _load_ext():
    global _ext, _ext_error

    if _ext is not None:
        return _ext
    if _ext_error is not None:
        raise RuntimeError("IIR trailing CUDA extension is unavailable") from _ext_error

    with _ext_lock:
        if _ext is not None:
            return _ext
        if _ext_error is not None:
            raise RuntimeError("IIR trailing CUDA extension is unavailable") from _ext_error

        try:
            _ext = load_extension(_EXT_NAME, _SOURCES, _CUDA_CFLAGS)
        except Exception as exc:
            _ext_error = exc
            raise

    return _ext


def iir_trailing_cuda(signal: torch.Tensor, strength: float) -> torch.Tensor:
    """Causal 1-pole IIR trailing filter using CUDA kernel.

    Equivalent to the Python loop version but runs all rows in parallel.

    Args:
        signal: Any shape float32 CUDA tensor. IIR applied along last dim.
        strength: 0-1, maps to alpha = 1 - strength * 0.70.

    Returns:
        Filtered tensor (same shape).
    """
    if strength <= 0:
        return signal

    alpha = 1.0 - min(max(strength, 0.0), 1.0) * 0.70

    ext = _load_ext()
    return ext.iir_trailing_forward(signal.contiguous(), alpha)
