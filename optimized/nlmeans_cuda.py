"""NLMeans CUDA wrapper with lazy extension build.

The heavy C++/CUDA extension is built on first use instead of at import time so
loading the degradation registry does not block on toolchain work. The build is
cached per user (see cuda_ext); without a cached build and without cl.exe the
import fails, and callers use their PyTorch fallback.
"""

from __future__ import annotations

import os
import threading

from torch import Tensor

from optimized.cuda_ext import check_buildable, load_extension

_EXT_NAME = "nlmeans_cuda_ext_wtp_gui"
_csrc_dir = os.path.join(os.path.dirname(__file__), "csrc")
_SOURCES = [
    os.path.join(_csrc_dir, "nlmeans.cpp"),
    os.path.join(_csrc_dir, "nlmeans_kernel.cu"),
]
_CUDA_CFLAGS = ["--use_fast_math"]
_ext_lock = threading.Lock()
_nlmeans_ext = None
_nlmeans_error = None

check_buildable(_EXT_NAME, _SOURCES, _CUDA_CFLAGS, "the PyTorch NLMeans fallback")


def _load_nlmeans_ext():
    global _nlmeans_ext, _nlmeans_error

    if _nlmeans_ext is not None:
        return _nlmeans_ext
    if _nlmeans_error is not None:
        raise RuntimeError("NLMeans CUDA extension is unavailable") from _nlmeans_error

    with _ext_lock:
        if _nlmeans_ext is not None:
            return _nlmeans_ext
        if _nlmeans_error is not None:
            raise RuntimeError("NLMeans CUDA extension is unavailable") from _nlmeans_error

        try:
            _nlmeans_ext = load_extension(_EXT_NAME, _SOURCES, _CUDA_CFLAGS)
        except Exception as exc:
            _nlmeans_error = exc
            raise

    return _nlmeans_ext


def nlmeans_denoise_cuda(
    x: Tensor,
    h: float = 30.0,
    template_size: int = 7,
    search_size: int = 21,
) -> Tensor:
    """Denoise a 3-channel BCHW float32 CUDA tensor with the custom NLMeans kernel.

    The kernel indexes planar memory, so the input is made contiguous first.
    """

    ext = _load_nlmeans_ext()
    return ext.nlmeans_forward(x.contiguous(), h, template_size, search_size)
