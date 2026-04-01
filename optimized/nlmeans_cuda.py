"""NLMeans CUDA wrapper with lazy extension build.

The heavy C++/CUDA extension is built on first use instead of at import time so
loading the degradation registry does not block on toolchain work.
"""

from __future__ import annotations

import os
import threading

from torch import Tensor
from torch.utils.cpp_extension import load

_csrc_dir = os.path.join(os.path.dirname(__file__), "csrc")
_ext_lock = threading.Lock()
_nlmeans_ext = None
_nlmeans_error = None

# Use a distinct, repo-stable extension name so this rebuild avoids the damaged
# cache entry from the broken experimental tree without baking the directory name
# into the extension identity.
_EXT_NAME = "nlmeans_cuda_ext_wtp_gui"


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
            _nlmeans_ext = load(
                name=_EXT_NAME,
                sources=[
                    os.path.join(_csrc_dir, "nlmeans.cpp"),
                    os.path.join(_csrc_dir, "nlmeans_kernel.cu"),
                ],
                extra_cuda_cflags=["--use_fast_math"],
                verbose=False,
            )
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
    """Denoise a BCHW CUDA tensor with the custom NLMeans kernel."""

    ext = _load_nlmeans_ext()
    return ext.nlmeans_forward(x, h, template_size, search_size)
