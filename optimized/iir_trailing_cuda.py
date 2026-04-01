"""Lazy-loading CUDA IIR trailing filter (tape trailing effect).

JIT-compiles on first use, caches to avoid recompilation.
Follows the same pattern as nlmeans_cuda.py.
"""

from __future__ import annotations

import os
import subprocess
import sys
import threading

import torch
from torch.utils.cpp_extension import load

_EXT_NAME = "iir_trailing_cuda_ext_wtp_gui"
_csrc_dir = os.path.join(os.path.dirname(__file__), "csrc")
_ext_lock = threading.Lock()
_ext = None
_ext_error = None


def _suppress_console_load(**kwargs):
    """Call torch cpp_extension.load() without spawning visible console windows on Windows."""
    if sys.platform != "win32":
        return load(**kwargs)

    _orig_popen = subprocess.Popen

    class _SilentPopen(_orig_popen):
        def __init__(self, *args, **kw):
            kw.setdefault("creationflags", 0)
            kw["creationflags"] |= subprocess.CREATE_NO_WINDOW
            super().__init__(*args, **kw)

    subprocess.Popen = _SilentPopen  # type: ignore[misc]
    try:
        return load(**kwargs)
    finally:
        subprocess.Popen = _orig_popen  # type: ignore[misc]


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
            _ext = _suppress_console_load(
                name=_EXT_NAME,
                sources=[
                    os.path.join(_csrc_dir, "iir_trailing.cpp"),
                    os.path.join(_csrc_dir, "iir_trailing_kernel.cu"),
                ],
                extra_cuda_cflags=["--use_fast_math", "-O3"],
                verbose=False,
            )
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
