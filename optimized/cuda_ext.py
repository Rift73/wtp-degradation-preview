"""Cached loader for the JIT-built CUDA extensions.

torch.utils.cpp_extension.load() re-runs ninja and probes ``cl`` in every
fresh process, so even a cached build costs a toolchain round trip, and it
fails outright without the MSVC environment. Builds here live in a per-user
directory keyed by the Python and torch versions plus a digest of the sources
and flags; once the module exists it is imported directly, and load() runs
only to build it.
"""

from __future__ import annotations

import hashlib
import importlib.util
import logging
import os
import shutil
import subprocess
import sys
from types import ModuleType
from unittest import mock

import torch
from torch.utils.cpp_extension import get_default_build_root, load

logger = logging.getLogger(__name__)

# CUDA 13's CCCL headers refuse MSVC's traditional preprocessor
_HOST_CUDA_CFLAGS = ["-Xcompiler", "/Zc:preprocessor"] if sys.platform == "win32" else []
_MODULE_SUFFIX = ".pyd" if sys.platform == "win32" else ".so"


def _build_paths(name: str, sources: list[str], cuda_cflags: list[str]) -> tuple[str, str]:
    """Return (build directory, module path) for this source and flag set."""
    digest = hashlib.sha256(repr(cuda_cflags).encode())
    for path in sources:
        with open(path, "rb") as f:
            digest.update(f.read())
    root = os.environ.get("LOCALAPPDATA") or get_default_build_root()
    versions = f"py{sys.version_info.major}{sys.version_info.minor}_torch{torch.__version__}"
    build_directory = os.path.join(
        root, "wtp_preview", "torch_ext", versions, f"{name}_{digest.hexdigest()[:12]}",
    )
    return build_directory, os.path.join(build_directory, name + _MODULE_SUFFIX)


def check_buildable(name: str, sources: list[str], extra_cuda_cflags: list[str], fallback: str) -> None:
    """Raise ImportError when there is no cached build and no cl.exe to make one.

    Called at import so the caller's ImportError guard takes the fallback
    without a build attempt (torch would log a WinError 2 traceback first).
    """
    if sys.platform != "win32" or shutil.which("cl"):
        return
    _, module_path = _build_paths(name, sources, extra_cuda_cflags + _HOST_CUDA_CFLAGS)
    if not os.path.exists(module_path):
        logger.info("%s: no cached build and cl.exe is not on PATH; using %s", name, fallback)
        raise ImportError(f"{name} is not built and cl.exe is not on PATH")


def _load_without_console(**kwargs) -> ModuleType:
    """Call load() without flashing a console window per tool call on Windows."""
    if sys.platform != "win32":
        return load(**kwargs)

    class _SilentPopen(subprocess.Popen):
        def __init__(self, *args, **kw):
            kw["creationflags"] = kw.get("creationflags", 0) | subprocess.CREATE_NO_WINDOW
            super().__init__(*args, **kw)

    with mock.patch.object(subprocess, "Popen", _SilentPopen):
        return load(**kwargs)


def load_extension(name: str, sources: list[str], extra_cuda_cflags: list[str]) -> ModuleType:
    """Import the cached build of a CUDA extension, building it on first use."""
    cuda_cflags = extra_cuda_cflags + _HOST_CUDA_CFLAGS
    build_directory, module_path = _build_paths(name, sources, cuda_cflags)
    if os.path.exists(module_path):
        spec = importlib.util.spec_from_file_location(name, module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    os.makedirs(build_directory, exist_ok=True)
    return _load_without_console(
        name=name,
        sources=sources,
        extra_cuda_cflags=cuda_cflags,
        build_directory=build_directory,
        verbose=False,
    )
