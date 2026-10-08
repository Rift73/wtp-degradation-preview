"""
Video codec backend for the compress step.

The backend is the first that works of: the ffmpeg executable the user
located, the ffmpeg on PATH, PyAV's bundled FFmpeg (in-process). Preferring
PyAV puts it first. detect() decides once and caches the answer; the
setters drop the cache. Every function is safe to call from any thread.
"""

import logging
import os
import re
import shutil
import subprocess
import sys
import threading
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# No console window per ffmpeg call on Windows
POPEN_FLAGS = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
KINDS = ("ffmpeg", "pyav")
_PROBE_TIMEOUT_S = 5  # a GUI program picked by mistake would never exit


@dataclass(frozen=True)
class Backend:
    kind: str              # "ffmpeg" | "pyav" | "none"
    path: str | None       # the ffmpeg executable when kind is "ffmpeg"
    version: str           # "8.1", "git 2026-07-28", PyAV's FFmpeg version, or ""
    encoders: frozenset    # video encoder names from `ffmpeg -encoders`; empty unless "ffmpeg"


_lock = threading.Lock()
_preference = "ffmpeg"   # the kind tried first
_configured = None       # the executable the user located, or None
_backend = None          # detect() cache
_probes = {}             # executable -> (version, encoders), or None when it is not a working ffmpeg
_pixel_formats = {}      # (executable, encoder) -> tuple of pixel format names


def _run(path, *args):
    """stdout of `path -hide_banner *args`; raises OSError or SubprocessError."""
    proc = subprocess.run(
        [path, "-hide_banner", *args], stdin=subprocess.DEVNULL, capture_output=True,
        encoding="utf-8", errors="replace", timeout=_PROBE_TIMEOUT_S, check=True,
        creationflags=POPEN_FLAGS,
    )
    return proc.stdout


def _short_version(token):
    """'8.1' from '8.1-full_build-www.gyan.dev' or 'n8.1'; 'git 2026-07-28' from a
    dated git build; else the token's first 16 characters."""
    release = re.match(r"n?(\d+\.\d+(?:\.\d+)?)", token)
    if release:
        return release.group(1)
    date = re.search(r"\d{4}-\d{2}-\d{2}", token)
    if date:
        return f"git {date.group(0)}"
    return token[:16]


def _probe(path):
    """(version, encoders) of the ffmpeg at path, or None when it is not one."""
    if path not in _probes:
        try:
            version_text = _run(path, "-version")
            encoders_text = _run(path, "-encoders")
        except (OSError, subprocess.SubprocessError) as exc:
            logger.warning("%s is not a working ffmpeg: %s", path, exc)
            _probes[path] = None
            return None
        match = re.search(r"ffmpeg version (\S+)", version_text)
        if match is None:
            logger.warning("%s does not report an ffmpeg version", path)
            _probes[path] = None
            return None
        listing = encoders_text.split("------", 1)[-1]  # after the flag legend
        encoders = frozenset(re.findall(r"^\s*V[A-Z.]{5}\s+(\S+)", listing, re.MULTILINE))
        _probes[path] = (_short_version(match.group(1)), encoders)
    return _probes[path]


def _system_ffmpeg():
    """Backend for the located ffmpeg, else the one on PATH, or None."""
    for path in (_configured, shutil.which("ffmpeg")):
        if path:
            path = os.path.abspath(path)
            probe = _probe(path)
            if probe is not None:
                return Backend("ffmpeg", path, *probe)
    return None


def _pyav():
    try:
        import av
    except ImportError:
        return None
    return Backend("pyav", None, av.ffmpeg_version_info, frozenset())


def detect():
    """The backend in use (cached)."""
    global _backend
    with _lock:
        if _backend is None:
            order = (_pyav, _system_ffmpeg) if _preference == "pyav" else (_system_ffmpeg, _pyav)
            _backend = next(
                (b for b in (find() for find in order) if b is not None),
                Backend("none", None, "", frozenset()),
            )
            logger.info("Video backend: %s %s %s", _backend.kind, _backend.version,
                        _backend.path or "")
        return _backend


def ffmpeg_available():
    """True when a located or PATH ffmpeg works, whichever backend is preferred."""
    with _lock:
        return _system_ffmpeg() is not None


def pyav_available():
    return _pyav() is not None


def set_preference(kind):
    """Try kind ("ffmpeg" or "pyav") first from now on."""
    global _preference, _backend
    if kind not in KINDS:
        raise ValueError(f"unknown video backend '{kind}'")
    with _lock:
        _preference = kind
        _backend = None


def restore_ffmpeg(cfg):
    """Apply cfg's "ffmpeg_path" and "video_backend"; returns detect()."""
    global _configured, _preference, _backend
    path = cfg.get("ffmpeg_path", "")
    kind = cfg.get("video_backend", "ffmpeg")
    with _lock:
        _configured = path if path and os.path.isfile(path) else None
        _preference = kind if kind in KINDS else "ffmpeg"
        _backend = None
    return detect()


def register_ffmpeg(path, cfg):
    """Use a user-selected ffmpeg executable, preferred over PyAV, and remember
    both in cfg. Returns False, changing nothing, when path is not a working ffmpeg."""
    global _configured, _preference, _backend
    path = os.path.abspath(path)
    with _lock:
        if _probe(path) is None:
            return False
        _configured = path
        _preference = "ffmpeg"
        _backend = None
    cfg["ffmpeg_path"] = path
    cfg["video_backend"] = "ffmpeg"
    return True


def pixel_formats(path, encoder):
    """Pixel formats the encoder of the ffmpeg at path accepts (cached; empty
    when ffmpeg does not list them). Raises OSError or SubprocessError."""
    key = (path, encoder)
    with _lock:
        if key not in _pixel_formats:
            text = _run(path, "-h", f"encoder={encoder}")
            match = re.search(r"Supported pixel formats:([^\n]*)", text)
            _pixel_formats[key] = tuple(match.group(1).split()) if match else ()
        return _pixel_formats[key]
