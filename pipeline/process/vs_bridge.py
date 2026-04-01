"""VapourSynth bridge for detail mask generation.

Handles VS plugin loading and numpy<->VS conversion so that vsmasktools
functions can be called from the numpy-based degradation pipeline.

Plugins are loaded once on first use from a hardcoded directory.
If VapourSynth is not installed, _HAS_VS is False and all mask
functions return None.
"""

import logging

import numpy as np

_HAS_VS = False
_plugins_loaded = False

_VS_PLUGIN_DIR = r"C:\Users\PC\AppData\Local\Programs\VapourSynth\plugins"

try:
    import vapoursynth as vs
    import glob
    import os

    _HAS_VS = True
except ImportError:
    vs = None

# BT.709 luma coefficients
_LUMA_R, _LUMA_G, _LUMA_B = 0.2126, 0.7152, 0.0722


def _ensure_plugins():
    """Load all VS plugins from the plugin directory (idempotent)."""
    global _plugins_loaded
    if _plugins_loaded or not _HAS_VS:
        return
    _plugins_loaded = True

    core = vs.core
    for dll in glob.glob(os.path.join(_VS_PLUGIN_DIR, "*.dll")):
        try:
            core.std.LoadPlugin(dll)
        except Exception:
            pass
    logging.debug("VS plugins loaded from %s", _VS_PLUGIN_DIR)


def numpy_to_vs_gray(img: np.ndarray) -> "vs.VideoNode":
    """Convert HWC float32 numpy image to a single-frame GRAYS VS clip.

    If input is 3-channel, extracts luma via BT.709 coefficients.
    If input is 2D (grayscale), uses directly.
    """
    _ensure_plugins()
    core = vs.core

    if img.ndim == 3:
        luma = (_LUMA_R * img[:, :, 0] +
                _LUMA_G * img[:, :, 1] +
                _LUMA_B * img[:, :, 2])
    else:
        luma = img

    luma = np.ascontiguousarray(luma, dtype=np.float32)
    h, w = luma.shape

    blank = core.std.BlankClip(width=w, height=h, format=vs.GRAYS, length=1)

    def _inject(n, f):
        fout = f.copy()
        np.copyto(np.asarray(fout[0]), luma)
        return fout

    return core.std.ModifyFrame(blank, blank, _inject)


def vs_to_numpy_gray(clip: "vs.VideoNode") -> np.ndarray:
    """Fetch frame 0 of a GRAY VS clip as (H, W) float32 numpy array."""
    frame = clip.get_frame(0)
    return np.asarray(frame[0]).copy()


def apply_detail_mask_neo(
    img: np.ndarray,
    sigma: float = 1.0,
    detail_brz: float = 0.05,
    lines_brz: float = 0.08,
) -> np.ndarray | None:
    """Generate detail mask using vsmasktools.detail_mask_neo.

    Returns (H, W) float32 mask where 1=detail, 0=flat, or None if VS unavailable.
    """
    if not _HAS_VS:
        return None

    from vsmasktools import detail_mask_neo

    _ensure_plugins()
    clip = numpy_to_vs_gray(img)
    mask_clip = detail_mask_neo(clip, sigma=sigma, detail_brz=detail_brz, lines_brz=lines_brz)
    return vs_to_numpy_gray(mask_clip)


def apply_detail_mask(
    img: np.ndarray,
    brz_mm: float = 0.05,
    brz_ed: float = 0.08,
) -> np.ndarray | None:
    """Generate detail mask using vsmasktools.detail_mask (Kirsch-based).

    Returns (H, W) float32 mask where 1=detail, 0=flat, or None if VS unavailable.
    """
    if not _HAS_VS:
        return None

    from vsmasktools import detail_mask

    _ensure_plugins()
    clip = numpy_to_vs_gray(img)
    mask_clip = detail_mask(clip, brz_mm=brz_mm, brz_ed=brz_ed)
    return vs_to_numpy_gray(mask_clip)
