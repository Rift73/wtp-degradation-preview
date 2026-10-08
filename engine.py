"""
Processing engine for the WTP Degradation Preview GUI.

Runs the degradation pipeline on a worker thread with per-step timing, error
capture and deterministic per-step seeding, and holds the image and FFmpeg
helpers the window needs.
"""

import importlib
import logging
import os
import random
import shutil
import sys
import time
import traceback
from dataclasses import dataclass

import cv2
import numpy as np
from PySide6.QtCore import QCoreApplication, QObject, QThread, Signal
from PySide6.QtGui import QImage, QPixmap

logger = logging.getLogger(__name__)

_SUMMARY_LEN = 120


@dataclass
class StepResult:
    index: int                 # position in the configs list given to request_run
    type_key: str
    elapsed_ms: float
    error: str | None          # full traceback text, None when the step succeeded
    error_summary: str | None  # "ExceptionType: message", at most 120 chars


@dataclass
class RunResult:
    lq: np.ndarray
    hq: np.ndarray
    steps: list[StepResult]
    total_ms: float
    seed: int


# ──────────────────────────────────────────────
# Pipeline execution
# ──────────────────────────────────────────────

_pipeline_loaded = False


def _ensure_pipeline():
    """Import the degradation modules on first use (slow: torch, codecs)."""
    global _pipeline_loaded
    if _pipeline_loaded:
        return
    importlib.import_module("pipeline.process")  # registers every *_degr class
    _pipeline_loaded = True


def _clip_summary(text):
    text = text.strip()
    if len(text) > _SUMMARY_LEN:
        return text[:_SUMMARY_LEN - 1] + "…"
    return text


def _exception_summary(exc):
    message = str(exc).strip().split("\n")[0]
    name = type(exc).__name__
    return _clip_summary(f"{name}: {message}" if message else name)


def _unavailable(type_key, failed_modules):
    """(error, summary) for a type with no registered class."""
    module = f"{type_key}_degr"
    tb = failed_modules.get(module)
    if tb is None:
        return f"Unknown degradation type '{type_key}'", f"Unknown type '{type_key}'"
    last_line = next(
        (line for line in reversed(tb.strip().split("\n")) if line.strip()), ""
    )
    error = f"Module '{module}' failed to load (missing dependency?):\n{tb}"
    return error, _clip_summary(f"Module failed to load: {last_line}")


def _seed_step(seed, index):
    """Seed every RNG the pipeline uses, so each step's randomness depends
    only on the session seed and its own position."""
    step_seed = (seed * 1_000_003 + index) % 2**32
    np.random.seed(step_seed)
    random.seed(step_seed)
    torch = sys.modules.get("torch")
    if torch is not None:
        torch.manual_seed(step_seed)


def _run_pipeline(source, configs, seed):
    _ensure_pipeline()
    from pipeline.process import FAILED_MODULES
    from pipeline.utils.registry import get_class

    t_run = time.perf_counter()
    lq = source.copy()
    hq = source.copy()
    steps = []
    for index, config in enumerate(configs):
        type_key = config["type"]
        t0 = time.perf_counter()
        error = summary = None
        cls = get_class(type_key)
        if cls is None:
            error, summary = _unavailable(type_key, FAILED_MODULES)
        else:
            _seed_step(seed, index)
            try:
                result = cls(config).run(lq, hq)
            except Exception as exc:
                error = traceback.format_exc()
                summary = _exception_summary(exc)
            else:
                if result is not None:
                    lq, hq = result
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        steps.append(StepResult(index, type_key, elapsed_ms, error, summary))
    total_ms = (time.perf_counter() - t_run) * 1000.0
    return RunResult(lq, hq, steps, total_ms, seed)


class _RunThread(QThread):
    """Runs one job; the engine reads ``result`` / ``fatal`` after ``finished``."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.job = None
        self.result = None
        self.fatal = None

    def run(self):
        self.result = None
        self.fatal = None
        try:
            self.result = _run_pipeline(*self.job)
        except Exception:
            self.fatal = traceback.format_exc()


class PipelineEngine(QObject):
    """Runs pipelines off the GUI thread, one at a time.

    A request made while busy replaces any earlier waiting request and runs
    as soon as the current one finishes. A failing step is recorded in its
    StepResult and skipped; ``failed`` fires only when the engine itself breaks.
    """

    result_ready = Signal(object)  # RunResult
    failed = Signal(str)           # traceback text
    busy_changed = Signal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._thread = _RunThread(self)
        self._thread.finished.connect(self._on_finished)
        self._busy = False
        self._pending = None
        self._logged_errors = set()  # (type_key, summary) logged by the previous run
        app = QCoreApplication.instance()
        if app is not None:
            app.aboutToQuit.connect(self._on_quit)

    def is_busy(self):
        return self._busy

    def request_run(self, source, configs, seed):
        job = (source, configs, seed)
        if self._busy:
            self._pending = job
            return
        self._busy = True
        self.busy_changed.emit(True)
        self._start(job)

    def _start(self, job):
        self._thread.job = job
        self._thread.start()

    def _on_finished(self):
        thread = self._thread
        if thread.fatal is not None:
            logger.error("Pipeline engine failed:\n%s", thread.fatal)
            self.failed.emit(thread.fatal)
        else:
            self._log_step_errors(thread.result)
            self.result_ready.emit(thread.result)
        if self._pending is not None:
            job, self._pending = self._pending, None
            self._start(job)
        else:
            self._busy = False
            self.busy_changed.emit(False)

    def _log_step_errors(self, result):
        """Log each failing step once, not again on every re-run while it keeps failing."""
        current = set()
        for step in result.steps:
            if step.error is None:
                continue
            key = (step.type_key, step.error_summary)
            current.add(key)
            if key not in self._logged_errors:
                logger.error(
                    "Step %d (%s) failed:\n%s", step.index + 1, step.type_key, step.error
                )
        self._logged_errors = current

    def _on_quit(self):
        self._pending = None
        self._thread.wait()


# ──────────────────────────────────────────────
# Image utilities
# ──────────────────────────────────────────────

def load_image(path):
    """Load an image as float32 RGB in [0, 1], or None if it cannot be read.

    Grayscale is stacked to three channels, alpha is dropped, 8/16-bit are
    normalized and float images are clipped. Reads through numpy so that
    non-ASCII paths work on Windows.
    """
    try:
        data = np.fromfile(path, dtype=np.uint8)
    except OSError:
        return None
    img = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    if img.dtype == np.uint8:
        img = img.astype(np.float32) / 255.0
    elif img.dtype == np.uint16:
        img = img.astype(np.float32) / 65535.0
    else:
        img = img.astype(np.float32)
    if img.ndim == 3:
        channels = img.shape[2]
        if channels == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
        elif channels == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:  # gray or gray + alpha
            img = img[:, :, 0]
    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)
    return np.ascontiguousarray(np.clip(img, 0.0, 1.0))


def numpy_to_qpixmap(img):
    """Convert a float [0, 1] image (HxW, HxWx1 or HxWx3) to a QPixmap."""
    if img.ndim == 3 and img.shape[2] == 1:
        img = img[:, :, 0]
    img_u8 = np.ascontiguousarray((np.clip(img, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8))
    h, w = img_u8.shape[:2]
    if img_u8.ndim == 2:
        qimg = QImage(img_u8.data, w, h, w, QImage.Format.Format_Grayscale8)
    else:
        qimg = QImage(img_u8.data, w, h, w * 3, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(qimg.copy())


# ──────────────────────────────────────────────
# FFmpeg discovery
# ──────────────────────────────────────────────

_dll_dirs = {}  # directory -> handle from os.add_dll_directory (kept alive)


def ffmpeg_available():
    """True if an ffmpeg executable is reachable through PATH."""
    return shutil.which("ffmpeg") is not None


def _use_ffmpeg_dir(ffdir):
    """Put ffdir on PATH (ffmpeg CLI) and on the DLL search path, so
    torchcodec can resolve a shared FFmpeg build's libav* DLLs."""
    if ffdir not in os.environ.get("PATH", "").split(os.pathsep):
        os.environ["PATH"] = ffdir + os.pathsep + os.environ.get("PATH", "")
    if sys.platform == "win32" and ffdir not in _dll_dirs:
        _dll_dirs[ffdir] = os.add_dll_directory(ffdir)


def restore_ffmpeg(cfg):
    """Make FFmpeg usable from cfg["ffmpeg_path"], or else from the ffmpeg
    already on PATH. Call before the first run so the DLL directory is
    registered before the pipeline imports torchcodec. Returns availability."""
    path = cfg.get("ffmpeg_path", "")
    if not (path and os.path.isfile(path)):
        path = shutil.which("ffmpeg")
    if path:
        _use_ffmpeg_dir(os.path.dirname(os.path.abspath(path)))
    return ffmpeg_available()


def register_ffmpeg(path, cfg):
    """Remember a user-selected ffmpeg executable in cfg and put it to use."""
    cfg["ffmpeg_path"] = path
    _use_ffmpeg_dir(os.path.dirname(os.path.abspath(path)))
