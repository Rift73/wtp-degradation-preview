"""
Processing engine for the WTP Degradation Preview GUI.

Runs the degradation pipeline on a worker thread with per-step timing, error
capture, deterministic per-step seeding and a cache of step outputs (a run
resumes after the longest unchanged prefix), and holds the image and FFmpeg
helpers the window needs.
"""

import importlib
import json
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
_CACHE_BUDGET_BYTES = 1536 * 1024 ** 2  # cached step outputs (1.5 GiB)


@dataclass
class StepResult:
    index: int                 # position in the configs list given to request_run
    type_key: str
    elapsed_ms: float          # 0.0 for a cached step
    error: str | None          # full traceback text, None when the step succeeded
    error_summary: str | None  # "ExceptionType: message", at most 120 chars
    cached: bool = False       # output reused from an earlier run, not run again


@dataclass
class RunResult:
    lq: np.ndarray
    hq: np.ndarray
    steps: list[StepResult]
    total_ms: float            # work this run did; cached steps add nothing
    seed: int
    cached_steps: int          # leading steps whose output came from the cache
    hq_changed: bool           # a step replaced HQ; False means hq still equals the source
    lq_u8: np.ndarray          # lq clipped and rounded to contiguous uint8, for display
    hq_u8: np.ndarray | None   # hq likewise, or None when hq_changed is False


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


@dataclass
class _CacheEntry:
    depth: int          # index of the step that produced it
    lq: np.ndarray
    hq: np.ndarray      # is lq when the step left one array as both
    hq_changed: bool    # RunResult.hq_changed up to and including this step
    used: int           # the last run that stored it or resumed through it

    def arrays(self):
        return (self.lq,) if self.hq is self.lq else (self.lq, self.hq)


class _StepCache:
    """Private copies of the outputs of successful steps, used only from the
    worker thread.

    Each step is reseeded right before it runs, so step i's output depends
    only on the source, configs[:i + 1] and the seed: an entry is keyed by the
    seed and the canonical JSON of that prefix, and a different source array
    object clears the cache (callers pass a new array for a new image and
    never modify one in place). Steps may write into their inputs (halo does),
    so no step ever receives a cached array.
    """

    def __init__(self):
        self.source = None
        self.entries = {}  # (seed, tuple of config JSON up to the step) -> _CacheEntry
        self.run_id = 0

    def begin(self, source, configs, seed):
        """Start a run. Returns the per-step keys of the request, the number of
        leading steps it can skip, and the entry to resume from (or None)."""
        if source is not self.source:
            self.source = source
            self.entries.clear()
        self.run_id += 1
        texts = [json.dumps(config, sort_keys=True) for config in configs]
        keys = [(seed, tuple(texts[:i + 1])) for i in range(len(texts))]
        start, entry = 0, None
        for depth in range(len(keys), 0, -1):  # eviction leaves gaps, so search from the deep end
            if keys[depth - 1] in self.entries:
                start, entry = depth, self.entries[keys[depth - 1]]
                break
        for key in keys[:start]:
            if key in self.entries:
                self.entries[key].used = self.run_id
        return keys, start, entry

    def store(self, key, depth, lq, hq, hq_copy, hq_changed):
        """Keep copies of a step's output. hq_copy is a cached array equal to
        hq, or None. Returns the cached array now equal to hq (hq_copy as given
        when the output alone exceeds the budget and is not kept)."""
        shared = lq is hq
        nbytes = lq.nbytes + (0 if shared or hq_copy is not None else hq.nbytes)
        if nbytes > _CACHE_BUDGET_BYTES:
            return hq_copy
        lq_copy = lq.copy(order="K")
        if shared:
            hq_copy = lq_copy
        elif hq_copy is None:
            hq_copy = hq.copy(order="K")
        self.entries[key] = _CacheEntry(depth, lq_copy, hq_copy, hq_changed, self.run_id)
        self._evict()
        return hq_copy

    def _evict(self):
        """Drop entries until the cached arrays fit the budget: entries of
        older runs first (stale branches of earlier edits), then the
        shallowest, since edits to the last steps are the common case. An
        array shared by several entries counts once."""
        holders = {}  # id(array) -> [nbytes, entries holding it]
        for entry in self.entries.values():
            for array in entry.arrays():
                holders.setdefault(id(array), [array.nbytes, 0])[1] += 1
        total = sum(nbytes for nbytes, _ in holders.values())
        order = sorted(self.entries.items(), key=lambda item: (item[1].used, item[1].depth))
        for key, entry in order:
            if total <= _CACHE_BUDGET_BYTES:
                break
            del self.entries[key]
            for array in entry.arrays():
                holder = holders[id(array)]
                holder[1] -= 1
                if holder[1] == 0:
                    total -= holder[0]


def _run_pipeline(cache, source, configs, seed):
    _ensure_pipeline()
    from pipeline.process import FAILED_MODULES
    from pipeline.utils.registry import get_class

    t_run = time.perf_counter()
    keys, start, entry = cache.begin(source, configs, seed)
    steps = [StepResult(i, configs[i]["type"], 0.0, None, None, cached=True) for i in range(start)]
    if entry is None:
        lq = source.copy()
        hq = source.copy()
        hq_copy, hq_changed = None, False
    else:
        # Fresh copies, one array for both when the cold run had one, so the
        # remaining steps see exactly what they would have seen
        lq = entry.lq.copy(order="K")
        hq = lq if entry.hq is entry.lq else entry.hq.copy(order="K")
        hq_copy, hq_changed = entry.hq, entry.hq_changed
    # hq_copy is a cached array equal to the working hq, or None when unknown;
    # entries share it while no step replaces HQ (steps write into lq, never
    # into an hq they return unchanged, unless hq is also their lq).
    # Only successful steps are cached. After a failed step nothing more is:
    # the later steps' outputs depend on the failure.
    caching = True
    for index in range(start, len(configs)):
        config = configs[index]
        type_key = config["type"]
        hq_in, hq_shape, shared_in = hq, hq.shape, lq is hq
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
        replaced = hq is not hq_in or hq.shape != hq_shape
        hq_changed = hq_changed or replaced
        if replaced or shared_in:  # a write into lq may have reached hq
            hq_copy = None
        if error is not None:
            caching = False
        elif caching:
            hq_copy = cache.store(keys[index], index, lq, hq, hq_copy, hq_changed)
    lq_u8 = _to_uint8(lq)
    hq_u8 = _to_uint8(hq) if hq_changed else None
    total_ms = (time.perf_counter() - t_run) * 1000.0
    return RunResult(lq, hq, steps, total_ms, seed, start, hq_changed, lq_u8, hq_u8)


class _RunThread(QThread):
    """Runs one job; the engine reads ``result`` / ``fatal`` after ``finished``."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.job = None
        self.result = None
        self.fatal = None
        self.cache = _StepCache()

    def run(self):
        self.result = None
        self.fatal = None
        try:
            self.result = _run_pipeline(self.cache, *self.job)
        except Exception:
            self.fatal = traceback.format_exc()


class PipelineEngine(QObject):
    """Runs pipelines off the GUI thread, one at a time.

    A request made while busy replaces any earlier waiting request and runs
    as soon as the current one finishes. A run resumes after the longest
    prefix of steps it shares with earlier runs on the same source array and
    seed. A failing step is recorded in its StepResult and skipped; ``failed``
    fires only when the engine itself breaks.
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
        """configs must be JSON-serializable; source must not be modified
        afterwards (pass a new array for a new image)."""
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


def _to_uint8(img):
    """Clip a float [0, 1] image and round it to contiguous uint8 (same shape)."""
    out = np.clip(img, 0.0, 1.0)
    out *= 255.0
    out += 0.5
    return np.ascontiguousarray(out.astype(np.uint8))


def numpy_to_qpixmap(img):
    """Convert an image (HxW, HxWx1 or HxWx3; uint8, or float in [0, 1]) to a QPixmap."""
    if img.dtype != np.uint8:
        img = _to_uint8(img)
    if img.ndim == 3 and img.shape[2] == 1:
        img = img[:, :, 0]
    img = np.ascontiguousarray(img)
    h, w = img.shape[:2]
    if img.ndim == 2:
        qimg = QImage(img.data, w, h, w, QImage.Format.Format_Grayscale8)
    else:
        qimg = QImage(img.data, w, h, w * 3, QImage.Format.Format_RGB888)
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
