"""
Processing engine for the WTP Degradation Preview GUI.

Runs the degradation pipeline on a worker thread with per-step timing, error
capture, deterministic per-step seeding, a cache of step outputs (a run
resumes after the longest unchanged prefix, from host memory or, inside a
group of GPU steps, from the device) and GPU-resident hand-off between
consecutive GPU steps, and holds the image helpers the window needs.
"""

import importlib
import json
import logging
import random
import sys
import time
import traceback
from dataclasses import dataclass

import cv2
import numpy as np
from PySide6.QtCore import QCoreApplication, QObject, QThread, Signal
from PySide6.QtGui import QImage, QPixmap

import video_backend

logger = logging.getLogger(__name__)

_SUMMARY_LEN = 120
_CACHE_BUDGET_BYTES = 1536 * 1024 ** 2  # cached step outputs in host memory (1.5 GiB)
_DEVICE_CACHE_BYTES = 2 * 1024 ** 3      # cached tensors of steps inside GPU groups: at most 2 GiB
_DEVICE_CACHE_SHARE = 0.25               # and this share of the GPU memory free at first use


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
    hq_u8: np.ndarray | None   # hq likewise, or None when hq_changed is False; the previous
                               # result's own array when hq_same_as_previous (never modify it)
    hq_same_as_previous: bool  # hq is known to equal the previous run's hq


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
    """A step's output: copies of its arrays, or for a step inside a GPU group
    (a tensor entry) the tensors its run_tensor left, uncopied."""
    depth: int          # index of the step that produced it
    lq: object          # np.ndarray, or a CUDA tensor in a tensor entry
    hq: object          # is lq when the step left one array as both
    hq_changed: bool    # RunResult.hq_changed up to and including this step
    used: int           # the last run that stored it or resumed through it
    # Tensor entries only: per lq and hq, while it is still the group's upload,
    # the cached array that upload came from (the cold run passes that array
    # on), else None
    sources: tuple | None = None

    def arrays(self, tensors=False):
        """The distinct host arrays the entry holds, or with tensors its CUDA tensors."""
        if self.sources is None:
            held = () if tensors else (self.lq, self.hq)
        else:
            held = (self.lq, self.hq) if tensors else self.sources
        return tuple({id(array): array for array in held if array is not None}.values())


class _StepCache:
    """Private copies of the outputs of successful steps, used only from the
    worker thread.

    Each step is reseeded right before it runs, so step i's output depends
    only on the source, configs[:i + 1], the seed and the video backend: an
    entry is keyed by the seed and the canonical JSON of that prefix, and a
    different source array object or video backend clears the cache (callers
    pass a new array for a new image and never modify one in place). Steps
    may write into their inputs (halo does), so no step ever receives a
    cached array. The last step of a GPU group is stored as arrays, the steps
    before it as tensor entries on the device (run_tensor never writes into
    its inputs), under a budget of their own.

    The cache also keeps the last run's hq for display, with the cached array
    it came from: a run that ends on that same array (cached arrays are never
    modified) reuses it instead of converting hq again.
    """

    def __init__(self):
        self.source = None
        self.backend = None
        self.entries = {}  # (seed, tuple of config JSON up to the step) -> _CacheEntry
        self.run_id = 0
        self.device_budget = None  # bytes for tensor entries, set by the first run on a GPU
        self.shown_hq = None  # the cached array the last run's hq equals, or None when unknown
        self.shown_u8 = None  # that run's RunResult.hq_u8

    def begin(self, source, backend, configs, seed):
        """Start a run. Returns the per-step keys of the request, the number of
        leading steps it can skip, and the entry to resume from (or None)."""
        if source is not self.source or backend != self.backend:
            self.source = source
            self.backend = backend
            self.entries.clear()
            self.shown_hq = self.shown_u8 = None
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
        hq, or None. Returns the cached arrays now equal to lq and hq (None and
        hq_copy as given when the output alone exceeds the budget and is not
        kept)."""
        shared = lq is hq
        nbytes = lq.nbytes + (0 if shared or hq_copy is not None else hq.nbytes)
        if nbytes > _CACHE_BUDGET_BYTES:
            return None, hq_copy
        lq_copy = lq.copy(order="K")
        if shared:
            hq_copy = lq_copy
        elif hq_copy is None:
            hq_copy = hq.copy(order="K")
        self.entries[key] = _CacheEntry(depth, lq_copy, hq_copy, hq_changed, self.run_id)
        self._evict()
        return lq_copy, hq_copy

    def store_tensors(self, key, depth, lq, hq, sources, hq_changed):
        """Keep the tensors a step inside a GPU group left, as they are."""
        self.entries[key] = _CacheEntry(depth, lq, hq, hq_changed, self.run_id, sources)
        self._evict()

    def display_hq(self, hq, hq_copy, hq_changed):
        """End a run: returns its RunResult.hq_u8 and hq_same_as_previous.
        hq_copy is a cached array equal to hq, or None when unknown; the same
        one as the last run's means the same pixels, so that run's hq_u8 is
        reused as it is."""
        same = hq_copy is not None and hq_copy is self.shown_hq
        if not hq_changed:
            hq_u8 = None
        elif same:
            hq_u8 = self.shown_u8
        else:
            hq_u8 = _to_uint8(hq)
        self.shown_hq, self.shown_u8 = hq_copy, hq_u8
        self._evict()
        return hq_u8, same

    def _evict(self):
        """Drop entries until the cached host arrays fit the host budget and
        the cached tensors the device budget: entries of older runs first
        (stale branches of earlier edits), then the shallowest, since edits to
        the last steps are the common case. An array or tensor held by several
        entries counts once (each cached tensor owns its memory: it is an
        upload or a settled output). shown_u8 counts as held by the entries
        holding shown_hq, and both are forgotten with the last of them."""
        for tensors, budget in ((False, _CACHE_BUDGET_BYTES), (True, self.device_budget)):
            held = {key: entry.arrays(tensors) for key, entry in self.entries.items()}
            if not tensors and self.shown_u8 is not None:
                for key, arrays in held.items():
                    if any(array is self.shown_hq for array in arrays):
                        held[key] = arrays + (self.shown_u8,)
            holders = {}  # id(array) -> [nbytes, entries holding it]
            for arrays in held.values():
                for array in arrays:
                    holders.setdefault(id(array), [array.nbytes, 0])[1] += 1
            if not holders:
                continue
            total = sum(nbytes for nbytes, _ in holders.values())
            order = sorted(held, key=lambda key: (self.entries[key].used, self.entries[key].depth))
            for key in order:
                if total <= budget:
                    break
                if not held[key]:
                    continue
                del self.entries[key]
                for array in held[key]:
                    holder = holders[id(array)]
                    holder[1] -= 1
                    if holder[1] == 0:
                        total -= holder[0]
        if self.shown_hq is not None and not any(
                array is self.shown_hq for entry in self.entries.values() for array in entry.arrays()):
            self.shown_hq = self.shown_u8 = None


def _run_step(config, index, lq, hq, seed, get_class, failed_modules):
    """Run one step through its numpy run(). Returns (lq, hq, [StepResult])."""
    type_key = config["type"]
    t0 = time.perf_counter()
    error = summary = None
    cls = get_class(type_key)
    if cls is None:
        error, summary = _unavailable(type_key, failed_modules)
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
    return lq, hq, [StepResult(index, type_key, elapsed_ms, error, summary)]


def _device_budget():
    """Bytes the cache may keep on the GPU: _DEVICE_CACHE_BYTES, at most
    _DEVICE_CACHE_SHARE of the device memory free now."""
    free, _ = sys.modules["torch"].cuda.mem_get_info()
    return min(_DEVICE_CACHE_BYTES, int(free * _DEVICE_CACHE_SHARE))


def _gpu_handoff():
    """optimized.gpu_degradations when steps can hand tensors over on a CUDA
    device, else None."""
    torch = sys.modules.get("torch")
    if torch is None or not torch.cuda.is_available():
        return None
    try:
        from optimized import gpu_degradations
    except ImportError:
        return None
    return gpu_degradations


def _is_rgb(img):
    return img.ndim == 3 and img.shape[2] == 3


def _group_end(configs, index, get_class):
    """End (exclusive) of the run of steps from index whose classes define run_tensor."""
    end = index
    while end < len(configs) and hasattr(get_class(configs[end]["type"]), "run_tensor"):
        end += 1
    return end


def _run_group(gpu, configs, start, end, lq, hq, sources, seed, get_class, resume=None,
               keep=False):
    """Run configs[start:end] through run_tensor with lq and hq on the GPU:
    uploaded at the first step, or continued from resume (the tensor entry of
    step start - 1), and downloaded after the last. Between steps a new
    output is settled exactly as a download and an upload would leave it, so
    the result is bit-identical to running each step's run() in turn.

    lq and hq are the arrays the group's uploads stand for, passed on as they
    are while the steps leave them unchanged (as run() does); sources are
    cached arrays equal to them, or None. With keep, the tensors after each
    step but the last, up to the first failure, are returned for the cache
    as (index, lq, hq, sources), unless an unchanged upload has no source.

    Step times come from CUDA events on the stream: the first step's includes
    the upload, the last step's the download. A failed upload fails the step
    (the next one tries again); a failed download fails the run.
    Returns (lq, hq, [StepResult], kept).
    """
    torch = sys.modules["torch"]
    events = [torch.cuda.Event(enable_timing=True) for _ in range(end - start + 1)]
    events[0].record()
    if resume is None:
        uploaded = False
        lq_up = hq_up = lq_t = hq_t = None
    else:
        uploaded = True
        lq_t, hq_t = resume.lq, resume.hq
        lq_up = lq_t if sources[0] is not None else None
        hq_up = hq_t if sources[1] is not None else None
    outcomes = []  # (error, summary) per step
    kept = []
    for index in range(start, end):
        config = configs[index]
        error = summary = None
        _seed_step(seed, index)
        try:
            if not uploaded:
                lq_t = gpu.image_to_tensor(lq)
                hq_t = lq_t if hq is lq else gpu.image_to_tensor(hq)
                lq_up, hq_up, uploaded = lq_t, hq_t, True
            lq_out, hq_out = get_class(config["type"])(config).run_tensor(lq_t, hq_t)
            if index < end - 1:  # the last step's outputs are downloaded instead
                # An input returned as is is unchanged; anything else is new
                if lq_out is not lq_t:
                    lq_out = gpu.settle_tensor(lq_out)
                if hq_out is not hq_t:
                    hq_out = gpu.settle_tensor(hq_out)
        except Exception as exc:
            error = traceback.format_exc()
            summary = _exception_summary(exc)
            keep = False  # the later outputs depend on the failure
        else:
            lq_t, hq_t = lq_out, hq_out
            lq_known = lq_t is not lq_up or sources[0] is not None
            hq_known = hq_t is not hq_up or sources[1] is not None
            if keep and index < end - 1 and lq_known and hq_known:
                kept.append((index, lq_t, hq_t, (sources[0] if lq_t is lq_up else None,
                                                 sources[1] if hq_t is hq_up else None)))
        outcomes.append((error, summary))
        if index < end - 1:
            events[index - start + 1].record()
    if not uploaded:  # nothing was uploaded, so every step failed
        lq_new, hq_new = lq, hq
    else:
        lq_new = lq if lq_t is lq_up else gpu.tensor_to_image(lq_t, 3)
        hq_new = hq if hq_t is hq_up else gpu.tensor_to_image(hq_t, 3)
    events[-1].record()
    events[-1].synchronize()
    steps = [
        StepResult(index, configs[index]["type"],
                   events[pos].elapsed_time(events[pos + 1]), error, summary)
        for pos, (index, (error, summary)) in enumerate(zip(range(start, end), outcomes))
    ]
    return lq_new, hq_new, steps, kept


def _run_pipeline(cache, source, configs, seed):
    _ensure_pipeline()
    from pipeline.process import FAILED_MODULES
    from pipeline.utils.registry import get_class

    t_run = time.perf_counter()
    gpu = _gpu_handoff()
    if gpu is not None and cache.device_budget is None:
        cache.device_budget = _device_budget()
    keys, start, entry = cache.begin(source, video_backend.detect(), configs, seed)
    steps = [StepResult(i, configs[i]["type"], 0.0, None, None, cached=True) for i in range(start)]
    resume = None  # a tensor entry whose GPU group the run goes on with
    if entry is None:
        lq = source.copy()
        hq = source.copy()
        lq_copy, hq_copy, hq_changed = None, source, False
    elif entry.sources is None:
        # Fresh copies, one array for both when the cold run had one, so the
        # remaining steps see exactly what they would have seen
        lq = entry.lq.copy(order="K")
        hq = lq if entry.hq is entry.lq else entry.hq.copy(order="K")
        lq_copy, hq_copy, hq_changed = entry.lq, entry.hq, entry.hq_changed
    else:
        # Inside a GPU group: it goes on from the cached tensors, and fresh
        # copies of the sources stand for the arrays the cold run's group
        # would pass on as they are (None where a step replaced the upload)
        resume = entry
        lq_copy, hq_copy = entry.sources
        lq = hq = None
        if lq_copy is not None:
            lq = lq_copy.copy(order="K")
        if hq_copy is lq_copy:
            hq = lq
        elif hq_copy is not None:
            hq = hq_copy.copy(order="K")
        hq_changed = entry.hq_changed
    # lq_copy and hq_copy are cached arrays equal to the working lq and hq, or
    # None when unknown (the source is one for hq, never handed to a step).
    # Entries share hq_copy while no step replaces HQ (steps write into lq,
    # never into an hq they return unchanged, unless hq is also their lq);
    # lq_copy holds only until the next step runs.
    # Only successful steps are cached. After a failed step nothing more is:
    # the later steps' outputs depend on the failure. Consecutive GPU steps
    # run as one group, cached as arrays at its end and as tensors before.
    caching = True
    index = start
    while index < len(configs) or resume is not None:
        hq_in, shared_in, sources = hq, lq is hq, (lq_copy, hq_copy)
        lq_copy = None
        if resume is not None or (gpu is not None and _is_rgb(lq) and _is_rgb(hq)):
            end = _group_end(configs, index, get_class)
        else:
            end = index
        if resume is not None or end - index > 1:
            lq, hq, done, kept = _run_group(gpu, configs, index, end, lq, hq, sources, seed,
                                            get_class, resume, caching and cache.device_budget > 0)
            for depth, lq_t, hq_t, held in kept:
                cache.store_tensors(keys[depth], depth, lq_t, hq_t, held,
                                    hq_changed or held[1] is None)
            resume = None
            replaced = hq is not hq_in
        else:
            end = index + 1
            hq_shape = hq.shape
            lq, hq, done = _run_step(configs[index], index, lq, hq, seed, get_class,
                                     FAILED_MODULES)
            replaced = hq is not hq_in or hq.shape != hq_shape
        steps.extend(done)
        hq_changed = hq_changed or replaced
        if replaced or shared_in:  # a write into lq may have reached hq
            hq_copy = None
        if any(step.error is not None for step in done):
            caching = False
        elif caching and done:
            lq_copy, hq_copy = cache.store(keys[end - 1], end - 1, lq, hq, hq_copy, hq_changed)
        index = end
    lq_u8 = _to_uint8(lq)
    # A failed step may have written into hq before it raised, so after one
    # hq_copy no longer counts as equal to hq
    hq_u8, hq_same = cache.display_hq(hq, hq_copy if caching else None, hq_changed)
    total_ms = (time.perf_counter() - t_run) * 1000.0
    return RunResult(lq, hq, steps, total_ms, seed, start, hq_changed, lq_u8, hq_u8, hq_same)


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
    seed, inside a group of GPU steps without leaving the device. A failing
    step is recorded in its StepResult and skipped; ``failed`` fires only
    when the engine itself breaks.
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
