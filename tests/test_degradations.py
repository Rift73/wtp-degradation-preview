"""
Tests for the degradation pipeline, the GUI schemas and the processing engine.

Runs under pytest, or standalone (pytest is not required):
    venv\\Scripts\\python.exe tests\\test_degradations.py
"""

import importlib.util
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
import types
import warnings

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import cv2
import numpy as np
from PySide6.QtCore import QEventLoop, QTimer
from PySide6.QtGui import QGuiApplication

import engine
from schema import (
    CATEGORY_COLORS, CATEGORY_OF, CODEC_QUALITY_PROFILES, SCHEMA_ORDER, SCHEMAS,
    build_config, summarize,
)

# The video codecs (h264, hevc, mpeg2, mpeg4, vp9) run through the system ffmpeg or PyAV
HAS_AV = importlib.util.find_spec("av") is not None

VIDEO_CODECS = {"h264", "hevc", "mpeg2", "mpeg4", "vp9"}


def _app():
    return QGuiApplication.instance() or QGuiApplication([])


def _image(size=128):
    """Synthetic RGB float32 test image: gradients, flat patches, texture."""
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    img = np.stack([xx / size, yy / size, 0.5 + 0.5 * np.sin(xx / 5)], -1)
    q = size // 4
    img[q:3 * q, q:3 * q] = [0.9, 0.2, 0.2]
    img[3 * q // 2:5 * q // 2, 3 * q // 2:5 * q // 2] = [0.1, 0.1, 0.9]
    img += np.random.default_rng(0).normal(0, 0.02, img.shape).astype(np.float32)
    return np.clip(img, 0, 1).astype(np.float32)


def _defaults(key, schemas=SCHEMAS):
    return {p["key"]: p["default"] for p in schemas[key]["params"]}


def _run_step(key, values, img):
    """Run one degradation directly; return a problem string, or None if fine."""
    engine._ensure_pipeline()
    from pipeline.utils.registry import get_class
    from pipeline.process import FAILED_MODULES

    cfg = build_config(key, values)
    cls = get_class(cfg["type"])
    if cls is None:
        tb = FAILED_MODULES.get(f"{cfg['type']}_degr", "not registered")
        return f"no class: {tb.strip().splitlines()[-1]}"
    np.random.seed(1)
    try:
        lq, hq = cls(cfg).run(img.copy(), img.copy())
    except Exception as exc:
        return f"{type(exc).__name__}: {str(exc).strip()[:200]}"
    for name, arr in (("lq", lq), ("hq", hq)):
        if arr.dtype != np.float32:
            return f"{name} dtype {arr.dtype}"
        if not np.isfinite(arr).all() or arr.min() < 0.0 or arr.max() > 1.0:
            return f"{name} outside [0, 1]: {arr.min()}..{arr.max()}"
    return None


def _check_cases(cases):
    img = _image()
    problems = []
    for key, tag, values in cases:
        alg = values.get("algorithm")
        if key == "compress" and alg in VIDEO_CODECS and not HAS_AV:
            print(f"  SKIP {key} {tag}: PyAV (av) is not installed")
            continue
        problem = _run_step(key, values, img)
        if problem:
            problems.append(f"{key} [{tag}]: {problem}")
    assert not problems, "\n".join(problems)


# ──────────────────────────────────────────────
# (a) every schema default runs
# ──────────────────────────────────────────────

def test_schema_defaults_run():
    _check_cases([(key, "default", _defaults(key)) for key in SCHEMA_ORDER])


# ──────────────────────────────────────────────
# (b) every option, every flipped bool, min/max of the first numeric param
# ──────────────────────────────────────────────

def _variant_cases():
    cases = []
    for key in SCHEMA_ORDER:
        base = _defaults(key)
        for p in SCHEMAS[key]["params"]:
            if p["type"] == "choice":
                for opt in p["options"]:
                    if opt == p["default"]:
                        continue
                    values = dict(base, **{p["key"]: opt})
                    if p["key"] == "algorithm" and opt in CODEC_QUALITY_PROFILES:
                        values["quality"] = CODEC_QUALITY_PROFILES[opt]["default"]
                    cases.append((key, f"{p['key']}={opt}", values))
            elif p["type"] == "bool":
                values = dict(base, **{p["key"]: not p["default"]})
                cases.append((key, f"{p['key']}={not p['default']}", values))
        first = next(p for p in SCHEMAS[key]["params"] if p["type"] in ("float", "int"))
        for bound in ("min", "max"):
            values = dict(base, **{first["key"]: first[bound]})
            cases.append((key, f"{first['key']}={bound}({first[bound]})", values))
    blur = _defaults("blur")
    for size in (4, 31):  # even sizes round up to odd
        values = dict(blur, filter="median", median_size=size)
        cases.append(("blur", f"median_size={size}", values))
    compress = _defaults("compress")
    for alg in sorted(VIDEO_CODECS):  # mpeg2/mpeg4 lack some chroma formats
        for sampling in ("444", "422"):
            values = dict(compress, algorithm=alg, video_sampling=sampling,
                          quality=CODEC_QUALITY_PROFILES[alg]["default"])
            cases.append(("compress", f"{alg} video_sampling={sampling}", values))
    return cases


def test_schema_variants_run():
    _check_cases(_variant_cases())


# ──────────────────────────────────────────────
# (c) build_config output unchanged against HEAD, except the blur fix
# ──────────────────────────────────────────────

def _head_schema():
    git = shutil.which("git")
    if git is None:
        return None
    proc = subprocess.run(
        [git, "show", "HEAD:schema.py"], cwd=ROOT, capture_output=True,
        text=True, encoding="utf-8",
    )
    if proc.returncode != 0:
        return None
    module = types.ModuleType("schema_head")
    exec(compile(proc.stdout, "HEAD:schema.py", "exec"), module.__dict__)
    return module


def test_build_config_matches_head():
    old = _head_schema()
    if old is None:
        print("  SKIP: git or HEAD:schema.py unavailable")
        return
    assert old.SCHEMA_ORDER == SCHEMA_ORDER
    problems = []
    for key in SCHEMA_ORDER:
        expected = old.build_config(key, _defaults(key, old.SCHEMAS))
        if key == "blur":
            expected["target_kernel"] = {"median": [5, 5]}
        got = build_config(key, _defaults(key))
        if repr(got) != repr(expected):
            problems.append(f"{key}:\n  HEAD {expected!r}\n  now  {got!r}")
    assert not problems, "\n".join(problems)


def test_schema_metadata():
    for key in SCHEMA_ORDER:
        assert CATEGORY_OF[key] in CATEGORY_COLORS, key
        for p in SCHEMAS[key]["params"]:
            assert p.get("help", "").strip(), f"{key}.{p['key']} has no help"
        variants = [values for k, _, values in _variant_cases() if k == key]
        for values in [_defaults(key)] + variants:
            text = summarize(key, values)
            assert text and len(text) <= 60, f"{key}: {text!r}"
    assert set(CATEGORY_OF) == set(SCHEMA_ORDER)
    assert summarize("blur", _defaults("blur")) == "gauss · σ 1.00"
    assert summarize("compress", _defaults("compress")) == "jpeg q80 4:2:0"
    assert summarize("resize", _defaults("resize")) == "lanczos ×4"
    assert summarize("noise", _defaults("noise")) == "gauss · 0.050 RGB"
    assert summarize("blur", {"filter": "median", "median_size": 4}) == "median · 5 px"


# ──────────────────────────────────────────────
# Engine
# ──────────────────────────────────────────────

def _run_engine(requests, timeout_s=120, eng=None):
    """Issue the requests back to back on eng (a new engine by default);
    return (results, busy events) once idle."""
    _app()
    if eng is None:
        eng = engine.PipelineEngine()
    results, busy, fatal = [], [], []
    loop = QEventLoop()
    slots = [
        (eng.result_ready, results.append),
        (eng.failed, fatal.append),
        (eng.busy_changed, busy.append),
        (eng.busy_changed, lambda is_busy: is_busy or loop.quit()),
    ]
    for signal, slot in slots:
        signal.connect(slot)
    for source, configs, seed in requests:
        eng.request_run(source, configs, seed)
    QTimer.singleShot(timeout_s * 1000, loop.quit)
    loop.exec()
    for signal, slot in slots:
        signal.disconnect(slot)
    assert not fatal, fatal[0]
    assert busy and busy[-1] is False and not eng.is_busy(), "engine did not finish"
    return results, busy


def _config(key, **changes):
    return build_config(key, dict(_defaults(key), **changes))


def test_engine_bad_step_is_skipped():
    img = _image()
    configs = [
        _config("noise"),
        {"type": "blur", "filter": ["no_such_filter"], "probability": 1.0},
        {"type": "no_such_type"},
        _config("saturation"),
    ]
    results, busy = _run_engine([(img, configs, 5)])
    assert busy == [True, False]
    assert len(results) == 1
    run = results[0]
    assert isinstance(run, engine.RunResult) and run.seed == 5
    assert [s.index for s in run.steps] == [0, 1, 2, 3]
    ok = [s for s in run.steps if s.error is None]
    assert [s.type_key for s in ok] == ["noise", "saturation"]
    bad = run.steps[1]
    assert "KeyError" in bad.error and bad.error_summary.startswith("KeyError")
    assert len(bad.error_summary) <= 120
    assert "Unknown" in run.steps[2].error_summary
    assert run.lq.shape == img.shape and not np.array_equal(run.lq, img)
    assert run.total_ms >= max(s.elapsed_ms for s in run.steps)


def test_engine_coalesces_requests():
    img = _image()
    configs = [_config("noise")]
    results, busy = _run_engine([(img, configs, 1), (img, configs, 2), (img, configs, 3)])
    assert busy == [True, False]
    assert [r.seed for r in results] == [1, 3]


# ──────────────────────────────────────────────
# Engine step cache and display arrays
# ──────────────────────────────────────────────

def _run_once(eng, source, configs, seed):
    results, _ = _run_engine([(source, configs, seed)], eng=eng)
    return results[0]


def _resume_configs(amount):
    # Step 1 leaves one array as both lq and hq, and step 2 (halo) writes into
    # its input: a resumed run must rebuild both of those exactly
    return [
        _config("blur"),
        dict(_config("noise"), lqhq=True),
        _config("halo", type_halo="unsharp_gray", amount=amount),
        _config("saturation"),
    ]


def test_engine_resume_matches_cold_run():
    img = _image()
    eng = engine.PipelineEngine()
    a = _run_once(eng, img, _resume_configs(1.0), 4)
    b = _run_once(eng, img, _resume_configs(2.0), 4)  # step 2 edited
    c = _run_once(engine.PipelineEngine(), img, _resume_configs(2.0), 4)
    assert all(s.error is None for s in a.steps + b.steps + c.steps)
    assert (a.cached_steps, b.cached_steps, c.cached_steps) == (0, 2, 0)
    assert [s.cached for s in b.steps] == [True, True, False, False]
    assert [s.elapsed_ms for s in b.steps[:2]] == [0.0, 0.0]
    assert not any(s.cached for s in a.steps + c.steps)
    assert np.array_equal(b.lq, c.lq) and np.array_equal(b.hq, c.hq)
    assert np.array_equal(b.lq_u8, c.lq_u8) and np.array_equal(b.hq_u8, c.hq_u8)
    assert b.hq_changed and c.hq_changed
    assert not np.array_equal(a.lq, b.lq), "the edit had no effect"
    again = _run_once(eng, img, _resume_configs(2.0), 4)
    assert again.cached_steps == 4 and all(s.cached for s in again.steps)
    assert np.array_equal(again.lq, c.lq) and np.array_equal(again.hq, c.hq)


def test_engine_new_source_clears_cache():
    img = _image()
    configs = [_config("blur"), _config("saturation")]
    eng = engine.PipelineEngine()
    assert _run_once(eng, img, configs, 1).cached_steps == 0
    assert _run_once(eng, img, configs, 1).cached_steps == 2
    assert _run_once(eng, img, configs, 2).cached_steps == 0  # the seed is part of the key
    # Equal pixels in a new array still count as a new image, and drop the old entries
    assert _run_once(eng, img.copy(), configs, 1).cached_steps == 0
    assert _run_once(eng, img, configs, 1).cached_steps == 0


def test_engine_failed_step_stops_caching():
    img = _image()
    configs = [_config("noise"), {"type": "no_such_type"}, _config("saturation")]
    eng = engine.PipelineEngine()
    first = _run_once(eng, img, configs, 1)
    second = _run_once(eng, img, configs, 1)
    assert [e.depth for e in eng._thread.cache.entries.values()] == [0]
    assert second.cached_steps == 1 and second.steps[1].error is not None
    assert not second.steps[2].cached and second.steps[2].error is None
    assert np.array_equal(first.lq, second.lq)


def test_engine_cache_eviction():
    img = _image()
    size = img.nbytes  # every step below keeps this size and leaves hq alone
    configs = [_config("blur"), _config("saturation"), _config("noise"), _config("blur", kernel=2.0)]
    budget = engine._CACHE_BUDGET_BYTES
    try:
        engine._CACHE_BUDGET_BYTES = 3 * size  # the shared hq and two lq
        eng = engine.PipelineEngine()
        cache = eng._thread.cache
        _run_once(eng, img, configs, 1)
        assert sorted(e.depth for e in cache.entries.values()) == [2, 3]
        # Editing the last step drops the stale old step 3 before the current step 2
        for kernel in (3.0, 4.0):
            edited = configs[:3] + [_config("blur", kernel=kernel)]
            assert _run_once(eng, img, edited, 1).cached_steps == 3
            assert sorted(e.depth for e in cache.entries.values()) == [2, 3]
        engine._CACHE_BUDGET_BYTES = size - 1  # no single output fits
        eng = engine.PipelineEngine()
        _run_once(eng, img, configs, 1)
        assert not eng._thread.cache.entries
        assert _run_once(eng, img, configs, 1).cached_steps == 0
    finally:
        engine._CACHE_BUDGET_BYTES = budget


def test_engine_display_arrays():
    img = _image()
    run = _run_once(engine.PipelineEngine(), img, [_config("blur")], 1)
    assert not run.hq_changed and run.hq_u8 is None and np.array_equal(run.hq, img)
    expected = (np.clip(run.lq, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    assert run.lq_u8.flags.c_contiguous and np.array_equal(run.lq_u8, expected)
    assert engine.numpy_to_qpixmap(run.lq_u8).toImage() == engine.numpy_to_qpixmap(run.lq).toImage()
    resized = _run_once(engine.PipelineEngine(), img, [_config("resize")], 1)
    assert resized.hq_changed and resized.hq_u8.dtype == np.uint8
    assert resized.hq_u8.shape == resized.hq.shape and resized.lq_u8.shape == resized.lq.shape


# ──────────────────────────────────────────────
# (e) seeding
# ──────────────────────────────────────────────

def _lq(configs, seed):
    results, _ = _run_engine([(_image(), configs, seed)])
    assert all(s.error is None for s in results[0].steps), results[0].steps
    return results[0].lq


PROCEDURAL_NOISES = ("perlin", "simplex", "opensimplex", "supersimplex")


def test_engine_seeding():
    same_seed = {}
    for name in PROCEDURAL_NOISES + ("gauss", "uniform"):
        configs = [_config("noise", type_noise=name)]
        a, b, c = _lq(configs, 7), _lq(configs, 7), _lq(configs, 8)
        assert np.array_equal(a, b), f"{name}: same seed gave different output"
        assert not np.array_equal(a, c), f"{name}: different seeds gave the same output"
        same_seed[name] = a
    names = list(same_seed)
    for i, name in enumerate(names):
        for other in names[i + 1:]:
            assert not np.array_equal(same_seed[name], same_seed[other]), (name, other)
    # Changing step 0 (identity either way, but different RNG use) must not
    # reshuffle step 1's noise.
    perlin = _config("noise", type_noise="perlin")
    first = [_config("noise", alpha=0.0), perlin]
    second = [_config("noise", alpha=0.0, type_noise="uniform"), perlin]
    assert np.array_equal(_lq(first, 3), _lq(second, 3)), "editing step 0 reshuffled step 1"


def test_fractal_noise_contract():
    from pipeline.process.procedural_noise import fractal_noise

    for name in PROCEDURAL_NOISES:
        for shape, octaves, frequency in (((48, 40), 1, 0.8), ((48, 40, 3), 3, 0.05),
                                          ((48, 40, 2), 2, 5.0)):
            noise = fractal_noise(shape, name, octaves, frequency, 0.4, seed=11)
            assert noise.shape == shape and noise.dtype == np.float32, (name, shape)
            assert np.abs(noise).max() <= 1.0 and np.abs(noise).mean() > 0.05, (name, shape)
            again = fractal_noise(shape, name, octaves, frequency, 0.4, seed=11)
            assert np.array_equal(noise, again), (name, shape)
            # The default torch path (CUDA when available) against the numpy reference;
            # without torch the default path is the reference itself
            reference = fractal_noise(shape, name, octaves, frequency, 0.4, seed=11,
                                      reference=True)
            assert np.abs(noise - reference).max() <= 2e-4, (name, shape)
            if len(shape) == 3:
                assert not np.array_equal(noise[..., 0], noise[..., 1]), (name, shape)
        # Channels are slices of one 3-D field at heights c * frequency, as in the
        # destroyer: near copies at low frequencies, unrelated at per-pixel ones
        for frequency, low, high in ((0.05, 0.8, 1.0), (0.8, -0.3, 0.3)):
            noise = fractal_noise((96, 96, 3), name, 1, frequency, 0.4, seed=11)
            r = np.corrcoef(noise[..., 0].ravel(), noise[..., 1].ravel())[0, 1]
            assert low <= r <= high, (name, frequency, r)


# ──────────────────────────────────────────────
# (f) image I/O
# ──────────────────────────────────────────────

def test_load_image_gray_png():
    gray8 = (np.arange(48 * 32, dtype=np.uint32).reshape(32, 48) % 256).astype(np.uint8)
    gray16 = gray8.astype(np.uint16) * 257
    with tempfile.TemporaryDirectory() as tmp:
        for name, data, scale in (("g8.png", gray8, 255.0), ("g16.png", gray16, 65535.0)):
            path = os.path.join(tmp, name)
            ok, buf = cv2.imencode(".png", data)
            assert ok
            buf.tofile(path)
            img = engine.load_image(path)
            assert img.shape == (32, 48, 3) and img.dtype == np.float32, (name, img.shape)
            assert np.array_equal(img[..., 0], img[..., 2])
            assert np.allclose(img[..., 1], data / scale, atol=1e-6)
        assert engine.load_image(os.path.join(tmp, "missing.png")) is None


def test_numpy_to_qpixmap_shapes():
    _app()
    rgb = _image(16)
    for img in (rgb, rgb[..., 0], rgb[..., :1]):
        pm = engine.numpy_to_qpixmap(img)
        assert (pm.width(), pm.height()) == (16, 16), img.shape


# ──────────────────────────────────────────────
# (g) GPU hand-off, GPU Beta noise, CUDA extension cache
# ──────────────────────────────────────────────

def _torch_cuda():
    """torch when a CUDA device is present, else None."""
    engine._ensure_pipeline()
    import torch
    return torch if torch.cuda.is_available() else None


def test_gpu_handoff_roundtrip():
    if _torch_cuda() is None:
        print("  SKIP: no CUDA device")
        return
    from optimized.gpu_degradations import image_to_tensor, tensor_to_image

    rgb = _image(32)
    for img in (rgb, rgb[..., 0].copy(), rgb[..., :1].copy()):
        tensor = image_to_tensor(img)
        channels = 1 if img.ndim == 2 else img.shape[2]
        assert tensor.is_cuda and tuple(tensor.shape) == (1, channels, 32, 32), img.shape
        out = tensor_to_image(tensor, img.ndim)
        assert out.dtype == np.float32 and out.flags.c_contiguous, img.shape
        assert np.array_equal(out, img), img.shape
    clamped = tensor_to_image(image_to_tensor(rgb * 3 - 1), 3)
    assert clamped.min() == 0.0 and clamped.max() == 1.0


def test_hf_noise_gpu_seeded():
    torch = _torch_cuda()
    if torch is None:
        print("  SKIP: no CUDA device")
        return
    from pipeline.process.hf_noise_degr import _beta_noise_gpu
    from pipeline.utils.registry import get_class

    img = _image(64)
    cfg = _config("hf_noise")

    def noisy_hq(seed):
        np.random.seed(seed)
        return get_class("hf_noise")(cfg).run(img.copy(), img.copy())[1]

    state = torch.cuda.get_rng_state()
    first, again, other = noisy_hq(4), noisy_hq(4), noisy_hq(5)
    assert torch.equal(state, torch.cuda.get_rng_state()), "global CUDA RNG state changed"
    assert np.array_equal(first, again), "same numpy seed gave different noise"
    assert not np.array_equal(first, other), "different seeds gave the same noise"
    assert first.dtype == np.float32 and first.min() >= 0.0 and first.max() <= 1.0
    assert 0.002 < float(np.abs(first - img).mean()) < 0.1
    for a, b in ((0.1, 0.1), (20.0, 0.1)):  # extreme Beta shapes stay finite
        for normalize in (True, False):
            hq = _beta_noise_gpu(img, a, b, 0.05, 3, normalize)
            assert np.isfinite(hq).all() and hq.shape == img.shape, (a, b, normalize)


def test_nlmeans_kernel_matches_fallback():
    if _torch_cuda() is None:
        print("  SKIP: no CUDA device")
        return
    from optimized.gpu_degradations import image_to_tensor
    from pipeline.process import hf_noise_degr

    # One channel never reaches the kernel (it writes three)
    gray = hf_noise_degr._nlmeans_gpu(_image(48)[..., 0].copy(), h=30.0)
    assert gray.shape == (48, 48) and gray.dtype == np.float32
    if hf_noise_degr.nlmeans_denoise_cuda is None:
        print("  SKIP kernel comparison: no cached build and no compiler")
        return
    x = image_to_tensor(_image(64))
    kernel = hf_noise_degr.nlmeans_denoise_cuda(x, 30.0, 7, 21)
    fallback = hf_noise_degr._nlmeans_core(x, 30.0, 7, 21)
    assert float((kernel - fallback).abs().max()) < 1e-5


def test_dither_kernel_matches_chainner_ext():
    if _torch_cuda() is None:
        print("  SKIP: no CUDA device")
        return
    from unittest import mock
    from chainner_ext import UniformQuantization, error_diffusion_dither
    from pipeline.constants import DITHERING_MAP
    from pipeline.process import dithering_degr
    from pipeline.utils.registry import get_class

    if not dithering_degr._HAS_CUDA_DIFFUSION:
        print("  SKIP: no cached build and no compiler")
        return
    problems = []
    # chainner_ext is stubbed out of the step, so every output below is the kernel's
    with mock.patch.object(dithering_degr, "error_diffusion_dither",
                           side_effect=AssertionError("the step used chainner_ext")):
        for height, width in ((128, 128), (257, 131)):
            yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
            ramp = 0.85 * xx / (width - 1) + 0.15 * yy / (height - 1)
            images = {
                "random": np.random.default_rng(height + width).random(
                    (height, width, 3), dtype=np.float32),
                "gradient": np.stack([ramp, 1 - ramp, 0.5 * ramp + 0.25], -1),
            }
            for kind, rgb in images.items():
                for img in (rgb, rgb[..., 0].copy()):  # gray rounds per scalar
                    for name, algorithm in DITHERING_MAP.items():
                        for levels in (2, 3, 7, 16, 64):
                            cfg = _config("dithering", dithering_type=name, color_ch=levels)
                            lq, _ = get_class("dithering")(cfg).run(img.copy(), img)
                            expected = np.squeeze(error_diffusion_dither(
                                img, UniformQuantization(levels), algorithm))
                            if not np.array_equal(lq.view(np.uint32), expected.view(np.uint32)):
                                problems.append(f"{name} {kind} {img.shape} levels {levels}")
    assert not problems, "\n".join(problems)


def test_dither_kernel_failure_falls_back():
    if _torch_cuda() is None:
        print("  SKIP: no CUDA device")
        return
    from unittest import mock
    from chainner_ext import UniformQuantization, error_diffusion_dither
    from pipeline.constants import DITHERING_MAP
    from pipeline.process import dithering_degr
    from pipeline.utils.registry import get_class

    img = _image(48)
    cfg = _config("dithering", dithering_type="burkes", color_ch=4)
    expected = error_diffusion_dither(img, UniformQuantization(4), DITHERING_MAP["burkes"])
    with mock.patch.object(dithering_degr, "_HAS_CUDA_DIFFUSION", True), \
            mock.patch.object(dithering_degr, "_DIFFUSION_FALLBACK_LOGGED", False), \
            mock.patch.object(dithering_degr, "error_diffusion_dither_cuda", create=True,
                              side_effect=RuntimeError("no kernel")), \
            mock.patch.object(dithering_degr.logging, "warning") as warning:
        for _ in range(2):
            lq, _ = get_class("dithering")(cfg).run(img.copy(), img)
            assert np.array_equal(lq, expected)
    assert warning.call_count == 1, "the fallback must be logged once"


def test_cuda_ext_skips_build_without_compiler():
    if sys.platform != "win32":
        print("  SKIP: the cl.exe check is Windows-only")
        return
    from unittest import mock
    from optimized import cuda_ext

    name, flags = "probe_ext", ["--use_fast_math"]
    sources = [os.path.join(ROOT, "optimized", "csrc", "iir_trailing.cpp")]
    with tempfile.TemporaryDirectory() as tmp, \
            mock.patch.dict(os.environ, {"LOCALAPPDATA": tmp}), \
            mock.patch.object(cuda_ext.logger, "info") as info:
        with mock.patch.object(cuda_ext.shutil, "which", return_value=r"C:\cl.exe"):
            cuda_ext.check_buildable(name, sources, flags, "the fallback")
        with mock.patch.object(cuda_ext.shutil, "which", return_value=None):
            try:
                cuda_ext.check_buildable(name, sources, flags, "the fallback")
            except ImportError:
                pass
            else:
                raise AssertionError("no cached build and no cl.exe must raise ImportError")
            assert info.call_count == 1 and "the fallback" in info.call_args.args
            build_dir, module_path = cuda_ext._build_paths(
                name, sources, flags + cuda_ext._HOST_CUDA_CFLAGS,
            )
            assert not os.path.exists(build_dir), "no build may be attempted"
            os.makedirs(build_dir)
            open(module_path, "wb").close()
            cuda_ext.check_buildable(name, sources, flags, "the fallback")  # cached: no compiler needed


# ──────────────────────────────────────────────
# (h) video backend: located ffmpeg, then PATH, then PyAV
# ──────────────────────────────────────────────

def test_video_backend_detect():
    import video_backend

    try:
        backend = video_backend.restore_ffmpeg({})
        system = shutil.which("ffmpeg")
        if system is not None:
            assert backend.kind == "ffmpeg" and os.path.samefile(backend.path, system), backend
            assert backend.version and "mpeg4" in backend.encoders, backend
            assert video_backend.detect() is backend, "detect() must be cached"
            assert video_backend.pixel_formats(backend.path, "mpeg4") == ("yuv420p",)
        if HAS_AV:
            video_backend.set_preference("pyav")
            assert video_backend.detect().kind == "pyav"
            if system is not None:
                video_backend.set_preference("ffmpeg")
                assert video_backend.detect().kind == "ffmpeg"
        with tempfile.TemporaryDirectory() as tmp:
            fake = os.path.join(tmp, "ffmpeg.exe")
            with open(fake, "wb") as f:
                f.write(b"not a program")
            cfg = {}
            before = video_backend.detect()
            assert not video_backend.register_ffmpeg(fake, cfg) and cfg == {}
            assert video_backend.detect() == before
            # A saved path that no longer works falls back to PATH / PyAV
            assert video_backend.restore_ffmpeg({"ffmpeg_path": fake}) == video_backend.restore_ffmpeg({})
        try:
            video_backend.set_preference("gstreamer")
        except ValueError:
            pass
        else:
            raise AssertionError("an unknown backend must raise ValueError")
    finally:
        video_backend.restore_ffmpeg({})


def test_compress_system_ffmpeg_matches_pyav():
    import video_backend
    from pipeline.utils.registry import get_class

    if shutil.which("ffmpeg") is None or not HAS_AV:
        print("  SKIP: needs an ffmpeg on PATH and PyAV")
        return
    engine._ensure_pipeline()
    img = _image(130)[:129, :127]  # odd: padded for 4:2:0, cropped back
    try:
        for alg in sorted(VIDEO_CODECS):
            cfg = _config("compress", algorithm=alg, video_sampling="420",
                          quality=CODEC_QUALITY_PROFILES[alg]["default"])
            out = {}
            for kind in ("ffmpeg", "pyav"):
                video_backend.set_preference(kind)
                assert video_backend.detect().kind == kind
                np.random.seed(1)
                out[kind] = get_class("compress")(cfg).run(img.copy(), img.copy())[0]
                assert out[kind].shape == img.shape and out[kind].dtype == np.float32, (alg, kind)
                assert np.abs(out[kind] - img).mean() * 255 < 12, (alg, kind, "not a roundtrip")
            # Builds differ in their RGB -> 4:2:0 conversion: about one level on average
            diff = np.abs(out["ffmpeg"] - out["pyav"]).mean() * 255
            assert diff < 3, (alg, diff)
    finally:
        video_backend.restore_ffmpeg({})


def test_ffmpeg_error_reaches_the_step():
    import video_backend
    from pipeline.process import compress_degr

    path = shutil.which("ffmpeg")
    if path is None:
        print("  SKIP: no ffmpeg on PATH")
        return
    encode = [path, "-hide_banner", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
              "-s", "4x4", "-i", "pipe:", "-c:v", "no_such_encoder", "-f", "h264", "pipe:"]
    decode = [path, "-hide_banner", "-loglevel", "error", "-f", "h264", "-i", "pipe:",
              "-f", "rawvideo", "pipe:"]
    try:
        compress_degr._encode_decode(encode, decode, bytes(48), "no_such_encoder")
    except RuntimeError as exc:
        first = str(exc).splitlines()[0]
        assert first.startswith("ffmpeg no_such_encoder encode failed (exit "), first
        assert "no_such_encoder" in first.split("):", 1)[1], first  # ffmpeg's own words
    else:
        raise AssertionError("an unknown encoder must raise")
    assert video_backend.POPEN_FLAGS == (subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0)


# ──────────────────────────────────────────────
# (i) GPU-resident hand-off between consecutive GPU steps
# ──────────────────────────────────────────────

GPU_STEPS = ("banding", "filmgrain", "ghosting", "interlace", "lowpass", "ntsc", "overshoot",
             "rainbow", "scanline")


def _same_bits(a, b):
    return a.shape == b.shape and a.dtype == b.dtype == np.float32 and \
        np.array_equal(a.view(np.uint32), b.view(np.uint32))


def test_run_tensor_matches_run():
    if _torch_cuda() is None:
        print("  SKIP: no CUDA device")
        return
    from optimized.gpu_degradations import image_to_tensor, result_image
    from pipeline.utils.registry import get_class

    hq = _image(64)
    # In range, and beyond [0, 1] (an output a step leaves unchanged is not clamped)
    sources = (hq * 0.8 + 0.1, hq * 1.2 - 0.1)
    problems = []
    for key in GPU_STEPS:
        cls = get_class(key)
        variants = [_defaults(key)] + [v for k, _, v in _variant_cases() if k == key]
        for values in variants:
            cfg = build_config(key, values)
            for lq in sources:
                for seed in (1, 2):
                    engine._seed_step(seed, 0)
                    lq_a, hq_a = cls(cfg).run(lq.copy(), hq.copy())
                    engine._seed_step(seed, 0)
                    lq_in, hq_in = lq.copy(), hq.copy()
                    lq_t, hq_t = image_to_tensor(lq_in), image_to_tensor(hq_in)
                    lq_out, hq_out = cls(cfg).run_tensor(lq_t, hq_t)
                    if hq_out is not hq_t or lq_out.device != lq_t.device:
                        problems.append(f"{key} {values}: hq replaced or device changed")
                    lq_b = result_image(lq_in, lq_t, lq_out)
                    if not (_same_bits(lq_a, lq_b) and _same_bits(hq_a, hq)):
                        problems.append(f"{key} {values} seed {seed} range "
                                        f"{lq.min():.1f}..{lq.max():.1f}")
    assert not problems, "\n".join(problems)


def _one_by_one(source, configs, seed):
    """lq and hq from running each step's run() in turn, seeded as the engine seeds."""
    from pipeline.utils.registry import get_class

    lq, hq = source.copy(), source.copy()
    for index, config in enumerate(configs):
        engine._seed_step(seed, index)
        lq, hq = get_class(config["type"])(config).run(lq, hq)
    return lq, hq


def _cached_depths(eng):
    return sorted(entry.depth for entry in eng._thread.cache.entries.values())


def _gpu_chain():
    # CPU, three GPU steps (one group), CPU. Overshoot feeds lowpass: FFT on
    # FFT output, where a hand-off that skipped the layout copy changes bits
    return [_config("blur"), _config("overshoot"), _config("lowpass"), _config("filmgrain"),
            _config("saturation")]


def test_engine_gpu_group_matches_steps():
    if _torch_cuda() is None:
        print("  SKIP: no CUDA device")
        return
    img = _image()
    cases = (
        (img, _gpu_chain(), [0, 3, 4]),
        # Beyond [0, 1], led by a step that leaves lq unchanged: the group must
        # pass the unclamped upload on, as run() does
        (img * 1.2 - 0.1, [_config("scanline", strength=0.0)] + _gpu_chain()[1:4], [3]),
    )
    for source, configs, depths in cases:
        eng = engine.PipelineEngine()
        run = _run_once(eng, source, configs, 6)
        assert all(s.error is None for s in run.steps), run.steps
        lq, hq = _one_by_one(source, configs, 6)
        assert _same_bits(run.lq, lq) and _same_bits(run.hq, hq), configs
        assert _cached_depths(eng) == depths, "a group is cached at its end only"
        assert [s.index for s in run.steps] == list(range(len(configs)))
        assert all(s.elapsed_ms > 0 for s in run.steps), [s.elapsed_ms for s in run.steps]
        assert not run.hq_changed and run.hq_u8 is None


def test_engine_gpu_group_resume_matches_cold_run():
    if _torch_cuda() is None:
        print("  SKIP: no CUDA device")
        return
    img = _image()
    head = [_config("overshoot"), _config("filmgrain")]
    tail = [_config("scanline"), _config("banding")]
    eng = engine.PipelineEngine()
    _run_once(eng, img, head, 9)
    assert _cached_depths(eng) == [1]
    # Resumes after the old group's end; the rest runs as a new group, while
    # the cold run takes all four steps as one group
    resumed = _run_once(eng, img, head + tail, 9)
    cold = _run_once(engine.PipelineEngine(), img, head + tail, 9)
    assert (resumed.cached_steps, cold.cached_steps) == (2, 0)
    assert _same_bits(resumed.lq, cold.lq) and _same_bits(resumed.hq, cold.hq)
    # An edit inside a group re-runs from the group's start
    eng = engine.PipelineEngine()
    _run_once(eng, img, head + tail, 9)
    assert _cached_depths(eng) == [3]
    edited = head + [_config("scanline", strength=0.9), tail[1]]
    rerun = _run_once(eng, img, edited, 9)
    assert rerun.cached_steps == 0
    assert _same_bits(rerun.lq, _run_once(engine.PipelineEngine(), img, edited, 9).lq)
    # An edit after a group resumes at the group's end
    eng = engine.PipelineEngine()
    chain = _gpu_chain()
    _run_once(eng, img, chain, 9)
    later = chain[:4] + [_config("saturation", rand=1.5)]
    resumed = _run_once(eng, img, later, 9)
    assert resumed.cached_steps == 4
    assert _same_bits(resumed.lq, _one_by_one(img, later, 9)[0])


# ──────────────────────────────────────────────
# Standalone runner
# ──────────────────────────────────────────────

def _main():
    logging.basicConfig(level=logging.CRITICAL)
    warnings.filterwarnings("ignore")
    tests = [(n, f) for n, f in list(globals().items()) if n.startswith("test_")]
    failed = 0
    for name, fn in tests:
        t0 = time.perf_counter()
        try:
            fn()
        except Exception:
            failed += 1
            print(f"FAIL {name}\n{traceback.format_exc()}")
        else:
            print(f"PASS {name} ({time.perf_counter() - t0:.1f} s)")
    print(f"\n{len(tests) - failed} passed, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(_main())
