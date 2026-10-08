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

# The video codecs (h264, hevc, mpeg2, mpeg4, vp9) run through PyAV
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

def _run_engine(requests, timeout_s=120):
    """Issue the requests back to back; return (results, busy events) once idle."""
    _app()
    eng = engine.PipelineEngine()
    results, busy, fatal = [], [], []
    loop = QEventLoop()
    eng.result_ready.connect(results.append)
    eng.failed.connect(fatal.append)
    eng.busy_changed.connect(busy.append)
    eng.busy_changed.connect(lambda is_busy: is_busy or loop.quit())
    for source, configs, seed in requests:
        eng.request_run(source, configs, seed)
    QTimer.singleShot(timeout_s * 1000, loop.quit)
    loop.exec()
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
# (e) seeding
# ──────────────────────────────────────────────

def _lq(configs, seed):
    results, _ = _run_engine([(_image(), configs, seed)])
    assert all(s.error is None for s in results[0].steps), results[0].steps
    return results[0].lq


def test_engine_seeding():
    configs = [_config("noise"), _config("noise", type_noise="perlin")]
    a, b, c = _lq(configs, 7), _lq(configs, 7), _lq(configs, 8)
    assert np.array_equal(a, b), "same seed gave different output"
    assert not np.array_equal(a, c), "different seeds gave the same output"
    # Changing step 0 (identity either way, but different RNG use) must not
    # reshuffle step 1's noise.
    first = [_config("noise", alpha=0.0), _config("noise")]
    second = [_config("noise", alpha=0.0, type_noise="uniform"), _config("noise")]
    assert np.array_equal(_lq(first, 3), _lq(second, 3)), "editing step 0 reshuffled step 1"


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
