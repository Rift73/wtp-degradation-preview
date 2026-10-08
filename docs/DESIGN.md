# WTP Degradation Preview — UI/UX overhaul (2026-10-09)

Owner brief: make the GUI work properly and significantly improve UI/UX; fix obvious errors; refactor as
needed; no performance work. Budget 2 h. Branch `ui-overhaul`, not pushed (push is owner-only).

## Verified defects (from running every schema default + option, see `tests/test_degradations.py`)

| # | Defect | Fix (lane) |
|---|--------|------------|
| 1 | `compress_degr` never loads: torchcodec raises `RuntimeError` on DLL load, guard catches only `ImportError`; JPEG/WebP/video all missing from GUI | catch `Exception`; `os.add_dll_directory(ffmpeg bin)` before import so torchcodec can load; PyAV fallback verified for all 7 codecs (C) |
| 2 | Median blur crashes: float sigma passed, `medianBlur` needs odd int ≥ 3 | dedicated `median_size` int param (3–31, odd) via `target_kernel` (C) |
| 3 | Halo crashes at sigma 0 | schema min 0.1 (C) |
| 4 | GUI copy of `resize_degr` secretly upsamples LQ back to HQ size (not in destroyer); preview + dims lie | remove hack; preview draws true LQ scaled nearest-neighbour to HQ rect, overlay shows real dims (C + B) |
| 5 | Non-modal error dialog on every slider tick while a step fails | per-step status on the card, one status-bar line, no dialogs (A + C + main) |
| 6 | Grayscale input crashes film grain | convert to RGB on load (C) |
| 7 | numpy never seeded: every edit re-rolls noise; Re-roll ≡ any edit | session seed, per-step seed = f(seed, step index), Re-roll bumps seed (C) |
| 8 | Labels truncated at 110 px; disabled-card style never applies; drop index counts hidden label; "Original" shown even when HQ was modified | new cards/layout (A); HQ/LQ naming (B) |
| 9 | README advertises logiop; GUI has no schema | README fix (D); logiop stays out of scope |

## Target UX

Left: pipeline panel (default 440 px). Header row: **Add step** (searchable popup grouped by category), Load, Save,
Copy HCL, Clear. Cards: drag handle · enable toggle · title · collapsed one-line summary · status dot (idle / ok + ms /
error, message in tooltip) · actions (duplicate, reset, remove). Params: label row above control, tooltips from schema
`help`, sliders+spin as now, codec-aware quality ranges as now. Empty state with a hint.

Right: preview with a slim toolbar: view mode **Wipe | Side-by-side | A/B** (Space toggles A/B), zoom **Fit · 100% · − · +**
with a persistent % readout, labels **HQ** / **LQ** with true dims; LQ smaller than HQ is drawn nearest-neighbour at HQ
size, overlay reads e.g. `LQ 128×128 (shown ×4)`. Busy overlay while processing. Ctrl+wheel zoom, drag to pan, double-click fit.

Top header: app title · **Open image** (Ctrl+O, drag-drop still works) · seed spin + **Re-roll** (R) · FFmpeg locator (only if missing).
Status bar: last run `n steps · total ms · seed` or the first error line. `config.json` remembers geometry, splitter,
last dir, ffmpeg path, last pipeline state (autosave on exit, restore on start).

## Interface contracts (lanes work in parallel against these)

**C — `engine.py`, `pipeline/`, `schema.py`, `tests/`**
```python
@dataclass
class StepResult: index: int; type_key: str; elapsed_ms: float; error: str | None; error_summary: str | None
@dataclass
class RunResult: lq: np.ndarray; hq: np.ndarray; steps: list[StepResult]; total_ms: float; seed: int
class PipelineEngine(QObject):
    result_ready = Signal(object)   # RunResult (per-step errors inside, run still completes with that step skipped)
    failed = Signal(str)            # fatal traceback (engine itself broke)
    busy_changed = Signal(bool)
    def request_run(self, source: np.ndarray, configs: list[dict], seed: int) -> None  # coalesces while busy
    def is_busy(self) -> bool
def load_image(path) -> np.ndarray | None      # float32 RGB [0,1]; gray->RGB; alpha dropped; 8/16-bit
def numpy_to_qpixmap(img: np.ndarray) -> QPixmap
def ffmpeg_available() -> bool; def restore_ffmpeg(cfg: dict) -> bool; def register_ffmpeg(path: str, cfg: dict) -> None
```
`schema.py` additions: each param may carry `"help": str`; `CATEGORY_OF: dict[key, str]` (Blur/Filter, Noise/Grain,
Compression, Color, Pattern, Edge/Sharpen, Geometric, Video signal); `CATEGORY_COLORS` keyed by category name;
`summarize(schema_key, values) -> str` (≤ 60 chars, e.g. `gauss · σ 1.00`). Config output of `build_config` unchanged
except the blur/halo fixes.

**B — `comparison.py`**
```python
class ComparisonView(QWidget):     # toolbar + canvas; owns view mode and zoom
    zoom_changed = Signal(float)
    def set_images(self, hq: QPixmap | None, lq: QPixmap | None, hq_dims: str, lq_dims: str) -> None
    def set_busy(self, busy: bool) -> None
    def clear(self) -> None        # back to empty state
    def set_view_mode(self, mode: str) -> None   # "wipe" | "side" | "ab"
    def zoom_fit(self) / zoom_100(self) / zoom_in(self) / zoom_out(self)
```

**A — `widgets.py`, `presets.py`**
```python
class PipelinePanel(QWidget):
    changed = Signal()                       # any config/order/enable change (main debounces)
    def get_configs(self) -> list[dict]      # enabled steps only, in order (build_config output)
    def get_state(self) -> list[dict]        # all steps: {"type", "enabled", "collapsed", "values"}
    def set_state(self, state: list[dict]) -> None
    def set_step_results(self, steps: list[StepResult]) -> None   # indexes refer to get_configs() order
    def clear_step_results(self) -> None
# presets.py
def save_preset(path: str, state: list[dict]) -> None; def load_preset(path: str) -> list[dict]
def to_hcl(configs: list[dict]) -> str     # one `degradation { ... }` block per config, destroyer syntax
```

**D — `style.qss`, `README.md`** — object names everyone must use: `#headerBar #appTitle #seedSpin #pipelinePanel
#panelToolbar #stepCard #stepCard[state="ok"|"error"|"disabled"] #cardHeader #dragHandle #cardTitle #cardSummary
#statusDot #cardActions #iconBtn #deleteBtn #paramLabel #addStepBtn #addMenu #previewToolbar #toolBtn
#toolBtn:checked #zoomReadout #emptyHint`. Tokens: bg `#0F1115`, panel `#161A20`, card `#1C2129`, card-hover
`#222833`, border `#2A3039`, text `#E6E8EB`, muted `#8B93A1`, accent `#5B9DF5`, danger `#E05555`, warn `#D4A843`,
ok `#4CAF6A`; radius 6; font Segoe UI Variable 10 pt.

**E — `venv314`, `vendor/chainner_ext/`, `requirements.txt`, `install.bat`, `run.bat`** — CPython 3.14 venv with every dependency; vendored C module; install/run scripts for GitHub users (`py -3.14`).

**main.pyw** (orchestrator): wires the three; debounce 120 ms; shortcuts; config.json persistence.

## Verification
Per lane: import smoke + `tests/test_degradations.py` where relevant. Final gate: full test file, offscreen launch smoke
(load image, add every step, run, no exceptions), one real launch for the owner.

## Decisions log (one line each)
- Python 3.14 upgrade: owner directive ("just upgrade to 3.14"). Lane E builds `venv314`; `chainner_ext` is the owner's C rewrite from chaiNNer-C (`backend/src/chainner_ext`, CPython 3.14, bit-exact vs 0.3.10 per its conformance manifest), vendored under `vendor/chainner_ext/`. Old venv kept as `venv_old312` until the owner deletes it.
- True-size LQ (nearest-neighbour display) replaces the fake upsample. HQ/LQ naming replaces Original/Degraded.
- logiop stays unexposed; README claim removed.
- 3.14 blocker: pepeline 0.3.14 and dataset-support 0.1.4 have no cp314 wheels (and no pyo3-3.14 source path). Ruling: port to pepeline 1.x (abi3) and replace dataset-support's two functions with numpy (branch `lane-e-pepeline1`, merged after Lane C); the noise types opensimplex/simplex are dropped because 1.x lacks them.
- Single working tree, disjoint files per lane (venv is in-tree; worktrees would lack it).

- pepeline 1.x `noise()` has no seed argument: perlin/opensimplex/supersimplex re-roll on every run even with a fixed seed (gauss/uniform/salt types are reproducible). Accepted for this release; follow-up: numpy fractal noise with a seed.
- Channel shift YUV mode: pepeline's BT.2020 YCbCr->RGB has a wrong green coefficient (tint up to 0.17 at zero shift); replaced by a numpy inverse in `shift_degr.py`, verified identity at zero shift.
- OWNER QUEUE: `vendor/chainner_ext/chainner_native.dll` contains GPL-3.0 code ported from chaiNNer while this repo is MIT; publishing it this way is the owner's call (see `vendor/PROVENANCE.md`). Also: install official CPython 3.14 and recreate the venv so it no longer depends on chaiNNer-C's runtime folder; delete `venv_old312` (about 5 GB) when satisfied.

- Owner 2026-10-09 (second pass): publish with the GPL'd vendored module as is; owner deletes `venv_old312`; the venv must not depend on chaiNNer-C's runtime -> official CPython 3.14.8 installed per-user (PSF-signed installer, verified) and the venv recreated by `install.bat`. `py` now defaults to 3.14 on this machine.
- torchcodec removed from compress and requirements: it can never load here (PyAV's FFmpeg DLLs are name-mangled, the FFmpeg build is static) and PyAV covers all seven codecs in-process, which is chaiNNer's own child-process approach done in-process.
- install.bat/run.bat were LF-ended and cmd misparsed them; now CRLF with ASCII comments, pinned by `.gitattributes`.

- Seeded procedural noise (Lane F): numpy perlin (own code) + a port of chaiNNer's GPL-3.0 simplex; destroyer's cycles-per-pixel frequency kept (0.8 = grain, 0.02-0.1 = blobs) so the preview matches the copied HCL; opensimplex/supersimplex render with the simplex generator on their own seed streams. The GPL-derived source falls under the owner's "publish as is" ruling; noted in README.

## STATUS
- [ ] A panel · [ ] B preview · [ ] C engine+pipeline+tests · [ ] D style+README · [ ] main.pyw · [ ] final gate · [ ] commit

## Performance pass (2026-10-09, owner: "obvious optimisation", under 2 h)
Measured with the profile harness (median of 3 after warm-up; `docs/perf-before.txt` vs `docs/perf-after.txt`):
- all-24-defaults chain: 2048² 1334 -> 793 ms; 1024² 326 -> 202 ms. A slider edit on step k now re-runs only steps k..n (engine prefix cache, bit-identical to a cold run, 1.5 GB budget): last-step edit at 2048² 670 -> 29 ms.
- noise: simplex 1857 -> 45 ms, perlin 488 -> 30, gauss 204 -> 29 (torch generators, exact match to the numpy reference; float32 block draws); Y/UV modes 524/935 -> 37/58.
- GPU steps 33 -> 18-20 ms per profile row (6 ms step cost; the rest is the harness's own copies); one shared upload/download helper, outputs byte-identical.
- hf_noise 316 -> 23 ms (Beta sampling on the GPU, seeded). NLMeans CUDA kernel had NEVER compiled (CUDA 13 headers need `/Zc:preprocessor` under MSVC); fixed, builds cached under `%LOCALAPPDATA%\wtp_preview	orch_ext`, second launch 8.5 s -> 0.14 s; without MSVC the fallback is now silent.
- UI: uint8 conversion moved into the worker; the HQ pixmap is reused when HQ's bytes did not change.
- Decisions: no pinned-memory downloads (2-4 ms per step for up to 1 GB page-locked RAM); riemersma, codecs and GPU-resident chains left alone.

### CUDA dithering in the GUI (2026-10-09, kernel shared with the traiNNer fork)
Exact error-diffusion and riemersma kernels (bit-identical to chainner_ext, 320 + 225 cases on Windows), built on
first use through `cuda_ext` (Windows needs `#undef small`: the SDK's rpcndr.h defines it as `char`). Single image,
8 levels, median of 50 (`cpu` = chainner_ext, `step` = kernel incl. upload/download):

| size | floydsteinberg | stucki | sierra | riemersma |
|---|---|---|---|---|
| 1024² | 17.5 → 4.1 ms | 32.8 → 4.5 | 29.5 → 4.5 | 59.5 → 37.8 |
| 2048² | 78.0 → 14.7 ms | 125.5 → 16.3 | 116.0 → 16.3 | 239.1 → 213.4 |

Riemersma goes to the kernel in the GUI too (on Windows chainner_ext's riemersma is slower than on Linux, so the
single-thread kernel still wins 1.1-1.6x); in traiNNer it is routed to the kernel from batch 2 upward.
