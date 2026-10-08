# WTP Degradation Preview

A PySide6 GUI for real-time preview of the `wtp_dataset_destroyer` degradation pipeline used in super-resolution dataset creation. Build a chain of degradations, see the low-quality (LQ) result next to your high-quality (HQ) image as you move the sliders, and tune the settings before committing to a full training dataset build.

## Features

- **Searchable Add step** -- type to filter the degradation list, grouped by category
- **Step cards** -- drag to reorder, enable/disable, duplicate, reset, remove; each card shows a one-line summary of its settings plus a status dot with the step's run time (an error shows its message on hover and does not stop the rest of the pipeline)
- **24 degradation types** with per-parameter sliders and tooltips:

  | Category | Degradations |
  |----------|-------------|
  | Blur/Filter | blur, lowpass, resize |
  | Noise/Grain | noise, hf_noise, filmgrain |
  | Compression | compress (JPEG, WebP, H.264, HEVC, MPEG-2, MPEG-4, VP9) |
  | Color | color, saturation, banding |
  | Pattern | dithering, screentone, scanline, sin, subsampling |
  | Edge/Sharpen | halo, overshoot, canny |
  | Geometric | pixelate, interlace, ghosting, shift |
  | Video Signal | ntsc, rainbow |

- **Presets** -- save and load a pipeline as JSON; the last pipeline is restored on start
- **Copy as HCL** -- copies the pipeline as `degradation { ... }` blocks in the `wtp_dataset_destroyer` config syntax (steps that exist only in this GUI, such as NTSC or film grain, are emitted too; the destroyer only accepts the ones it implements)
- **Three view modes** -- Wipe (slider), Side-by-side, and A/B flip
- **True-size LQ view** -- an LQ result smaller than the HQ is drawn nearest-neighbour at HQ size, with its real dimensions shown in the overlay
- **Seed and Re-roll** -- a session seed makes edits repeatable; Re-roll draws a new seed for fresh noise
- **Zoom and pan** -- Fit, 100%, zoom in/out, Ctrl+wheel, drag to pan, double-click to fit
- **Optional CUDA** acceleration for IIR trailing and NLMeans, with JIT compilation
- **Video codecs on your FFmpeg** -- a header chip shows the video backend in use and switches it (see below)

## Keyboard shortcuts

| Key | Action |
|-----|--------|
| Ctrl+O | Open image (drag and drop also works) |
| Space | Toggle A/B |
| R | Re-roll the seed |
| Ctrl+S | Save preset |
| Ctrl+L | Load preset |
| F | Fit to window |
| 1 | Zoom to 100% |
| Ctrl+= / Ctrl+- | Zoom in / out |
| Ctrl+Shift+C | Copy the pipeline as HCL |

## Requirements

- Python 3.14 (64-bit, Windows)
- Windows (bat scripts included; the Python code itself is cross-platform)
- FFmpeg (optional; any static or shared build with libx264, libx265 and libvpx; PyAV's bundled FFmpeg is the fallback)
- CUDA + Visual Studio Build Tools (optional, for GPU-accelerated degradations)

`chainner_ext` is vendored under `vendor/chainner_ext/` as a prebuilt C module from the author's chaiNNer-C project, so it is not installed from PyPI.

## Install

```
install.bat
```

This creates a virtual environment with the `py -3.14` launcher and installs all dependencies from `requirements.txt`.

## Usage

```
run.bat
```

Open an image with **Open image** (Ctrl+O), add steps with **Add step**, and adjust the sliders. Window geometry, splitter position, last folder, FFmpeg path, video backend and the current pipeline are remembered in `config.json`.

The video codecs of the compress step (H.264, HEVC, MPEG-2, MPEG-4, VP9) run on the first backend that works: the ffmpeg executable you picked with **Locate ffmpeg…**, then the `ffmpeg` on PATH, then PyAV's bundled FFmpeg (installed with the requirements; runs in-process). The header chip shows which one is in use, for example `Video: ffmpeg 8.1 (system)` or `Video: built-in (PyAV)`, and its menu locates an ffmpeg or switches between the two; choosing the built-in one tries PyAV first. A codec whose encoder the system ffmpeg lacks runs on PyAV. FFmpeg builds convert RGB to 4:2:0 slightly differently, so the two backends can differ along sharp colour edges (about one level in 255 on average).

## Project Structure

```
main.pyw            # Application entry point (header, shortcuts, config.json)
engine.py           # Background pipeline runner, seeding, per-step timing and errors
video_backend.py    # Video codec backend: located ffmpeg, then ffmpeg on PATH, then PyAV
widgets.py          # Pipeline panel, step cards, parameter editors
comparison.py       # Preview: Wipe / Side-by-side / A/B views, zoom and pan
presets.py          # Preset JSON save/load and HCL export
schema.py           # Degradation parameter schemas and config builders
style.qss           # Qt stylesheet (dark theme)
pipeline/
  logic/            # Pipeline orchestration
  process/          # Individual degradation implementations
  utils/            # Registry, random utilities
optimized/
  csrc/             # CUDA kernels (IIR trailing, NLMeans)
  gpu_degradations.py
tests/              # Degradation tests (every schema default and option)
vendor/
  chainner_ext/     # Prebuilt chainner_ext C module (chaiNNer-C)
docs/               # Design notes for the UI
```

## License

[MIT](LICENSE) for this project. The vendored `vendor/chainner_ext/` module is the author's C build from chaiNNer-C and contains GPL-3.0 code; see `vendor/PROVENANCE.md` and the licence files next to it. The simplex generator in `pipeline/process/procedural_noise.py` is a port of chaiNNer's GPL-3.0 numpy implementation (credited in its docstring).
