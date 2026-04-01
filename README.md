# WTP Degradation Preview

A PySide6 GUI for real-time preview of image degradation pipelines used in super-resolution dataset creation. Lets you visually fine-tune degradation settings before committing to a full training dataset build.

## Features

- **Before/after comparison slider** with Catmull-Rom interpolation zoom and pan
- **Drag-and-drop pipeline ordering** -- reorder degradation stages by dragging
- **25+ degradation types** with per-parameter sliders:

  | Category | Degradations |
  |----------|-------------|
  | Blur/Filter | blur, lowpass, resize |
  | Noise/Grain | noise, hf_noise, filmgrain |
  | Compression | compress (JPEG, WebP, AVIF, HEIC) |
  | Color | color, saturation, banding |
  | Pattern | dithering, screentone, scanline, sin, subsampling |
  | Edge/Sharpen | halo, overshoot, canny |
  | Geometric | pixelate, interlace, ghosting, shift |
  | Video Signal | ntsc, rainbow |
  | Other | logiop |

- **Category-colored cards** for quick visual identification of degradation types
- **CUDA-accelerated** optional degradations (IIR trailing, NLMeans) with JIT compilation
- **FFmpeg auto-detection** with manual path configuration fallback

## Requirements

- Python 3.10+
- Windows (bat scripts included; the Python code itself is cross-platform)
- FFmpeg (optional, for video-based degradations; must use shared build)
- CUDA + Visual Studio Build Tools (optional, for GPU-accelerated degradations)

## Install

```
install.bat
```

This creates a virtual environment and installs all dependencies from `requirements.txt`.

## Usage

```
run.bat
```

## Project Structure

```
main.pyw            # Application entry point
widgets.py          # Pipeline panel, parameter editors, degradation cards
comparison.py       # Before/after comparison slider widget
schema.py           # Degradation parameter schemas and config builders
style.qss           # Qt stylesheet (dark theme)
pipeline/
  logic/            # Pipeline orchestration
  process/          # Individual degradation implementations (25+ types)
  utils/            # Registry, random utilities
optimized/
  csrc/             # CUDA kernels (IIR trailing, NLMeans)
  gpu_degradations.py
```

## License

[MIT](LICENSE)
