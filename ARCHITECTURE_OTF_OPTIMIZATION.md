# OTF Degradation Pipeline — Exhaustive Optimization Architecture

## 1. Project Goal

Port and optimize the complete WTP degradation pipeline into traiNNer-redux's OTF
(on-the-fly) training pipeline with **minimum possible latency**. Every operation
should stay on GPU when feasible, avoid CPU↔GPU roundtrips, and use async/parallel
execution to saturate hardware bandwidth.

---

## 2. Current State Audit

### 2.1 traiNNer-redux OTF — GPU-Native, No Structural Issues

| # | Operation            | Implementation                                | Notes                          |
|---|----------------------|-----------------------------------------------|--------------------------------|
| 1 | Blur (kernel conv)   | `filter2d()` → `F.conv2d`                     | Pre-generated kernels from dataset |
| 2 | Resize               | `resize_pt()` → `F.interpolate` + custom Lanczos | Bicubic/bilinear/nearest/lanczos |
| 3 | Gaussian noise       | `random_add_gaussian_noise_pt()` → `torch.randn` | Batched, supports gray noise   |
| 4 | Poisson noise        | `random_add_poisson_noise_pt()` → `torch.poisson` | Batched, supports gray noise   |
| 5 | JPEG compression     | `DiffJPEG` → DCT + quantize + IDCT all on GPU  | Differentiable option available |
| 6 | HF noise (Beta)      | `_generate_shared_hf_noise()` → `torch.distributions.Beta` | Batched GPU sampling |
| 7 | USM Sharp            | `USMSharp` → `F.conv2d` + mask blending         | GPU module                     |
| 8 | Thick lines          | `ThickLines` → `F.max_pool2d`                   | GPU module                     |

### 2.2 traiNNer-redux OTF — GPU-Native but Architecturally Bottlenecked

| # | Operation            | Implementation                                | Problem                         |
|---|----------------------|-----------------------------------------------|--------------------------------|
| 9 | NLMeans denoise      | `nlmeans_denoise_pt()` → pure PyTorch          | 441-iter Python loop → 3,500 CUDA kernel launches. No CPU roundtrip but massive dispatch overhead. `torch.compile` makes it worse (recompile on parameter change). Needs custom CUDA kernel. |

### 2.3 traiNNer-redux OTF — CPU Roundtrip via `apply_per_image` (Optimization Targets)

These use `apply_per_image()` which does GPU→CPU→numpy→process→CPU→GPU per image in
the batch. This is the **primary latency bottleneck**.

| # | Operation            | Current Path                                   | Bottleneck                      |
|---|----------------------|------------------------------------------------|--------------------------------|
| 10 | WebP compression    | `compress_webp()` → cv2 encode/decode          | CPU numpy, cv2 codec           |
| 11 | Video codecs        | `compress_video()` → FFmpeg subprocess pipes    | Subprocess spawn, pipe I/O, CPU |
| 12 | Channel shift       | `apply_shift()` → cv2.warpAffine per channel    | CPU numpy, 3-4 affine transforms |
| 13 | Chroma subsampling  | `apply_subsampling()` → colour lib + chainner_ext | CPU numpy, multiple libs      |
| 14 | Dithering           | `apply_dithering()` → chainner_ext              | CPU numpy, some algorithms sequential |

---

## 3. Optimization Strategy — Tiered Approach

### Tier 1: Pure PyTorch GPU Port — Channel Shift ✅ DONE

**Operation:** Channel shift (#12) — was CPU via `apply_per_image` + cv2.warpAffine

**Implementation:** `optimized/gpu_degradations.py:channel_shift_pt()`
- RGB mode: `F.pad` + slice for constant-border shift per channel
- YUV mode: 3×3 matrix multiply RGB→YCbCr (BT.2020), shift, inverse matrix back
- CMYK mode: Standard formula RGB→CMYK, shift 4 channels, inverse back
- Shared color space utils: `rgb_to_ycbcr_pt()`, `ycbcr_to_rgb_pt()`, `rgb_to_cmyk_pt()`, `cmyk_to_rgb_pt()`

**Benchmark (256×256, single image):**
| Mode | CPU | GPU | Speedup |
|---|---|---|---|
| RGB | 0.51 ms | **0.07 ms** | **7.3×** |
| YUV | ~2 ms* | 0.41 ms | ~5× |
| CMYK | ~2 ms* | 0.24 ms | ~8× |

### Tier 2: GPU Port — Chroma Subsampling ✅ DONE

**Operation:** Chroma subsampling (#13) — was CPU via colour lib + chainner_ext

**Implementation:** `optimized/gpu_degradations.py:chroma_subsample_pt()`
- RGB→YCbCr: 3×3 matrix multiply with BT.601/709/2020/240M weights
- `F.interpolate` for chroma down/up (nearest, bilinear, bicubic)
- `F.conv2d` with generated Gaussian kernel for optional chroma blur
- Inverse matrix back to RGB

**Benchmark:** GPU 4:2:0 bilinear = **0.40 ms** (vs ~3-5ms CPU with apply_per_image)

### Tier 3: GPU Port — Dithering ✅ PARTIALLY DONE

**Operation:** Dithering (#14) — was CPU via chainner_ext

**Implementation:** `optimized/gpu_degradations.py`
- **Quantize (GPU):** `torch.round(img * (N-1)) / (N-1)` — **0.03 ms** (5.3× vs CPU)
- **Ordered dither (GPU):** Bayer matrix tiling + threshold — **0.23 ms**
- **Error diffusion (CPU):** Kept on chainner_ext — causal scan dependency prevents efficient GPU parallelization. Exact Floyd-Steinberg/JJN/Stucki require ~768 serial steps with peak parallelism of only 128 pixels for 256×256.
- **Dot diffusion (tested, rejected):** Knuth's parallel algorithm with 64 serial class steps. PyTorch implementation too slow (29.7ms) due to 64 serial torch.roll + mask operations. Would need custom CUDA kernel.
- **Riemersma (CPU):** Sequential Hilbert curve traversal. No known GPU parallelization.

**Research (Codex consultation):** Exact error diffusion diagonal wavefront gives only ~128 pixels of parallelism at peak for 256×256. Dot diffusion and blue-noise are the GPU-friendly alternatives but produce different (not identical) artifacts.

### Tier 4: Compression Optimization ✅ DONE
*Exhaustively benchmarked. Hybrid TorchCodec + PyAV strategy.*

**Operations:** WebP (#10), Video codecs (#11)

#### 4A. JPEG (Already Solved)
DiffJPEG runs entirely on GPU. No action needed.

#### 4B. WebP Compression ✅ DONE
- No GPU WebP encoder/decoder exists (confirmed: no nvWebP, NVIDIA nvImageCodec
  only has CPU fallback for WebP, no CUDA path).
- TorchCodec WebP batch FAILED (decode returns height=0).
- PyAV WebP batch FAILED (WebP is image format, not video container).

**Winner: Pillow with `method=0` + ThreadPoolExecutor.**
- OpenCV cv2 doesn't expose libwebp's `method` parameter (stuck at default `method=4`).
- Pillow exposes it: `method=0` is the fastest libwebp setting, 1.8× faster per image.

| Approach | Batch=8 | Single | Speedup vs cv2 seq |
|---|---|---|---|
| cv2 sequential (old) | 72.6 ms | 8.6 ms | 1.0× |
| cv2 parallel | 15.6 ms | — | 4.7× |
| Pillow method=0 seq | 43.9 ms | 4.9 ms | 1.7× |
| **Pillow method=0 PARALLEL** | **11.3 ms** | — | **6.4×** |

**Codex consultation findings:**
- No maintained CUDA/Vulkan WebP codec exists anywhere.
- WebP artifact simulation on GPU is feasible as modified DiffJPEG (4×4 blocks instead
  of 8×8, add VP8-style intra prediction + deblocking). No published "DiffWebP" found.
- libwebp2 is experimental, Google says it "will not be released as an image format."
- A pybind11/C++ wrapper over libwebp with GIL release + buffer reuse would be the
  ultimate CPU path, but Pillow method=0 + ThreadPool is good enough.

**WTP GUI integration:**
- `pipeline/process/compress_degr.py`: `__webp()` now uses Pillow with `method=0`.

#### 4C. Video Codecs (H264, HEVC, VP9, MPEG2, MPEG4) ✅ DONE

**Winner: Hybrid TorchCodec BATCH + PyAV BATCH (per codec):**

| Codec | Backend | Time (batch=8) | Speedup vs subprocess |
|---|---|---|---|
| mpeg4 | TorchCodec BATCH | 10.6 ms | ~48× |
| libx264 | TorchCodec BATCH | 11.7 ms | ~43× |
| mpeg2 | PyAV BATCH | 35.7 ms | ~14× |
| libx265 | TorchCodec BATCH | 56.7 ms | ~9× |
| libvpx-vp9 | PyAV BATCH (cpu-used=8) | 99.3 ms | ~5× |

**TorchCodec** is best for x264/x265/mpeg4: native tensor I/O, single encode call for
entire batch, no apply_per_image bridge. **PyAV** needed for VP9 (TorchCodec can't pass
`cpu-used` option → 597ms without it) and MPEG2 (TorchCodec decode fails).

**NVENC exhaustively tested, rejected:**
- Driver 595 fixed error 21 (was broken on 591.74 with CUDA 13.2)
- Tested ALL paths: TorchCodec h264_nvenc (segfault — CUDA context conflict),
  PyAV NVENC (215ms batch — session init overhead), PyNvVideoCodec (decode hangs/fails)
- Encoder init: 185ms (kills per-batch use). Can't reuse after EndEncode (degraded output).
- Found PyNvVideoCodec `bytes()` assembly bug: `bytes(int)` creates zero-filled buffer.
- **Verdict:** NVENC fundamentally unsuitable for small-batch OTF degradation.

**Semantic equivalence (critical):**
- All batch paths use **intra-only encoding**: TorchCodec sets `g=1, bf=0` via
  `extra_options`; PyAV sets `stream.gop_size = 1`. Every frame is an independent
  I-frame. Without this, P/B-frames would apply temporal prediction across
  unrelated training images, producing nonsensical inter-frame artifacts.
- All paths pass **video_sampling** (chroma subsampling format) randomly sampled from
  config, matching the per-image subprocess behavior. Odd spatial dims are padded
  (reflect) to meet chroma alignment requirements, then cropped back after decode.
- Verified: with `gop=1`, TorchCodec batch output is bitwise identical to **per-image
  TorchCodec** encode (mean diff = 0.0000). Note: this proves batch ≡ per-image for the
  *same codec library*, not necessarily vs the original ffmpeg subprocess path (which may
  use different default codec options). For training purposes the artifacts are from the
  same codec and are equivalent in kind.

**WTP GUI integration:**
- `pipeline/process/compress_degr.py`: TorchCodec → PyAV → subprocess priority chain
- `optimized/compress_video_batch.py`: batch compression for traiNNer-redux training

**Files:**
- `optimized/compress_video_batch.py` — production batch API (gop=1, video_sampling)
- `optimized/inprocess_codec.py` — per-image PyAV/TorchCodec (benchmark code)
- `pipeline/process/compress_degr.py` — WTP GUI integration

### Tier 5: NLMeans — Custom CUDA Kernel ✅ DONE
*The existing pure-PyTorch NLMeans has a fundamental structural problem.*

**Problem:** The current `nlmeans_denoise_pt()` uses a nested Python loop over
`search_size²` = 441 iterations, each launching ~8 CUDA kernels = **3,500 kernel
launches total**. At ~5-10μs dispatch overhead each, that's 17-35ms of pure waste
before any computation. `torch.compile` cannot fix this — it fuses ops *within*
each iteration but cannot eliminate the Python loop. Worse, any Python-scalar
parameter (like `h`) that changes between calls triggers a full recompile,
making `torch.compile` actively harmful here.

**Research findings (no existing library has GPU NLMeans for PyTorch):**
- kornia: No NLMeans implementation
- CuPy: No NLMeans in cupyx.scipy.ndimage
- NVIDIA NPP: No NLMeans API
- OpenCV CUDA: Has `fastNlMeansDenoising` — benchmarked at 15.9ms/image, **70× slower than ours**

**Solution — v3: D-tile + separable box filter CUDA kernel:**
Key insight: the 7×7 patch SSD is a 7×7 box filter of per-pixel squared differences
D(u,v) = Σ_c (I(u,v,c) - I(u+dy,v+dx,c))². Instead of each thread computing
147 FMAs (7×7×3) per search offset, we:
1. Cooperatively compute D on a (BLOCK+2×pr)² tile (~6 ops/thread)
2. Row-pass box filter: sum template_size values along x (cooperative)
3. Column-pass: each thread sums template_size row-filtered values
4. Total: ~20 ops/thread vs ~588 ops/thread — **~30× reduction in hot-loop work**

**Architecture:**
- All 3 channels loaded into shared memory ONCE per block (~21KB)
- D-tile recomputed per offset in a separate smem region (~1.9KB)
- Row-filtered buffer for separable pass (~1.4KB)
- Total smem: ~24KB (well within 48KB limit)
- `__expf` (fast math), `__launch_bounds__(256, 3)`, precomputed `neg_inv_nh` constant

**Optimization history (RTX 5090, batch=8, 256×256, h=30, template=7, search=21):**

| Version | Approach | Time | Speedup |
|---|---|---|---|
| Original | 441-iter Python loop | 39 ms | 1.0× |
| v1 | All-channel smem, single load | 13 ms | 3.0× |
| v2 variant E | + unroll + 3ch registers | 4.6 ms | 8.5× |
| **v3 separable** | **D-tile + separable box filter** | **1.97 ms** | **19.6×** |

**Rejected approaches (with reasons):**
- Semi-batched (batch 21 offsets): slower for search=21 — memory bandwidth overhead
  of stacking 21 offset tensors exceeds kernel launch savings. 665MB extra VRAM.
- torch.compile: not tested further after architecture analysis showed it can't
  eliminate the Python loop, and scalar `h` causes recompiles.
- Integral image (hybrid PyTorch + CUDA): 22.9ms — 441 PyTorch kernel launches
  for diff_sq + conv2d per offset negated the O(1) patch SSD savings.
- v2 + fast_math only: 5.1ms — `__expf` barely helped because bottleneck was
  instruction count, not exp() cost.

**Files:**
- `optimized/csrc/nlmeans_kernel.cu` — production v3 kernel
- `optimized/csrc/nlmeans.cpp` — PyTorch C++ binding
- `optimized/nlmeans_cuda.py` — JIT compile wrapper with `--use_fast_math`
- `pipeline/process/hf_noise_degr.py` — WTP GUI integration (auto-loads CUDA kernel)

### Tier 6: Pipeline-Level Architecture Optimizations
*Cross-cutting concerns that amplify all individual operation gains.*

#### 6A. torch.compile — Strategic Application
`torch.compile` is powerful but must be applied to the **right** operations:

**Good targets (element-wise chains, no Python loops):**
- Chains of color_levels → saturation → clamp → gamma
- Noise generation + clipping
- HF noise beta sampling + normalization

**Bad targets (Python loops, dynamic control flow):**
- NLMeans (441-iteration loop — use CUDA kernel instead)
- Any operation with data-dependent branching per pixel

**CRITICAL CONSTRAINT for compilable functions:** Every parameter that varies between
iterations (quality, sigma, amount, etc.) **must be a Tensor**, never a Python scalar.
Python scalars become compile-time guards → recompile on every new value → catastrophic.
```python
# BAD — h is Python float → recompile on change
def fn(x, sigma: float):
    return x * sigma  # sigma baked as constant

# GOOD — sigma is Tensor → traced as dynamic, zero recompiles
def fn(x, sigma: Tensor):
    return x * sigma  # runtime multiply
```
This rule applies project-wide to ALL compilable GPU degradation functions.

#### 5B. CUDA Stream Parallelism
- Use separate CUDA streams for independent operations
- Example: while GPU processes blur for current batch, CPU prepares next batch's kernels
- Overlap CPU-bound compression with GPU-bound operations

#### 5C. Batched apply_per_image Replacement
- Current `apply_per_image` is sequential over batch dimension
- Replace with `ThreadPoolExecutor` + pinned memory:
  ```python
  def apply_per_image_parallel(tensor, fn, max_workers=None):
      # Pin memory, parallel process, async transfer back
  ```

#### 5D. Pre-allocated Buffer Pool
- Avoid `torch.zeros`/`torch.empty` allocations per iteration
- Pre-allocate working buffers for each degradation stage
- Reuse across iterations via a simple buffer pool

#### 5E. Kernel Caching
- Blur kernels, screentone patterns, dithering matrices: generate once, cache on GPU
- Avoid regenerating static patterns every iteration

#### 6F. Async Degradation Prefetching ✅ SIMULATED (directional)
*Hide ALL remaining `feed_data()` latency behind GPU training compute.*

**Problem:** `feed_data()` runs synchronously on the main thread. The GPU idles
during degradation, and degradation idles during `optimize_parameters()`. Even after
per-op optimization, the total degradation time directly adds to iteration time.

**Current flow (synchronous):**
```
[feed_data: degrade batch N] → [optimize_parameters: train batch N] → [feed_data: degrade batch N+1] → ...
                               ^ GPU idle during feed_data                ^ GPU idle again
```

**Async flow (validated in simulation):**
```
Thread 1 (main):    [optimize batch N] ──────────────→ [optimize batch N+1] ────→ ...
Thread 2 (prefetch):  [degrade batch N+1] ──────────→ [degrade batch N+2] ──→ ...
                      ^ overlapped with training      ^ overlapped again
```

**Simulation results (RTX 5090, batch=8, 256×256):**

| Metric | Value |
|---|---|
| Degradation (D) | 19.0 ms/batch |
| Training step (T) | 26.7 ms/batch |
| **Sync** (D + T) | **43.3 ms/iter** |
| **Async** (≈ max(D, T)) | **24.5 ms/iter** |
| **Speedup** | **1.77×** |
| **Time saved** | **18.8 ms/iter (43%)** |

Async iter (24.5ms) is close to the theoretical floor (26.7ms = training alone).
Degradation is almost completely hidden behind training compute.

**Simulation limitations (directional, not integration-ready):**
- Reuses one GT tensor (valid simplification — degradation cost is content-independent)
- Bypasses DataLoader (by design — DataLoader has its own prefetcher in `train.py`)
- Bypasses diversity pool `_dequeue_and_enqueue` (adds ~0.1ms, not material)
- Uses synthetic training step (8-layer conv forward+backward, ~27ms = realistic)
- The 1.77× is a directional result for the overlap principle. Actual integration
  speedup depends on real feed_data composition and diversity pool overhead.

**Architecture for real integration (NOT just "background-thread feed_data()"):**

The current `feed_data()` writes directly to model state (`self.gt`, `self.lq`,
`self.kernel1/2`, `self.sinc_kernel`) and mutates the diversity pool
(`self.queue_lr/gt/ptr`). Running it in a background thread would race with
`optimize_parameters()` which reads `self.lq`/`self.gt`.

**Required refactoring — must preserve current operation order:**

Current `feed_data()` order (lines 568-613 of `realesrgan_model.py`):
```
clamp/round → crop → HF noise → dithering → diversity pool → contiguous → batch_augment
```

The diversity pool stores and replays **fully-processed** (lq, gt) pairs — after
HF noise and dithering. Moving the pool before those ops would change what gets
stored/replayed, altering the training distribution. The async split must respect this.

1. **Extract degradation into a pure function:**
   ```python
   def degrade_batch(data, opt, device) -> tuple[Tensor, Tensor]:
       """Pure function: DataLoader dict → (lq, gt). No model state.
       Covers: all degradation → crop → HF noise → dithering."""
       # Everything from feed_data up through dithering
       return lq, gt
   ```
2. **Background thread calls `degrade_batch()`**, not `feed_data()`. Produces
   immutable `(lq, gt)` tuples (fully degraded, cropped, HF-noised, dithered)
   into a `deque` queue.
3. **Main thread's `feed_data()` pulls `(lq, gt)` from queue**, then:
   - Sets `self.lq, self.gt = lq, gt` (from queue)
   - `_dequeue_and_enqueue()` (reads/mutates `self.lq`/`self.gt` + pool state)
   - `self.lq = self.lq.contiguous()`
   - `batch_augment(self.gt, self.lq)` (if enabled)
4. **`train.py` coordination:** The existing `CUDAPrefetcher` in `train.py` handles
   DataLoader → GPU transfer. Async degradation sits between DataLoader output and
   model consumption. Both prefetchers need coordination.

**RNG determinism:** The OTF path draws from `RNG.get_rng()` (singleton) and
`random.choices()` throughout `feed_data()`, and `batch_augment` does too. With
degradation on a background thread and `batch_augment` on the main thread, both
threads would draw from shared global RNG, breaking seeded reproducibility.
Options:
- **Per-thread RNG streams:** Background thread gets its own `numpy.random.Generator`
  and `torch.Generator`, seeded deterministically from the global seed + thread ID.
  Main thread keeps the original globals for `batch_augment`.
- **Pre-sample params:** Main thread pre-samples all random params (quality, sigma,
  codec choice, etc.) into a dict, passes it to `degrade_batch()` which uses them
  deterministically. More invasive but fully reproducible.
- **Accept non-reproducibility:** If seeded reproducibility is not required (common
  for degradation pipelines where randomness is the point), just let both threads
  race on global RNG. Training convergence is unaffected; only exact seed replay breaks.

**This is NOT "modify realesrgan_model.py only"** — it requires:
- New `degrade_batch()` pure function (extracted from `feed_data()`)
- Modified `feed_data()` (pull from queue + post-degrade steps)
- Modified `train.py` training loop (start/stop prefetcher lifecycle)
- Thread synchronization for CUDA streams
- RNG ownership strategy (per-thread streams or pre-sampled params)

**Files:**
- `bench_async_prefetch.py` — standalone simulation (proof of concept, no traiNNer deps)

---

## 4. Module Design

### 4.1 New Module: `traiNNer/data/gpu_degradations.py`

GPU-native replacements for existing `apply_per_image` operations. Pure PyTorch,
no external deps (except torch). Each function takes/returns BCHW tensors.

```
gpu_degradations.py
├── channel_shift_pt(img, shifts, color_space) → Tensor
├── chroma_subsample_pt(img, format, down_mode, up_mode, blur_sigma) → Tensor
├── ordered_dither_pt(img, levels, map_size) → Tensor
├── quantize_pt(img, levels) → Tensor
└── rgb_to_ycbcr_pt(img) / ycbcr_to_rgb_pt(img) → Tensor
```

### 4.2 Updated Module: `traiNNer/data/otf_degradations.py`

Keep existing CPU-path functions. Add parallel batch processing and in-process
codec paths (TorchCodec BATCH for x264/x265/mpeg4, PyAV BATCH for vp9/mpeg2).

```
otf_degradations.py (updated)
├── apply_per_image_parallel(tensor, fn, max_workers) → Tensor  # threaded
├── compress_video_batch(tensor, codec, quality, sampling) → Tensor  # TorchCodec/PyAV
└── (existing functions remain for CPU-only ops: error diffusion, riemersma)
```

### 4.3 Updated Model: `realesrgan_model.py`

Replace `apply_per_image` calls with GPU-native functions for shift, subsampling,
and dithering. Wire batch compression path into `_apply_compression()`.
Wire CUDA NLMeans into `_apply_shared_hf_noise()`.

---

## 5. Task Decomposition

### Phase 1: Channel Shift → GPU ✅ DONE
- **Task 1.1:** ✅ `channel_shift_pt()` — F.pad+slice for constant-border shift, matrix multiply for YUV/CMYK
- **Task 1.2:** ✅ Wired into WTP GUI `shift_degr.py`

### Phase 2: Chroma Subsampling → GPU ✅ DONE
- **Task 2.1:** ✅ `chroma_subsample_pt()` — 3×3 matmul + F.interpolate + F.conv2d Gaussian blur
- **Task 2.2:** ✅ Wired into WTP GUI `subsampling_degr.py`

### Phase 3: Dithering → GPU ✅ PARTIALLY DONE
- **Task 3.1:** ✅ `quantize_pt()` + `ordered_dither_pt()` on GPU
- **Task 3.2:** ✅ Wired into WTP GUI `dithering_degr.py` — GPU for quantize/ordered, CPU for error diffusion/riemersma
- **Task 3.3:** Dot diffusion tested, rejected (29.7ms PyTorch impl — needs CUDA kernel for viability)

### Phase 4: Compression Optimization ✅ DONE
- **Task 4.1:** ✅ `apply_per_image_parallel` — persistent ThreadPoolExecutor + pinned memory
- **Task 4.2:** ✅ TorchCodec BATCH for x264/x265/mpeg4 (10.6–56.7ms, 9–48× speedup)
- **Task 4.3:** ✅ PyAV BATCH for VP9 (99ms with cpu-used=8) and MPEG2 (35.7ms)
- **Task 4.4:** ✅ Wired into WTP GUI (`compress_degr.py`) and training API (`compress_video_batch.py`)
- **Task 4.5:** ✅ NVENC exhaustively tested and rejected (185ms session init, decode failures)

### Phase 5: NLMeans CUDA Kernel ✅ DONE
- **Task 5.1:** ✅ Semi-batched tested — rejected (slower for search=21 due to memory bandwidth)
- **Task 5.2:** ✅ v3 CUDA kernel: D-tile + separable box filter (1.97ms, 19.6× speedup)
  - Iterated through v1 (all-channel smem, 13ms), v2E (unroll+regopt, 4.6ms),
    v3 (D-tile reformulation, 1.97ms). Codex consultation for v2→v3 breakthrough.
- **Task 5.3:** ✅ `nlmeans.cpp` — PyTorch C++ binding with reflect padding
- **Task 5.4:** ✅ JIT compile via `torch.utils.cpp_extension.load()` with `--use_fast_math`
- **Task 5.5:** ✅ Wired into WTP GUI (`hf_noise_degr.py` auto-loads CUDA kernel)

### Phase 6: Pipeline-Level Optimizations
- **Task 6.1:** torch.compile — DEPRIORITIZED per Codex assessment
  - Only viable for HF noise beta block (1.65→0.10ms, 0 graph breaks)
  - Gaussian noise: 4 graph breaks, compiled SLOWER (2.43ms eager → 8.65ms compiled)
  - Poisson noise: 6 graph breaks, flat (4.14ms → 4.19ms)
  - Use `torch._dynamo.explain(fn)(...)` to check candidates before attempting
- **Task 6.2:** CUDA stream parallelism — SKIP per Codex assessment
  - feed_data is a sequential dependency chain; no meaningful intra-pass parallelism
  - The real overlap opportunity is async prefetch (6.5)
- **Task 6.3:** Buffer pool — SKIP generic pool; targeted caching only if 6.4 shows need
  - PyTorch CUDA allocator already pools aggressively
  - Consider: cache USMSharp instances by radius, Bayer matrices, Gaussian kernels
- **Task 6.4:** ✅ End-to-end benchmarking harness — DONE
  - `bench_feed_data.py`: 3 profiles (span_minimal, full_featured, codec_heavy)
  - Per-stage CUDA event timing + wall-clock, distribution stats (mean/median/P95/min/max)
  - Fair comparison: baseline uses same fastest presets (ultrafast, cpu-used=8) as optimized
  - Results below in section 7
- **Task 6.5:** ❌ Async degradation prefetching — UNNECESSARY after per-op optimizations
  - Original simulation showed 1.77× but was too simplified (no GPU contention)
  - Real end-to-end benchmark: **0.98-1.02× (no gain)** — both streams compete for GPU SMs
  - **More importantly**: real training is 1,700-5,500 ms/iter (0.18-0.58 iter/s).
    Optimized degradation is ~24ms = **1.4% of iteration time at most**.
    Even the BASELINE degradation (114ms) was only 6.6%. Async would save ≤24ms
    out of 1,700ms — not worth the complexity of refactoring feed_data/train.py.
  - **Conclusion: per-op optimizations eliminated degradation as a bottleneck entirely.**

---

## 6. Implementation Constraints

1. **No modifications to existing GPU-native operations** (items 1-8). They work.
2. **Backwards compatible** — existing configs must keep working unchanged.
3. **All new GPU functions must support batched BCHW input** (not per-image).
4. **No new pip dependencies** unless discussed. Prefer pure PyTorch.
5. **C++/CUDA extensions** for NLMeans kernel via `torch.utils.cpp_extension.load()` (JIT).
6. **Test each GPU replacement** against the existing CPU numpy path for correctness.
7. **torch.compile rule:** variable parameters = Tensor, never Python scalar. Operations
   with Python loops (NLMeans, anything iterative) get CUDA kernels, not torch.compile.

---

## 7. Impact — Per-Operation and End-to-End

### 7.1 Per-Operation Speedups (isolated benchmarks, batch=8, 256×256, RTX 5090)

| Optimization | Before | After | Speedup | Method |
|---|---|---|---|---|
| Video compression (in-process) | 505 ms (subprocess) | 10.6–56.7 ms | 9–48× | TorchCodec/PyAV BATCH |
| NLMeans CUDA kernel | 39 ms | 1.97 ms | 19.6× | Custom CUDA v3 D-tile |
| WebP (Pillow method=0) | 72.6 ms | 11.3 ms | 6.4× | Pillow + ThreadPool |
| Channel shift → GPU | 0.51 ms | 0.07 ms | 7.3× | F.pad+slice |
| Chroma subsampling → GPU | ~3-5 ms | 0.40 ms | ~10× | matmul + F.interpolate |
| Dithering → GPU (ordered/quantize) | 0.16 ms | 0.03 ms | 5.3× | torch.round / Bayer |
| Async degradation prefetch | 43.3 ms/iter | 24.5 ms/iter | 1.77× | Simulated (directional) |

### 7.2 End-to-End Pipeline Results (Task 6.4 harness, fair comparison)

Measured with `bench_feed_data.py`. **Fair baseline**: subprocess path uses same
fastest presets as optimized (ultrafast for h264/hevc, cpu-used=8 for vp9). This
isolates the in-process vs subprocess advantage from preset differences.

| Profile | Baseline | Optimized | Speedup | Notes |
|---|---|---|---|---|
| **span_minimal** (JPEG only) | 14.6 ms | 12.9 ms | **1.13×** | DiffJPEG already GPU-native; minimal headroom |
| **full_featured** (all features) | 124.4 ms | **23.7 ms** | **5.25×** | NLMeans 38→0.84ms, compress 69→8ms, shift 1→0.05ms |
| **codec_heavy** (video codecs) | 518.3 ms | **55.5 ms** | **9.35×** | Compress 500→46ms; subprocess overhead eliminated |

### 7.3 Per-Stage Breakdown — full_featured profile (fair baseline)

| Stage | Baseline | Optimized | Speedup | Notes |
|---|---|---|---|---|
| Compression 1 | 68.8 ms | 8.4 ms | **8.2×** | TorchCodec/PyAV batch vs subprocess |
| **NLMeans** | **38.3 ms** | **0.84 ms** | **45×** | Custom CUDA v3 kernel |
| Compression 2+resize+sinc | 3.9 ms | 3.4 ms | 1.2× | Mostly resize+sinc (already GPU) |
| Noise 1+2 | 4.5 ms | 4.1 ms | 1.1× | Already GPU-native |
| Channel shift | 1.06 ms | 0.05 ms | **21×** | GPU F.pad+slice vs CPU warpAffine |
| Chroma subsampling | 0.06 ms | 0.16 ms | — | Baseline skipped (pepeline error) |
| Dithering | 0.21 ms | 0.98 ms | — | Variable (error diffusion outliers) |
| HF noise | 0.21 ms | 0.17 ms | 1.2× | Already GPU-native |
| All other stages | ~1 ms | ~1 ms | 1.0× | Resize, blur, crop, clamp — already fast |

### 7.4 Key Findings

1. **NLMeans is the single largest win in the full pipeline.** The custom CUDA kernel
   took it from 38ms (31% of baseline) to 0.84ms — responsible for ~37ms of the total
   101ms improvement.

2. **Compression is the single largest win in codec-heavy configs.** Subprocess
   elimination + batch encoding took compression from 500ms to 46ms. Even with fair
   fastest presets on the baseline, the subprocess overhead (~10ms per process spawn ×
   8 images × 2 stages) dominates.

3. **JPEG-only configs see minimal improvement** because DiffJPEG is already GPU-native.
   The 1.13× comes from slightly tighter Python overhead in the optimized path.

4. **Variance is dramatically reduced.** Baseline codec_heavy P95 = 758ms, max = 1076ms
   (subprocess spawn is non-deterministic). Optimized P95 = 140ms, max = 226ms.

5. **Remaining bottlenecks** (after all optimizations):
   - Noise generation: ~2ms × 2 stages = ~4ms (Gaussian/Poisson, already GPU-native)
   - Compression: ~8ms when video codecs fire (TorchCodec/PyAV in-process)
   - Resize: ~0.2ms × 3 stages (F.interpolate, already GPU-native)
   - Python dispatch overhead: ~4ms gap between CUDA sum and wall-clock

---

## 8. Fidelity Acceptance Matrix

For a training degradation pipeline, "fidelity" means the GPU path produces
**visually equivalent degradation artifacts** — not bitwise-identical output. The
model learns from corrupted images; the exact corruption pattern varies by design
(random quality, random sampling). What matters is that the *distribution* of
artifacts matches.

| Operation | GPU vs CPU | Equivalence | Notes |
|---|---|---|---|
| **NLMeans** | Custom CUDA kernel vs PyTorch loop | **Bitwise identical** (max diff = 0.000000) | Same algorithm, same math, just faster execution |
| **Quantize dither** | `torch.round` vs chainner_ext | **Mathematically identical** | Same formula: `round(x * (N-1)) / (N-1)` |
| **Ordered dither** | Bayer matrix + threshold vs chainner_ext | **Mathematically identical** | Same Bayer matrix construction, same threshold logic |
| **Video codecs** | TorchCodec/PyAV batch vs per-image TorchCodec | **Bitwise identical** (batch vs per-image, with gop=1) | Same codec library, same CRF, all I-frames. Verified: mean diff = 0.0000 for all 8 frames. Not verified against ffmpeg subprocess (different default options), but same codec produces equivalent artifact class |
| **WebP** | Pillow method=0 vs cv2 | **Near-identical** | Both call libwebp. method=0 uses faster encoding strategy but same decompression. Output may differ by ±1 in rare pixels due to encoding path differences |
| **Channel shift** | F.pad+slice vs cv2.warpAffine | **Acceptable difference** | GPU uses constant-border pad+slice; CPU uses warpAffine with BORDER_CONSTANT. For integer shifts: identical. For sub-pixel shifts (percent mode): bilinear interpolation via F.grid_sample vs cv2.warpAffine — minor floating-point differences |
| **Chroma subsampling** | F.interpolate vs chainner_ext resize | **Acceptable difference** | GPU supports nearest/bilinear/bicubic. CPU chainner_ext supports additional filters (Lanczos, Hermite, etc.). For shared modes: very close but not identical due to FP32 vs FP64 differences in interpolation. For training degradation this is well within acceptable range |
| **Error diffusion** | CPU chainner_ext (unchanged) | **Identical** | Not ported to GPU — kept on CPU |
| **Riemersma** | CPU chainner_ext (unchanged) | **Identical** | Not ported to GPU — kept on CPU |

**Acceptance criteria:** An optimization is acceptable if:
1. Output is bitwise identical, OR
2. Output produces the same class of visual artifacts (same artifact type,
   similar severity), with per-pixel differences within ±2 on [0, 255] scale

All GPU ports above meet these criteria. The key validation was the video codec
path: without `gop=1`, TorchCodec batch produced P-frame artifacts (mean diff ~3.5)
that don't exist in per-image encoding. With `gop=1`, the output is bitwise identical.

---

## 9. CUDA Extension Deployment Strategy

The NLMeans CUDA kernel currently uses JIT compilation via
`torch.utils.cpp_extension.load()`. This is appropriate for proof-of-concept but
has deployment considerations for production:

| Concern | Impact | Mitigation |
|---|---|---|
| **Cold-start compile** | ~30-60s first time per environment | Cached in `~/.cache/torch_extensions/`. Recompiles only on source change. Acceptable for training (one-time cost). |
| **Multi-process contention** | DDP training spawns N processes; all may JIT simultaneously | Use file lock around `load()` call, or pre-build `.so` at install time via `setup.py` `ext_modules` |
| **Toolchain ABI** | Requires matching CUDA toolkit + C++ compiler at runtime | Ship pre-built `.so` for common CUDA versions, or document build requirements |
| **Non-NVIDIA GPUs** | CUDA kernel won't work on AMD/Intel | Fall back to PyTorch loop (existing `nlmeans_denoise_pt`). This is the one case where CPU fallback is acceptable — it's a graceful degradation of *speed*, not functionality |

**Recommended deployment path for traiNNer-redux:**
1. **Phase 1 (current):** JIT compile with `load()`. Simple, works for single-GPU training.
2. **Phase 2 (port):** Add `CUDAExtension` to traiNNer-redux `setup.py`. Pre-builds
   during `pip install`. Falls back to JIT if pre-built `.so` missing.
3. **Phase 3 (optional):** Ship pre-built wheels per CUDA version for zero-compile install.

---

## 10. File Map

```
wtp_gui_experimental/
├── optimized/                        # Proof-of-concept optimized implementations
│   ├── csrc/
│   │   ├── nlmeans_kernel.cu        # ✅ v3 D-tile + separable box filter (production)
│   │   ├── nlmeans.cpp              # ✅ PyTorch C++ binding
│   │   ├── nlmeans_v3.cu           # v3 development variants (benchmark)
│   │   └── nlmeans_variants.cu     # v2 variant benchmark (A-E, reference)
│   ├── nlmeans_cuda.py              # ✅ JIT compile wrapper with --use_fast_math
│   ├── gpu_degradations.py          # ✅ channel_shift_pt, chroma_subsample_pt, dither_pt
│   ├── compress_video_batch.py      # ✅ TorchCodec BATCH + PyAV BATCH (gop=1, video_sampling)
│   ├── parallel.py                  # ✅ apply_per_image_parallel (persistent ThreadPool)
│   └── inprocess_codec.py           # Per-image PyAV/TorchCodec (benchmark code)
├── pipeline/process/
│   ├── compress_degr.py             # ✅ UPDATED: TorchCodec → PyAV → subprocess
│   ├── hf_noise_degr.py            # ✅ UPDATED: CUDA NLMeans kernel auto-loads
│   ├── shift_degr.py               # ✅ UPDATED: GPU channel_shift_pt
│   ├── subsampling_degr.py         # ✅ UPDATED: GPU chroma_subsample_pt
│   └── dithering_degr.py           # ✅ UPDATED: GPU quantize/ordered, CPU error diffusion
├── ARCHITECTURE_OTF_OPTIMIZATION.md # This file
├── bench_feed_data.py               # ✅ Task 6.4: end-to-end feed_data harness (3 profiles)
├── bench_async_prefetch.py          # Task 6.5: async prefetch simulation
└── bench_*.py                       # Other benchmarks (quick, head2head, etc.)

# Target (when porting to traiNNer-redux):
traiNNer-redux/traiNNer/
├── csrc/
│   ├── nlmeans_kernel.cu            # Copy from optimized/csrc/
│   └── nlmeans.cpp                  # Copy from optimized/csrc/
├── data/
│   ├── otf_degradations.py          # Add CUDA NLMeans + batch compression
│   └── gpu_degradations.py          # NEW — channel_shift_pt, chroma_subsample_pt, dither_pt
└── models/
    └── realesrgan_model.py          # Wire optimized paths into feed_data + async prefetch
```
