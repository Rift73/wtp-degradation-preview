"""Runtime optimization helpers used by the GUI degradation pipeline.

This package intentionally keeps only the modules that are imported by the
shipping GUI runtime:
- gpu_degradations: GPU-accelerated degradations used by pipeline/process/*.py
- nlmeans_cuda: Lazy NLMeans CUDA extension loader and wrapper
- iir_trailing_cuda: Lazy CUDA extension used by the optimized NTSC path
- dither_cuda: Lazy CUDA extension: exact error-diffusion and riemersma dithering
- cuda_ext: Per-user build cache and loader shared by the three extensions
"""
