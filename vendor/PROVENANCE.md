# vendor/chainner_ext

`chainner_ext` for CPython 3.14 on 64-bit Windows: a drop-in for chainner_ext 0.3.10
(chaiNNer-rs, which has no Python 3.14 wheel). It is the C rewrite from chaiNNer-C. `install.bat`
puts this folder on the venv's import path with `venv\Lib\site-packages\wtp_vendor.pth`.

## Source

- **Repository:** chaiNNer-C, the owner's C conversion of chaiNNer
  (local `C:\Users\PC\Desktop\Repository\chaiNNer-C`, public fork https://github.com/Rift73/chaiNNer-C).
- **Commit:** vendored on 2026-10-09 at chaiNNer-C `e146bc73147211aee6b6c5925f9e0cb45119fade`.
  The binaries were linked on 2026-10-08 08:41 +0800 from the source tree later committed as
  `e13ebdef` (the DLL exports that commit's `cn_planar_to_interleaved_f32`). The sources of both
  binaries are identical at `e13ebdef`, at `e146bc73`, and at the public tag `v0.3.1` (`197bf0c2`).
  The only `native/` difference, `native/src/video_io.cpp`, builds `_chainner_graph.pyd`, which is
  not vendored here.
- **Build:** `native/build` (CMake Release), clang-cl from LLVM 23.1.2 with ThinLTO and MSVC 14.44
  libraries. It is portable: no `-march` (`CHAINNER_C_MARCH` empty), and kernels choose their ISA
  at run time. `chainner_ext.pyd` uses the stable ABI (`python3.dll`).
  `chainner_native.dll` imports only `KERNEL32.dll`.
- **Conformance:** chaiNNer-C `native/tests/chainner_ext/manifest.json` has 6480 cases compared by
  SHA-256 of the output bytes against chainner_ext 0.3.10. Its exclusions (palette ties and the
  clipboard failure cleanup) are listed in the manifest.

## Files

| File | Bytes | SHA-256 |
|---|---|---|
| `chainner_ext.pyd` | 674304 | `e59941177c46cc8a5f97c812750d8ea69d8e3f1b3ccb37dc1162a451a3232276` |
| `chainner_native.dll` | 2500096 | `6da1217311bb59a67d1a7ad50e20007ef733b5dbce763ed6bfcf708271cfecf5` |
| `__init__.pyi` | 3563 | `44c5f8713cd02c52ff5bf4dee58f5d41c27171e0ca410e5bd3fdf9ec463c64fe` |
| `chainner_ext.pyi` | 329 | `1612460155f1b61ca1c362c6437dfba52d12ffc8f12953e71ecb6a751fbd4692` |

`__init__.py` is chaiNNer-C's `backend/src/chainner_ext/__init__.py` with one change. Upstream
adds `backend/src/nodes/impl` (where chaiNNer-C keeps `chainner_native.dll`) to the DLL search
path. This copy keeps the DLL next to the `.pyd`, which Windows searches by itself.

## Licences

- `chainner_ext.pyd` ports the chaiNNer-rs bindings: MIT OR Apache-2.0, see `LICENSE-MIT` and
  `LICENSE-APACHE`. Both are copied unmodified from chaiNNer-C `native/third_party/chainner_ext/`,
  which records their origin: chaiNNer-rs commit `3b6687c857`.
- `chainner_native.dll` combines code under several licences:
  - GPL-3.0-only code ported from chaiNNer, for example `blend.c` and `channels.c`;
  - code derived from OpenCV (Apache-2.0);
  - code derived from chaiNNer-rs (MIT OR Apache-2.0).

  As distributed, the DLL is therefore under GPL-3.0-only (`LICENSE-GPL-3.0`, copied from
  chaiNNer-C's `LICENSE`). Its corresponding source is chaiNNer-C `native/` at tag `v0.3.1`.
