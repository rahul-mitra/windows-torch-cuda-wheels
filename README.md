# windows-torch-cuda-wheels

Prebuilt **Windows x64** wheels of CUDA extensions that link against PyTorch's C++ ABI, for combos where
no public Windows builds exist yet. Wheels are published as **GitHub Release assets** (not in git).

## Release `torch2.15.0-cu134-py313-win_amd64`

Target: **torch 2.15.0+cu134** · **CUDA 13.4** · **CPython 3.13** (`cp313`, native Windows, not WSL) ·
**RTX 50xx / Blackwell, compute capability 12.0 (`sm_120`)** · NVIDIA driver with CUDA ≥ 13.4 support.

Release page: https://github.com/rahul-mitra/windows-torch-cuda-wheels/releases/tag/torch2.15.0-cu134-py313-win_amd64

| Wheel | Upstream (exact ref) | Licence | GPU code | Size | SHA256 |
|---|---|---|---|---|---|
| `sageattention-2.2.0.post7+cu134torch2.15.0-cp310-abi3-win_amd64.whl` | [woct0rdho/SageAttention](https://github.com/woct0rdho/SageAttention) tag `v2.2.0-windows.post7` (`79cd62a5`) — Windows fork of [thu-ml/SageAttention](https://github.com/thu-ml/SageAttention) | Apache-2.0 | SASS sm_80, sm_89, sm_90a, **sm_120a** (`_qattn_sm80`, `_qattn_sm89`, `_qattn_sm90`, `_fused`) | 15.7 MB | `78413833e7a5d4696a7a07ea617ca0b26a39ccd6eea23e7dc220029007475cd8` |
| `sageattn3-1.0.0+cu134torch2.15.0-cp310-abi3-win_amd64.whl` | `sageattention3_blackwell/` from [woct0rdho/SageAttention](https://github.com/woct0rdho/SageAttention) `v2.2.0-windows.post7` (`79cd62a5`), CUTLASS `f3fde583`; kernels identical to [thu-ml/SageAttention](https://github.com/thu-ml/SageAttention/tree/main/sageattention3_blackwell) `d1a57a54` | Apache-2.0 (CUTLASS: BSD-3-Clause) | SASS **sm_120a** | 0.5 MB | `7572809c1ac15de024bb6b1c7b41fcdd8ef498cdfeafb8e9e898a770deaeb32a` |
| `nunchaku-1.3.0.dev20260906+cu134torch2.15.0-cp313-cp313-win_amd64.whl` | [nunchux-ai/nunchaku](https://github.com/nunchux-ai/nunchaku) `main` @ `302e0e97` (2026-09-06) | Apache-2.0 (bundled: CUTLASS BSD-3, Block-Sparse-Attention BSD-3, spdlog/fmt MIT, nlohmann/json MIT) | SASS **sm_120a** | 26.0 MB | `c115f5d9ea3003310998966d90877cdcafb298e3c8fdf072f0bb0c528b803520` |
| `flash_attn-2.8.3.post2+cu134torch2.15.0-cp313-cp313-win_amd64.whl` | [Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention) tag `v2.8.3.post2` (`49774b31`) — FA2, not `hopper/` | BSD-3-Clause (CUTLASS: BSD-3-Clause) | SASS **sm_120** | 60.3 MB | `f2fe31f8c12091be601a2265758b4bcc30950b3edefdf8d036ff139ffdd02e5e` |

`SHA256SUMS.txt` and the full verification log `verify.txt` are attached to the release.

## Install

Into a Python 3.13 x64 environment that already has `torch==2.15.0+cu134`
(and `triton-windows` for sageattention):

```bat
pip install https://github.com/rahul-mitra/windows-torch-cuda-wheels/releases/download/torch2.15.0-cu134-py313-win_amd64/sageattention-2.2.0.post7%2Bcu134torch2.15.0-cp310-abi3-win_amd64.whl
pip install https://github.com/rahul-mitra/windows-torch-cuda-wheels/releases/download/torch2.15.0-cu134-py313-win_amd64/sageattn3-1.0.0%2Bcu134torch2.15.0-cp310-abi3-win_amd64.whl
pip install https://github.com/rahul-mitra/windows-torch-cuda-wheels/releases/download/torch2.15.0-cu134-py313-win_amd64/nunchaku-1.3.0.dev20260906%2Bcu134torch2.15.0-cp313-cp313-win_amd64.whl
pip install --no-deps https://github.com/rahul-mitra/windows-torch-cuda-wheels/releases/download/torch2.15.0-cu134-py313-win_amd64/flash_attn-2.8.3.post2%2Bcu134torch2.15.0-cp313-cp313-win_amd64.whl
```

Notes:

- **Exact torch match required** for `nunchaku` and `flash_attn` (they use torch's full C++ ABI): install only
  next to `torch 2.15.0+cu134`. `sageattention` and `sageattn3` are built against torch's stable ABI (`abi3`)
  but were only tested to import against 2.15.0.
- **GPU archs**: SASS only, no PTX — `nunchaku`, `sageattn3`, `flash_attn` run only on compute capability 12.0
  (RTX 50xx desktop Blackwell). `sageattention` also covers sm80/86/89/90.
- **No extra DLLs**: the CUDA runtime is statically linked (from the CUDA 13.4 toolkit); cuBLAS etc. come from
  torch's own bundled CUDA 13.4 DLLs. You need an NVIDIA driver whose `nvidia-smi` reports CUDA ≥ 13.4 (e.g. 621.01 reports 13.5).
  The MSVC runtime (`VCRUNTIME140_1.dll`, `MSVCP140.dll`) must be present — torch needs it too.
- `sageattention` needs `triton-windows` at import time; `nunchaku` needs its Python deps (`diffusers`,
  `transformers`, `accelerate`, `peft`, …) — pip installs them unless you pass `--no-deps`.
- `flash_attn` is installed with `--no-deps` above so pip never tries to change your torch.
- These builds were verified on a GPU-less machine (import, DLL dependencies, embedded SASS, wheel tags);
  kernel correctness was **not** tested at build time.

## Reproducing

See [`BUILD_NOTES.md`](BUILD_NOTES.md) for the toolchain, timings and problems hit, [`build/`](build/) for
the scripts (`build_all.ps1` runs everything: CUDA 13.4 from NVIDIA redist archives → venv → 4 wheels →
verification) and [`patches/`](patches/) for every source change applied to upstream:

| Patch | Why |
|---|---|
| `nunchaku-setup-sm-targets-and-version.patch` | `NUNCHAKU_SM_TARGETS` override (GPU-less build), version label |
| `nunchaku-spdlog-fmt-nvcc13.4-char32.patch` | nvcc 13.4 rejects out-of-range `char32_t` escapes in spdlog's bundled fmt |
| `sageattn3-pymodinit-func-export.patch` | export `PyInit_*` (`PyMODINIT_FUNC`) so the `.pyd` imports on Windows |
| `flash-attn-setup-cxx20-zc-preprocessor.patch` | C++20 + `/Zc:preprocessor` required by torch 2.15 / CUDA 13 CCCL |

## Licences and attribution

This repository only contains build scripts, patches and documentation; the wheels are builds of
third-party projects (with the small patches listed above), each distributed under its own licence (included in every wheel's
`*.dist-info/licenses/`):

- SageAttention — © SageAttention team (Tsinghua University), [Apache License 2.0](https://github.com/thu-ml/SageAttention/blob/main/LICENSE); Windows fork by woct0rdho.
- Nunchaku — © MIT HAN Lab / Nunchaku team, [Apache License 2.0](https://github.com/nunchux-ai/nunchaku/blob/main/LICENCE.txt).
- FlashAttention — © Tri Dao and contributors, [BSD 3-Clause](https://github.com/Dao-AILab/flash-attention/blob/main/LICENSE).
- NVIDIA CUTLASS — © NVIDIA, [BSD 3-Clause](https://github.com/NVIDIA/cutlass/blob/main/LICENSE.txt).
- Block-Sparse-Attention (MIT HAN Lab, BSD 3-Clause), spdlog / {fmt} (MIT), nlohmann/json (MIT), mio (MIT), bundled in nunchaku.

The patches in `patches/` are offered under the licence of the file they modify. Not affiliated with or
endorsed by any of the upstream projects.
