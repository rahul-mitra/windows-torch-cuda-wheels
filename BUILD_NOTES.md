# Build notes — torch 2.15.0 + CUDA 13.4 + CPython 3.13, win_amd64, sm_120

Built 2026-10-29 on a GPU-less Windows 11 machine (no NVIDIA driver installed). Kernels could not be
executed on the build machine; correctness/perf testing happens on the RTX 5080 target.

## Toolchain

| Component | Version |
|---|---|
| OS | Windows 11 Pro 10.0.26200 x64 (native, not WSL) |
| Host compiler | MSVC `cl.exe` 19.51.36260 (toolset 14.51.36231), Visual Studio Community 2026 18.10.4 |
| Windows SDK | 10.0.26100.0 |
| CUDA | 13.4 Update 3 redist (`nvcc` V13.4.92, `cuda_13.4.r13.4/compiler.38855100_0`) |
| Python | CPython 3.13.13 x64, MSC v.1944 (python-build-standalone via `uv`) |
| torch | 2.15.0+cu134 (`torch.version.cuda == "13.4"`) from download.pytorch.org/whl/cu134 |
| Build tools | setuptools 84.2.1, wheel 0.48.0, ninja 1.13.2, pip 26.2.2 |
| Hardware | i7-13620H (16 threads), 32 GB RAM |

### CUDA toolkit without an installer

The build machine had no CUDA toolkit and the session was not elevated, so instead of the MSI/winget
installer, `build/00_get_cuda.ps1` downloads NVIDIA's official per-component redistributable archives
for release **13.4.3** (`https://developer.download.nvidia.com/compute/cuda/redist/redistrib_13.4.3.json`),
checks each SHA256 against the manifest and merges them into one `CUDA_HOME` (`C:\b\cuda`):
`cuda_nvcc, cuda_crt, libnvvm, cuda_cudart, cccl, cuda_nvtx, cuda_profiler_api, cuda_nvrtc,
cuda_cuobjdump, cuda_nvdisasm, libcublas, libcusparse, libcusolver, libcurand, libcufft, libnvjitlink`.
This is the same layout the installer produces (`bin/`, `include/`, `lib/x64/`, `nvvm/`).

`nvcc` 13.4 accepts MSVC 14.51 (VS 2026) natively; `-allow-unsupported-compiler` was not needed.

### Environment for every build (`build/env.ps1`)

- `vcvars64.bat` imported into PowerShell (x64 `cl.exe`)
- `CUDA_HOME = CUDA_PATH = C:\b\cuda`, `DISTUTILS_USE_SDK=1`, `MAX_JOBS` per package
- short build root `C:\b` (sources in `C:\b\src\...`, `TMP=C:\b\tmp`) — Windows long paths are off on this machine
- `pip wheel . --no-build-isolation --no-deps`

## Per-package details

| Package | Source | Arch flags | MAX_JOBS | Build time |
|---|---|---|---|---|
| sageattention 2.2.0.post7 | woct0rdho/SageAttention `v2.2.0-windows.post7` (`79cd62a`) | `TORCH_CUDA_ARCH_LIST="8.0 8.6 8.9 9.0 12.0"` (upstream CI list) | 4 | 8 min |
| sageattn3 1.0.0.post1 | thu-ml/SageAttention `d1a57a5`, `sageattention3_blackwell/`, CUTLASS `7f5fe3e` | `TORCH_CUDA_ARCH_LIST="12.0"` → upstream's `arch=compute_120a,code=sm_120a` | 2 | 8 min |
| nunchaku 1.3.0.dev20260906 | nunchux-ai/nunchaku `main` @ `302e0e9` (2026-09-06) | `NUNCHAKU_SM_TARGETS=120a` (patch) | 3 | 20 min |
| flash_attn 2.8.3.post2 | Dao-AILab/flash-attention `v2.8.3.post2` (`49774b3`) | `FLASH_ATTN_CUDA_ARCHS=120` → `sm_120` | 4 | 57 min |

All wheels: SASS only (no PTX), so they run on compute capability 12.0 GPUs (RTX 50xx) — plus, for
sageattention, sm80/86/89/90 as upstream ships. They will not JIT on other archs.

### sageattn3 1.0.0.post1

The first upload, `sageattn3-1.0.0+cu134torch2.15.0`, imported fine but `fp4attn_cuda.fwd` failed on the RTX 5080
with `CUDA error: misaligned address` for every shape (`fp4quant_cuda` worked). It had been built from
woct0rdho's repackaging of `sageattention3_blackwell` (TORCH_LIBRARY ops, abi3). The earlier claim here that its
kernels were identical to thu-ml's was wrong: `kernel_traits.h` and the bindings differ too.

Root cause, from comparing against the known-good `sageattn3-1.0.0+cu132torch2.14.0` Windows wheel (ussoewwin,
Hugging Face `ussoewwin/Sage-Attention-for-Windows`). Same 16 attention kernel instantiations (2 head dims ×
2 × bf16/fp16 × causal), so nothing was missing. Its 20 MB `fp4attn_cuda` holds sm_100a + sm_120a + sm_121a and
`-lineinfo`, about 6.7 MB per arch. **The kernel signatures differ, though:**

    reference:  compute_attn_ws<...>(Flash_fwd_params const*, CollectiveMainloopFwd<...>::Params const*, ...)
    1.0.0:      compute_attn_ws<...>(Flash_fwd_params, CollectiveMainloopFwd<...>::Params, ...)   // __grid_constant__ by value

The parameter structs hold the TMA descriptors (`CUtensorMap`, which must be 64-byte aligned). With MSVC the
by-value launch faults. The fix comes from Jianqiao Huang in
[mengqin/SageAttention `8bb81e4a`](https://github.com/mengqin/SageAttention/commit/8bb81e4a), "corrected the MSVC
parameter passing method", which ussoewwin's fork merges. Under `_MSC_VER` it copies the four structs into one
`cudaMallocAsync` buffer on the launch stream and passes pointers.

post1 is therefore built from **thu-ml/SageAttention `sageattention3_blackwell` @ `d1a57a5`**. It keeps upstream's
nvcc flags (`-O3 --use_fast_math -lineinfo -DQBLKSIZE=128 -DKBLKSIZE=128 -DCTA256 -DDQINRMEM -DNDEBUG
-DEXECMODE=0 --threads 4`, `-gencode arch=compute_120a,code=sm_120a`), upstream's pybind11 modules (`fp4attn_cuda`,
`fp4quant_cuda`, installed top-level as upstream's `api.py` imports them) and upstream's Python code, plus:

- `sageattn3-setup-gpuless-cutlass-pin-cxx20.patch` (setup.py):
  - target from `TORCH_CUDA_ARCH_LIST` instead of `torch.cuda.get_device_capability()`;
  - no unpinned `git clone cutlass` at build time;
  - `-std=c++20` instead of `-std=c++17`, plus `-Xcompiler /Zc:preprocessor /Zc:__cplusplus`. Both are required:
    torch 2.15 headers need C++20, and CUDA 13 CCCL needs the conforming preprocessor;
  - a version suffix.
- `sageattn3-msvc-kernel-params-by-pointer.patch`: the kernel/launch part of mengqin `8bb81e4a`. It is MSVC-only,
  and Linux builds still use `__grid_constant__`.
- CUTLASS pinned to **`7f5fe3edf123a336bf59b96e02cac7b13472aebd`** ("v4.3.4 update", main, 2025-12-21). Upstream's
  setup does `git clone --depth 1` of CUTLASS main and pins nothing. This is the commit that clone fetched when
  thu-ml last changed `sageattention3_blackwell` (`c03f15f`, 2025-12-22).
- The `PyMODINIT_FUNC` patch from 1.0.0 is not needed. Upstream uses `PYBIND11_MODULE`, which already exports
  `PyInit_*` with dllexport (checked with `dumpbin /exports`).
- MSVC 14.51 (VS 2026) was kept. The failure was the parameter-passing ABI, not the compiler version, so the
  VS 2022 v143 toolset was not needed.

cubin comparison (sm_120a, `cuobjdump -sass` / `-res-usage`), post1 vs the reference:
- `fp4attn_cuda`: the same 16 mangled kernel names, now with the pointer signature. Each uses 168 registers,
  1024 B static smem and 0 local memory, as in the reference. Stack is 0/48–56 B vs 0–32/224 B, since nvcc 13.4
  differs from 13.2.
- `fp4quant_cuda`: the same 12 kernels with identical register, stack and shared-memory usage.
- The reference also has cub's empty placeholder kernel (`cub::detail::EmptyKernel`) from CUDA 13.2's CCCL.

## Patches (all in `patches/`, applied by the `build/patch_*.py` scripts)

1. **`nunchaku-setup-sm-targets-and-version.patch`** — `setup.py`: in the default `FAST` mode the SM list
   comes from `torch.cuda.device_count()`, which is 0 without a GPU → `assert len(sm_targets) > 0`.
   Adds `NUNCHAKU_SM_TARGETS` (here `120a`). Also `NUNCHAKU_DEV_DATE` (dev date = commit date instead
   of build date) and `NUNCHAKU_LOCAL_VERSION` (`cu134torch2.15.0` instead of `cu13.4torch2.15`).
2. **`nunchaku-spdlog-fmt-nvcc13.4-char32.patch`** — `third_party/spdlog/.../fmt/bundled/format.h`
   (spdlog v1.14.1's bundled fmt): nvcc 13.4's front end rejects `U"\x9999999a..."` char32_t escapes above
   U+10FFFF with a hard `error: character value is out of range` (no diag number to suppress). The 8-entry
   constexpr lookup table is rewritten as a `switch` returning the same values.
3–4. **`sageattn3-setup-gpuless-cutlass-pin-cxx20.patch`** and **`sageattn3-msvc-kernel-params-by-pointer.patch`**:
   see [sageattn3 1.0.0.post1](#sageattn3-100post1) above.
5. **`flash-attn-setup-cxx20-zc-preprocessor.patch`** — `setup.py`: the Windows branch forces
   `/std:c++17` (cl) and `-std=c++17` (nvcc), but torch 2.15's headers require C++20 (`std::bit_width`,
   `std::strong_ordering`, designated initializers), and CUDA 13's CCCL `#error`s on MSVC's traditional
   preprocessor. Switched to C++20 and added `/Zc:preprocessor` for both cl and nvcc.

sageattention 2 needed no source patch (its version suffix `.post7+cu134torch2.15.0` is passed via
`SAGEATTENTION_WHEEL_VERSION_SUFFIX`).

Not a code change: `sageattention3_blackwell/` has no LICENSE file of its own, so `build_sageattn3.ps1`
copies the repository's Apache-2.0 `LICENSE` into it before building; the wheel then carries it in
`*.dist-info/licenses/`. The other three wheels already ship their upstream licence files.

## Problems / workarounds

- **No CUDA toolkit, no admin** → assembled CUDA 13.4.3 from redist archives (above).
- **PyPI slow / timing out** from this network: `uv pip` timed out against pypi.org; pip worked but slowly.
  One pip run lost its temp file under `%LOCALAPPDATA%\Temp` mid-install (`OSError: [Errno 2] ... pip-unpack-...`);
  using `TMP=C:\b\tmp` fixed it.
- **Windows PowerShell 5.1** treats native-command stderr as a terminating error under
  `$ErrorActionPreference = "Stop"` even when redirected; the build scripts switch to `Continue` around
  `pip wheel` and check `$LASTEXITCODE`. `vcvars64.bat` also needs the VS Installer dir on `PATH` for `vswhere`.
- **sageattention imports `triton`** at import time → verification venv has `triton-windows==3.9.0.post30`
  (the same version as the target machine).
- **nunchaku imports diffusers/transformers/accelerate/peft** at import time → verification venv has them
  (plus `torchvision 0.29.2+cu134`). The extension `_C.pyd` was also loaded directly to check DLL resolution.

## Verification done on the build machine (no GPU)

For every wheel (`build/verify_wheels.ps1`, full output in the release's `verify.txt`):

1. `pip install` into the clean torch-2.15.0+cu134 venv and import (`import sageattention`,
   `from sageattn3 import sageattn3_blackwell`, `import nunchaku`, `import flash_attn`) — all OK.
2. `dumpbin /dependents` on each `.pyd`: only torch 2.15 DLLs (`c10.dll`, `c10_cuda.dll`, `torch_cpu.dll`,
   `torch_cuda.dll`, `torch_python.dll`), the Python DLL, and the MSVC runtime. **No CUDA 12 DLLs.**
   The CUDA runtime is linked **statically** (`cudart_static.lib` from the 13.4 toolkit — there are no
   `__cudaRegisterFatBinary` imports), so no `cudart64_13.dll` import appears; cuBLAS etc. are reached through
   torch's own CUDA 13.4 DLLs.
3. `cuobjdump --list-elf` shows `sm_120a` (or `sm_120` for flash-attn) cubins in every CUDA module;
   `--list-ptx` is empty (SASS-only builds).
4. Wheel tags from the `WHEEL` metadata: `cp310-abi3-win_amd64` (sageattention — stable ABI),
   `cp313-cp313-win_amd64` (sageattn3, nunchaku, flash_attn).
5. No local user paths in any wheel file (checked by string search; cubins carry `-lineinfo` source paths under `C:/b/src` only).
