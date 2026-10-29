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
| sageattn3 1.0.0 | woct0rdho/SageAttention `v2.2.0-windows.post7` (`79cd62a`), `sageattention3_blackwell/`, CUTLASS submodule `f3fde58` | `TORCH_CUDA_ARCH_LIST="12.0"` → `sm_120a` | 1–2 | 6 min (13 min for the rebuild that ran alongside flash-attn) |
| nunchaku 1.3.0.dev20260906 | nunchux-ai/nunchaku `main` @ `302e0e9` (2026-09-06) | `NUNCHAKU_SM_TARGETS=120a` (patch) | 3 | 20 min |
| flash_attn 2.8.3.post2 | Dao-AILab/flash-attention `v2.8.3.post2` (`49774b3`) | `FLASH_ATTN_CUDA_ARCHS=120` → `sm_120` | 4 | 57 min |

All wheels: SASS only (no PTX), so they run on compute capability 12.0 GPUs (RTX 50xx) — plus, for
sageattention, sm80/86/89/90 as upstream ships. They will not JIT on other archs.

### Why sageattn3 is built from the woct0rdho fork

The user-specified source was `thu-ml/SageAttention/sageattention3_blackwell` (HEAD `d1a57a5`). That
`setup.py` cannot build on this machine as-is: it calls `torch.cuda.get_device_capability()` (needs a GPU)
and `git clone`s an unpinned CUTLASS at build time. woct0rdho's Windows-maintained fork carries the same
directory with **identical CUDA kernels** (verified with `diff`); its changes are only packaging:
entry points re-wrapped as `TORCH_LIBRARY` ops (`torch.ops.sageattn3_fp4attn_cuda`,
`torch.ops.sageattn3_fp4quant_cuda`) so the wheel is `abi3`, `TORCH_CUDA_ARCH_LIST` support, MSVC flags,
and CUTLASS pinned as a submodule. Import name and API are unchanged: `from sageattn3 import sageattn3_blackwell`.

## Patches (all in `patches/`, applied by the `build/patch_*.py` scripts)

1. **`nunchaku-setup-sm-targets-and-version.patch`** — `setup.py`: in the default `FAST` mode the SM list
   comes from `torch.cuda.device_count()`, which is 0 without a GPU → `assert len(sm_targets) > 0`.
   Adds `NUNCHAKU_SM_TARGETS` (here `120a`). Also `NUNCHAKU_DEV_DATE` (dev date = commit date instead
   of build date) and `NUNCHAKU_LOCAL_VERSION` (`cu134torch2.15.0` instead of `cu13.4torch2.15`).
2. **`nunchaku-spdlog-fmt-nvcc13.4-char32.patch`** — `third_party/spdlog/.../fmt/bundled/format.h`
   (spdlog v1.14.1's bundled fmt): nvcc 13.4's front end rejects `U"\x9999999a..."` char32_t escapes above
   U+10FFFF with a hard `error: character value is out of range` (no diag number to suppress). The 8-entry
   constexpr lookup table is rewritten as a `switch` returning the same values.
3. **`sageattn3-pymodinit-func-export.patch`** — `csrc/blackwell/api.cu`, `csrc/quantization/fp4_quantization_4d.cu`:
   the dummy module init functions were declared `PyObject* PyInit_x(void)` without `dllexport`. setuptools
   84 does not pass `/EXPORT:PyInit_x` to `link.exe`, so the first build imported with
   `ImportError: dynamic module does not define module export function (PyInit_fp4attn_cuda)`.
   Changed to `PyMODINIT_FUNC` (= `extern "C" __declspec(dllexport) PyObject*`), as sageattention 2's own
   `pybind.cpp` does.
4. **`flash-attn-setup-cxx20-zc-preprocessor.patch`** — `setup.py`: the Windows branch forces
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
4. Wheel tags from the `WHEEL` metadata: `cp310-abi3-win_amd64` (sageattention, sageattn3 — stable ABI),
   `cp313-cp313-win_amd64` (nunchaku, flash_attn).
5. No local user paths in any `.pyd` (`/pathmap` is applied by the SageAttention setups; checked by string search).
