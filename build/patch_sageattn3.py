# Apply the sageattn3 patches to thu-ml/SageAttention sageattention3_blackwell
# (see patches/sageattn3-setup-gpuless-cutlass-pin-cxx20.patch and
#  patches/sageattn3-msvc-kernel-params-by-pointer.patch).
#
# setup.py (build only, no kernel/flag changes otherwise):
#  - target arch from TORCH_CUDA_ARCH_LIST instead of torch.cuda.get_device_capability() (no GPU here)
#  - no unpinned `git clone cutlass` at build time: csrc/cutlass must be the pinned checkout
#  - -std=c++20 + MSVC /Zc:preprocessor /Zc:__cplusplus (torch 2.15 headers need C++20, CUDA 13 CCCL
#    needs the conforming preprocessor)
#  - SAGEATTN3_VERSION_SUFFIX appended to the version
# kernel_ws.h / launch.h (MSVC only): pass the four kernel parameter structs through one device buffer
#  instead of by value as __grid_constant__. With MSVC the by-value launch faults with "misaligned address"
#  (the structs hold 64-byte-aligned CUtensorMap TMA descriptors). Port of Jianqiao Huang's fix in
#  mengqin/SageAttention 8bb81e4a, the same kernel signature as the known-good ussoewwin cu132 wheels.
# Usage: python patch_sageattn3.py <SageAttention-repo-dir>
import sys
from pathlib import Path

root = Path(sys.argv[1]) / "sageattention3_blackwell"
MARK = "[windows-torch-cuda-wheels]"


def patch(rel, pairs):
    f = root / rel
    raw = f.read_bytes().decode("utf-8")
    crlf = "\r\n" in raw
    s = raw.replace("\r\n", "\n")
    if MARK in s:
        print("already patched", f)
        return
    for old, new in pairs:
        if s.count(old) != 1:
            sys.exit(f"patch context not found (or not unique) in {f}:\n{old}")
        s = s.replace(old, new)
    f.write_bytes((s.replace("\n", "\r\n") if crlf else s).encode("utf-8"))
    print("patched", f)


patch("setup.py", [
    ("    cc_major, cc_minor = torch.cuda.get_device_capability()\n",
     f"    # {MARK} GPU-less build: take the target from TORCH_CUDA_ARCH_LIST (e.g. \"12.0\")\n"
     "    if os.getenv(\"TORCH_CUDA_ARCH_LIST\"):\n"
     "        cc_major, cc_minor = (int(x) for x in os.environ[\"TORCH_CUDA_ARCH_LIST\"].split()[0].split(\"+\")[0].split(\".\"))\n"
     "    else:\n"
     "        cc_major, cc_minor = torch.cuda.get_device_capability()\n"),
    ("    if not cutlass_dir.exists():\n"
     "        subprocess.run(\n"
     "            [\"git\", \"clone\", \"--depth\", \"1\", \"https://github.com/NVIDIA/cutlass.git\", str(cutlass_dir)],\n"
     "            check=True\n"
     "        )\n",
     f"    # {MARK} no unpinned clone at build time: csrc/cutlass must be the pinned CUTLASS checkout\n"
     "    if not (cutlass_dir / \"include\" / \"cutlass\").exists():\n"
     "        raise RuntimeError(f\"{cutlass_dir} missing: check out the pinned CUTLASS commit first\")\n"),
    ("        \"-std=c++17\",\n        \"-U__CUDA_NO_HALF_OPERATORS__\",\n",
     "        \"-std=c++20\",  # torch >= 2.13 headers need C++20\n        \"-U__CUDA_NO_HALF_OPERATORS__\",\n"),
    ("        \"-DDQINRMEM\",\n    ]\n",
     "        \"-DDQINRMEM\",\n    ]\n"
     f"    if os.name == \"nt\":  # {MARK} CUDA 13 CCCL needs MSVC's conforming preprocessor\n"
     "        nvcc_flags += [\"-Xcompiler\", \"/Zc:preprocessor\", \"-Xcompiler\", \"/Zc:__cplusplus\"]\n"),
    ("    version=\"1.0.0\",\n",
     "    version=\"1.0.0\" + os.getenv(\"SAGEATTN3_VERSION_SUFFIX\", \"\"),\n"),
])

patch("sageattn3/blackwell/kernel_ws.h", [
    ("template <typename Ktraits, bool Is_causal, typename TileScheduler>\n"
     "__global__ void __launch_bounds__(Ktraits::kNWarps * cutlass::NumThreadsPerWarp, 1)\n"
     "    compute_attn_ws(CUTE_GRID_CONSTANT Flash_fwd_params const params,\n"
     "                    CUTE_GRID_CONSTANT typename CollectiveMainloopFwd<Ktraits, Is_causal>::Params const mainloop_params,\n"
     "                    CUTE_GRID_CONSTANT typename CollectiveEpilogueFwd<Ktraits>::Params const epilogue_params,\n"
     "                    CUTE_GRID_CONSTANT typename TileScheduler::Params const scheduler_params\n"
     "                    ) {\n",
     f"// {MARK} kernel body moved to compute_attn_ws_impl; the __global__ wrappers are at the end of the file\n"
     "template <typename Ktraits, bool Is_causal, typename TileScheduler>\n"
     "__device__ inline void\n"
     "compute_attn_ws_impl(Flash_fwd_params const &params,\n"
     "                     typename CollectiveMainloopFwd<Ktraits, Is_causal>::Params const &mainloop_params,\n"
     "                     typename CollectiveEpilogueFwd<Ktraits>::Params const &epilogue_params,\n"
     "                     typename TileScheduler::Params const &scheduler_params) {\n"),
    ("\n} // namespace flash",
     "\n#if defined(_MSC_VER)\n"
     "// MSVC: parameters are passed through device memory (see launch.h); by-value __grid_constant__\n"
     "// parameters fault with \"misaligned address\" in Windows builds.\n"
     "template <typename Ktraits, bool Is_causal, typename TileScheduler>\n"
     "__global__ void __launch_bounds__(Ktraits::kNWarps * cutlass::NumThreadsPerWarp, 1)\n"
     "compute_attn_ws(Flash_fwd_params const *params,\n"
     "                typename CollectiveMainloopFwd<Ktraits, Is_causal>::Params const *mainloop_params,\n"
     "                typename CollectiveEpilogueFwd<Ktraits>::Params const *epilogue_params,\n"
     "                typename TileScheduler::Params const *scheduler_params) {\n"
     "    compute_attn_ws_impl<Ktraits, Is_causal, TileScheduler>(\n"
     "        *params, *mainloop_params, *epilogue_params, *scheduler_params);\n"
     "}\n"
     "#else\n"
     "template <typename Ktraits, bool Is_causal, typename TileScheduler>\n"
     "__global__ void __launch_bounds__(Ktraits::kNWarps * cutlass::NumThreadsPerWarp, 1)\n"
     "    compute_attn_ws(CUTE_GRID_CONSTANT Flash_fwd_params const params,\n"
     "                    CUTE_GRID_CONSTANT typename CollectiveMainloopFwd<Ktraits, Is_causal>::Params const mainloop_params,\n"
     "                    CUTE_GRID_CONSTANT typename CollectiveEpilogueFwd<Ktraits>::Params const epilogue_params,\n"
     "                    CUTE_GRID_CONSTANT typename TileScheduler::Params const scheduler_params\n"
     "                    ) {\n"
     "    compute_attn_ws_impl<Ktraits, Is_causal, TileScheduler>(\n"
     "        params, mainloop_params, epilogue_params, scheduler_params);\n"
     "}\n"
     "#endif  // _MSC_VER\n"
     "\n} // namespace flash"),
])

patch("sageattn3/blackwell/launch.h", [
    ("    cutlass::launch_kernel_on_cluster(launch_params, kernel, params, mainloop_params, epilogue_params, scheduler_params);\n",
     "#if defined(_MSC_VER)\n"
     f"    // {MARK} MSVC: copy the parameter structs into one stream-ordered device buffer and pass pointers\n"
     "    struct DeviceParamsPack {\n"
     "        Flash_fwd_params params;\n"
     "        typename CollectiveMainloop::Params mainloop;\n"
     "        typename CollectiveEpilogue::Params epilogue;\n"
     "        typename Scheduler::Params scheduler;\n"
     "    };\n"
     "    DeviceParamsPack h_pack{params, mainloop_params, epilogue_params, scheduler_params};\n"
     "    DeviceParamsPack *d_pack = nullptr;\n"
     "    C10_CUDA_CHECK(cudaMallocAsync(&d_pack, sizeof(DeviceParamsPack), stream));\n"
     "    C10_CUDA_CHECK(cudaMemcpyAsync(d_pack, &h_pack, sizeof(DeviceParamsPack), cudaMemcpyHostToDevice, stream));\n"
     "    // launch_kernel_on_cluster takes the address of each argument, so they must be lvalues\n"
     "    auto *d_params = &d_pack->params;\n"
     "    auto *d_mainloop_params = &d_pack->mainloop;\n"
     "    auto *d_epilogue_params = &d_pack->epilogue;\n"
     "    auto *d_scheduler_params = &d_pack->scheduler;\n"
     "    cutlass::launch_kernel_on_cluster(launch_params, kernel,\n"
     "        d_params, d_mainloop_params, d_epilogue_params, d_scheduler_params);\n"
     "    C10_CUDA_CHECK(cudaFreeAsync(d_pack, stream));\n"
     "#else\n"
     "    cutlass::launch_kernel_on_cluster(launch_params, kernel, params, mainloop_params, epilogue_params, scheduler_params);\n"
     "#endif\n"),
])
