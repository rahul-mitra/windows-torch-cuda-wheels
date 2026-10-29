# Apply the flash-attn setup.py patch (see patches/flash-attn-setup-cxx20-zc-preprocessor.patch).
# torch 2.15 headers require C++20, and CUDA 13.x CCCL requires MSVC's conforming preprocessor.
# Usage: python patch_flash_attn.py <flash-attention-src-dir>
import sys
from pathlib import Path

p = Path(sys.argv[1]) / "setup.py"
s = p.read_text(encoding="utf-8")

edits = [
    ('''    nvcc_flags = [
    "-O3",
    "-std=c++17",''', '''    nvcc_flags = [
    "-O3",
    "-std=c++20",'''),
    ('''        nvcc_flags.extend(["-Xcompiler", "/Zc:__cplusplus"])
        compiler_c17_flag=["-O2", "/std:c++17", "/Zc:__cplusplus"]''',
     '''        nvcc_flags.extend(["-Xcompiler", "/Zc:__cplusplus", "-Xcompiler", "/Zc:preprocessor"])
        compiler_c17_flag=["-O2", "/std:c++20", "/Zc:__cplusplus", "/Zc:preprocessor"]'''),
]
if edits[1][1] in s:
    print("already patched")
    sys.exit(0)
for old, new in edits:
    if old not in s:
        sys.exit(f"patch context not found:\n{old}")
    s = s.replace(old, new, 1)
p.write_text(s, encoding="utf-8", newline="\n")
print("patched", p)
