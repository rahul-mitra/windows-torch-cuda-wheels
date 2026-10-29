# Apply the sageattn3 patch (see patches/sageattn3-pymodinit-func-export.patch).
# The module init stubs are declared as plain `PyObject* PyInit_x(void)`, which only works when
# setuptools passes /EXPORT:PyInit_x to link.exe; current setuptools does not, so the .pyd fails
# with "dynamic module does not define module export function". PyMODINIT_FUNC adds dllexport.
# Usage: python patch_sageattn3.py <SageAttention-repo-dir>
import sys
from pathlib import Path

root = Path(sys.argv[1]) / "sageattention3_blackwell" / "csrc"
for rel, name in (("blackwell/api.cu", "PyInit_fp4attn_cuda"),
                  ("quantization/fp4_quantization_4d.cu", "PyInit_fp4quant_cuda")):
    f = root / rel
    s = f.read_text(encoding="utf-8")
    old, new = f"PyObject* {name}(void)", f"PyMODINIT_FUNC {name}(void)"
    if new in s:
        print("already patched", f)
        continue
    if old not in s:
        sys.exit(f"patch context not found in {f}")
    f.write_text(s.replace(old, new, 1), encoding="utf-8", newline="\n")
    print("patched", f)
