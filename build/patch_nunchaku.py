# Apply the Nunchaku setup.py patch (see patches/nunchaku-setup-sm-targets-and-version.patch).
# Usage: python patch_nunchaku.py <nunchaku-src-dir>
import sys
from pathlib import Path

p = Path(sys.argv[1]) / "setup.py"
s = p.read_text(encoding="utf-8")

old_sm = '''    install_mode = os.getenv("NUNCHAKU_INSTALL_MODE", "FAST")
    if install_mode == "FAST":'''
new_sm = '''    # Allow forcing SM targets on GPU-less build machines, e.g. NUNCHAKU_SM_TARGETS="120a"
    if os.getenv("NUNCHAKU_SM_TARGETS"):
        return os.getenv("NUNCHAKU_SM_TARGETS").replace(";", " ").replace(",", " ").split()

    install_mode = os.getenv("NUNCHAKU_INSTALL_MODE", "FAST")
    if install_mode == "FAST":'''

old_ver = '''    if "dev" in version:
        version = version + date.today().strftime("%Y%m%d")
    # Version format: 1.2.0+cu12.4torch2.4
    version = f"{version}+cu{cuda_version}torch{torch_major_minor_version}"'''
new_ver = '''    if "dev" in version:
        version = version + os.getenv("NUNCHAKU_DEV_DATE", date.today().strftime("%Y%m%d"))
    # Version format: 1.2.0+cu12.4torch2.4
    version = f"{version}+" + os.getenv("NUNCHAKU_LOCAL_VERSION", f"cu{cuda_version}torch{torch_major_minor_version}")'''

if new_sm in s:
    print("setup.py already patched")
else:
    for old, new in ((old_sm, new_sm), (old_ver, new_ver)):
        if old not in s:
            sys.exit(f"patch context not found:\n{old}")
        s = s.replace(old, new, 1)
    p.write_text(s, encoding="utf-8", newline="\n")
    print("patched", p)

# nvcc 13.4's front end rejects char32_t escapes above U+10FFFF ("character value is out of range"),
# which spdlog's bundled fmt uses as a constexpr lookup table. Same values, expressed as a switch.
f = Path(sys.argv[1]) / "third_party/spdlog/include/spdlog/fmt/bundled/format.h"
t = f.read_text(encoding="utf-8")
old_fmt = '''  return U"\\x9999999a\\x828f5c29\\x80418938\\x80068db9\\x8000a7c6\\x800010c7"
         U"\\x800001ae\\x8000002b"[index];'''
new_fmt = '''  switch (index) {
  case 0: return 0x9999999a;
  case 1: return 0x828f5c29;
  case 2: return 0x80418938;
  case 3: return 0x80068db9;
  case 4: return 0x8000a7c6;
  case 5: return 0x800010c7;
  case 6: return 0x800001ae;
  default: return 0x8000002b;
  }'''
if new_fmt in t:
    print("format.h already patched")
elif old_fmt in t:
    f.write_text(t.replace(old_fmt, new_fmt, 1), encoding="utf-8", newline="\n")
    print("patched", f)
else:
    sys.exit("fmt patch context not found")
