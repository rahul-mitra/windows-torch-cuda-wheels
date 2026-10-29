# End-to-end: toolkit + venv + 4 wheels + verification. Run from any PowerShell (no admin needed).
# Prereqs: Visual Studio 2022/2026 with "Desktop development with C++", git, uv (or any CPython 3.13 x64).
param([string]$Root = "C:\b")
$ErrorActionPreference = "Stop"
New-Item -ItemType Directory -Force $Root, "$Root\src", "$Root\tmp" | Out-Null

# 1. CUDA 13.4 toolkit from NVIDIA redistributable archives (matches torch 2.15.0+cu134)
& "$PSScriptRoot\00_get_cuda.ps1" -Dest "$Root\cuda" -Cache "$Root\dl"

# 2. Python 3.13 venv with torch 2.15.0+cu134 (CPU is enough to build)
if (-not (Test-Path "$Root\venv")) { uv venv "$Root\venv" --python 3.13 --seed }
$py = "$Root\venv\Scripts\python.exe"
$env:TMP = "$Root\tmp"; $env:TEMP = "$Root\tmp"
& $py -m pip install "torch==2.15.0+cu134" --index-url https://download.pytorch.org/whl/cu134 --extra-index-url https://pypi.org/simple
& $py -m pip install wheel setuptools ninja packaging build numpy einops "triton-windows==3.9.0.post30"

# 3. Wheels (sources are cloned at the pinned refs by each script if missing)
& "$PSScriptRoot\build_sageattention2.ps1" -Root $Root -MaxJobs 4
& "$PSScriptRoot\build_sageattn3.ps1"      -Root $Root -MaxJobs 2
& "$PSScriptRoot\build_nunchaku.ps1"       -Root $Root -MaxJobs 3
& "$PSScriptRoot\build_flash_attn.ps1"     -Root $Root -MaxJobs 4

# 4. Verification (import, DLL deps, SASS/PTX, tags) -> $Root\logs\verify.txt
& "$PSScriptRoot\verify_wheels.ps1" -Root $Root
