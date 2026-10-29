# SageAttention 2.2.0 (woct0rdho Windows fork) -> abi3 wheel, torch 2.15.0 + CUDA 13.4
param([string]$Root = "C:\b", [int]$MaxJobs = 4)
$ErrorActionPreference = "Stop"
. "$PSScriptRoot\env.ps1" -Root $Root -MaxJobs $MaxJobs

$src = "$Root\src\sage2"
if (-not (Test-Path $src)) {
    git clone --depth 1 --branch v2.2.0-windows.post7 https://github.com/woct0rdho/SageAttention $src
}

# Upstream CI arch list (keeps _qattn_sm80 / _qattn_sm89 / _qattn_sm90 / _fused); sm120 uses the sm89 kernels built as sm_120a.
$env:TORCH_CUDA_ARCH_LIST = "8.0 8.6 8.9 9.0 12.0"
$env:SAGEATTENTION_WHEEL_VERSION_SUFFIX = ".post7+cu134torch2.15.0"
$env:EXT_PARALLEL = "1"     # one extension at a time; nvcc --threads already parallelises per file

Push-Location $src
try {
    if (Test-Path build) { Remove-Item -Recurse -Force build }
    $t = Get-Date
    $ErrorActionPreference = "Continue"   # native stderr must not abort (PS 5.1)
    python -m pip wheel . --no-build-isolation --no-deps -v -w $WheelDir *> "$LogDir\sageattention2.log"
    if ($LASTEXITCODE) { throw "build failed, see $LogDir\sageattention2.log" }
    "sageattention2 built in $([int]((Get-Date)-$t).TotalMinutes) min"
} finally { Pop-Location }
