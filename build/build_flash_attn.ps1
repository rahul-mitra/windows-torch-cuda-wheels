# flash-attn 2.8.3.post2 (FA2, not hopper/FA3) -> cp313 wheel, sm_120 only, torch 2.15.0 + CUDA 13.4
param([string]$Root = "C:\b", [int]$MaxJobs = 4)
$ErrorActionPreference = "Stop"
. "$PSScriptRoot\env.ps1" -Root $Root -MaxJobs $MaxJobs

$src = "$Root\src\fa"   # short path: flash-attn's nested sources exceed MAX_PATH otherwise
if (-not (Test-Path $src)) {
    git clone --depth 1 --branch v2.8.3.post2 --recurse-submodules --shallow-submodules `
        https://github.com/Dao-AILab/flash-attention $src
}

python -I "$PSScriptRoot\patch_flash_attn.py" $src

$env:FLASH_ATTN_CUDA_ARCHS       = "120"
$env:FLASH_ATTENTION_FORCE_BUILD  = "TRUE"   # never download a prebuilt wheel
$env:FLASH_ATTN_LOCAL_VERSION     = "cu134torch2.15.0"
$env:NVCC_THREADS                 = "2"

Push-Location $src
try {
    if (Test-Path build) { Remove-Item -Recurse -Force build }
    $t = Get-Date
    $ErrorActionPreference = "Continue"   # native stderr must not abort (PS 5.1)
    python -m pip wheel . --no-build-isolation --no-deps -v -w $WheelDir *> "$LogDir\flash_attn.log"
    if ($LASTEXITCODE) { throw "build failed, see $LogDir\flash_attn.log" }
    "flash-attn built in $([int]((Get-Date)-$t).TotalMinutes) min"
} finally { Pop-Location }
