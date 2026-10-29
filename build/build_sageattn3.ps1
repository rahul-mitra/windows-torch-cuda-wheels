# SageAttention 3 (sageattn3, Blackwell FP4) -> abi3 wheel, sm_120a only.
# Built from the Windows-ported copy in woct0rdho/SageAttention (same kernels as
# thu-ml/SageAttention sageattention3_blackwell, wrapped as TORCH_LIBRARY ops; CUTLASS pinned as submodule).
param([string]$Root = "C:\b", [int]$MaxJobs = 4)
$ErrorActionPreference = "Stop"
. "$PSScriptRoot\env.ps1" -Root $Root -MaxJobs $MaxJobs

$repo = "$Root\src\sage2"
if (-not (Test-Path $repo)) {
    git clone --depth 1 --branch v2.2.0-windows.post7 https://github.com/woct0rdho/SageAttention $repo
}
git -C $repo submodule update --init --depth 1 sageattention3_blackwell/cutlass
python -I "$PSScriptRoot\patch_sageattn3.py" $repo
# The subproject has no LICENSE of its own; ship the repo's Apache-2.0 licence in the wheel's dist-info.
Copy-Item "$repo\LICENSE" "$repo\sageattention3_blackwell\LICENSE" -Force

$env:TORCH_CUDA_ARCH_LIST = "12.0"
$env:SAGEATTENTION_WHEEL_VERSION_SUFFIX = "+cu134torch2.15.0"

Push-Location "$repo\sageattention3_blackwell"
try {
    if (Test-Path build) { Remove-Item -Recurse -Force build }
    $t = Get-Date
    $ErrorActionPreference = "Continue"   # native stderr must not abort (PS 5.1)
    python -m pip wheel . --no-build-isolation --no-deps -v -w $WheelDir *> "$LogDir\sageattn3.log"
    if ($LASTEXITCODE) { throw "build failed, see $LogDir\sageattn3.log" }
    "sageattn3 built in $([int]((Get-Date)-$t).TotalMinutes) min"
} finally { Pop-Location }
