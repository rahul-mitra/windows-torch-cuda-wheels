# SageAttention 3 (sageattn3, Blackwell FP4) -> cp313 wheel, sm_120a only.
# Built from thu-ml/SageAttention sageattention3_blackwell with upstream's nvcc flags, CUTLASS pinned to the
# main commit upstream's unpinned `git clone --depth 1` fetched when sageattention3_blackwell last changed
# (2025-12-22), and the patches applied by patch_sageattn3.py.
param([string]$Root = "C:\b", [int]$MaxJobs = 2)
$ErrorActionPreference = "Stop"
. "$PSScriptRoot\env.ps1" -Root $Root -MaxJobs $MaxJobs

$repo = "$Root\src\sage3"
$sageCommit = "d1a57a546c3d395b1ffcbeecc66d81db76f3b4b5"
$cutlassCommit = "7f5fe3edf123a336bf59b96e02cac7b13472aebd"   # NVIDIA/cutlass main "v4.3.4 update. (#2892)"
$ErrorActionPreference = "Continue"   # git writes progress to stderr (PS 5.1)
if (-not (Test-Path $repo)) {
    git init -q $repo
    git -C $repo fetch -q --depth 1 https://github.com/thu-ml/SageAttention $sageCommit
    git -C $repo checkout -q FETCH_HEAD
}
$cutlass = "$repo\sageattention3_blackwell\csrc\cutlass"
if (-not (Test-Path $cutlass)) {
    git init -q $cutlass
    git -C $cutlass fetch -q --depth 1 https://github.com/NVIDIA/cutlass.git $cutlassCommit
    git -C $cutlass checkout -q FETCH_HEAD
}
$ErrorActionPreference = "Stop"
if ((git -C $cutlass rev-parse HEAD) -ne $cutlassCommit) { throw "CUTLASS is not at $cutlassCommit" }
python -I "$PSScriptRoot\patch_sageattn3.py" $repo
# The subproject has no LICENSE of its own; ship the repo's Apache-2.0 licence in the wheel's dist-info.
Copy-Item "$repo\LICENSE" "$repo\sageattention3_blackwell\LICENSE" -Force

$env:TORCH_CUDA_ARCH_LIST = "12.0"   # -> -gencode arch=compute_120a,code=sm_120a (upstream's sm_120 branch)
$env:SAGEATTN3_VERSION_SUFFIX = ".post1+cu134torch2.15.0"

Push-Location "$repo\sageattention3_blackwell"
try {
    if (Test-Path build) { Remove-Item -Recurse -Force build }
    $t = Get-Date
    $ErrorActionPreference = "Continue"   # native stderr must not abort (PS 5.1)
    python -m pip wheel . --no-build-isolation --no-deps -v -w $WheelDir *> "$LogDir\sageattn3.log"
    if ($LASTEXITCODE) { throw "build failed, see $LogDir\sageattn3.log" }
    "sageattn3 built in $([int]((Get-Date)-$t).TotalMinutes) min"
} finally { Pop-Location }
