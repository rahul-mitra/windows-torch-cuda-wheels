# Nunchaku (main) -> cp313 wheel, sm_120a only, torch 2.15.0 + CUDA 13.4
param([string]$Root = "C:\b", [int]$MaxJobs = 4, [string]$Commit = "302e0e97024ebd68688fe890e5df83731edf7b54")
$ErrorActionPreference = "Stop"
. "$PSScriptRoot\env.ps1" -Root $Root -MaxJobs $MaxJobs

$src = "$Root\src\nunchaku"
if (-not (Test-Path $src)) {
    git clone https://github.com/nunchux-ai/nunchaku $src
    git -C $src checkout $Commit
    git -C $src submodule update --init --recursive --depth 1
}
python -I "$PSScriptRoot\patch_nunchaku.py" $src

$env:NUNCHAKU_SM_TARGETS    = "120a"
$env:NUNCHAKU_BUILD_WHEELS  = "1"           # as upstream CI: no --generate-line-info
$env:NUNCHAKU_DEV_DATE      = (git -C $src log -1 --format=%cd --date=format:%Y%m%d)
$env:NUNCHAKU_LOCAL_VERSION = "cu134torch2.15.0"

Push-Location $src
try {
    if (Test-Path build) { Remove-Item -Recurse -Force build }
    $t = Get-Date
    $ErrorActionPreference = "Continue"   # native stderr must not abort (PS 5.1)
    python -m pip wheel . --no-build-isolation --no-deps -v -w $WheelDir *> "$LogDir\nunchaku.log"
    if ($LASTEXITCODE) { throw "build failed, see $LogDir\nunchaku.log" }
    "nunchaku built in $([int]((Get-Date)-$t).TotalMinutes) min"
} finally { Pop-Location }
