# Common build environment. Dot-source it:  . .\build\env.ps1
# Imports the MSVC x64 environment (vcvars64.bat) into this PowerShell session
# and points torch.utils.cpp_extension at the CUDA 13.4 toolkit.
param(
    [string]$Root    = "C:\b",
    [string]$Vcvars  = "",
    [int]$MaxJobs    = 4
)

if (-not $Vcvars) {
    $vs = & "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe" -latest -products * `
        -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
    $Vcvars = Join-Path $vs "VC\Auxiliary\Build\vcvars64.bat"
}
$env:PATH = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer;$env:PATH"   # vcvars calls vswhere
cmd.exe /c "`"$Vcvars`" >nul 2>&1 && set" | ForEach-Object {
    if ($_ -match '^([^=]+)=(.*)$') { Set-Item -Path "env:$($Matches[1])" -Value $Matches[2] }
}

$env:CUDA_HOME  = "$Root\cuda"
$env:CUDA_PATH  = "$Root\cuda"
$env:PATH       = "$Root\venv\Scripts;$Root\cuda\bin;$env:PATH"
$env:DISTUTILS_USE_SDK = "1"
$env:MAX_JOBS   = "$MaxJobs"
# Keep temp files on a short path (nvcc/cl intermediates).
$env:TMP  = "$Root\tmp"; $env:TEMP = "$Root\tmp"
New-Item -ItemType Directory -Force "$Root\tmp", "$Root\wheels", "$Root\logs" | Out-Null

$script:WheelDir = "$Root\wheels"
$script:LogDir   = "$Root\logs"
