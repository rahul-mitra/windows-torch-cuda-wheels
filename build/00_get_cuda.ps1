# Assemble a CUDA 13.4 toolkit (no admin, no installer) from NVIDIA's official
# per-component redistributable archives, merged into one CUDA_HOME tree.
param(
    [string]$Release = "13.4.3",
    [string]$Dest    = "C:\b\cuda",
    [string]$Cache   = "C:\b\dl"
)
$ErrorActionPreference = "Stop"
$base = "https://developer.download.nvidia.com/compute/cuda/redist"
$components = @(
    "cuda_nvcc", "cuda_crt", "libnvvm", "cuda_cudart", "cccl", "cuda_nvtx",
    "cuda_profiler_api", "cuda_nvrtc", "cuda_cuobjdump", "cuda_nvdisasm",
    "libcublas", "libcusparse", "libcusolver", "libcurand", "libcufft", "libnvjitlink"
)
$manifest = Invoke-RestMethod "$base/redistrib_$Release.json"
New-Item -ItemType Directory -Force $Dest, $Cache | Out-Null

foreach ($c in $components) {
    $info = $manifest.$c.'windows-x86_64'
    if (-not $info) { throw "component $c missing for windows-x86_64" }
    $zip = Join-Path $Cache (Split-Path $info.relative_path -Leaf)
    if (-not (Test-Path $zip) -or (Get-FileHash $zip -Algorithm SHA256).Hash -ne $info.sha256.ToUpper()) {
        Write-Host "downloading $c $($manifest.$c.version)"
        curl.exe -sSL --retry 3 -o $zip "$base/$($info.relative_path)"
        if ((Get-FileHash $zip -Algorithm SHA256).Hash -ne $info.sha256.ToUpper()) { throw "sha256 mismatch: $zip" }
    }
    $tmp = Join-Path $Cache "x_$c"
    if (Test-Path $tmp) { Remove-Item -Recurse -Force $tmp }
    tar.exe -xf $zip -C $Cache
    $top = Join-Path $Cache ([IO.Path]::GetFileNameWithoutExtension($zip))
    # merge bin/include/lib/nvvm/... into $Dest
    robocopy $top $Dest /E /NFL /NDL /NJH /NJS /NP | Out-Null
    Remove-Item -Recurse -Force $top
}
"CUDA $Release assembled at $Dest"
& "$Dest\bin\nvcc.exe" --version
