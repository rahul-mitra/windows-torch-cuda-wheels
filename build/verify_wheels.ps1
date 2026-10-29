# Verify built wheels on a GPU-less machine:
#  1. install into the torch-2.15.0+cu134 venv and import
#  2. dumpbin /dependents on every .pyd (torch DLLs + CUDA 13 runtime, no CUDA 12)
#  3. cuobjdump --list-elf / --list-ptx (sm_120 present)
#  4. wheel tag check
param([string]$Root = "C:\b", [string]$Report = "$Root\logs\verify.txt")
$ErrorActionPreference = "Continue"
. "$PSScriptRoot\env.ps1" -Root $Root

$imports = @{
    "sageattention" = "import sageattention, sageattention._fused, sageattention._qattn_sm80, sageattention._qattn_sm89, sageattention._qattn_sm90; from sageattention import sageattn; print('sageattention', sageattention.__file__)"
    "sageattn3"     = "import torch; from sageattn3 import sageattn3_blackwell; import sageattn3; print('sageattn3 ok', [o for o in ('sageattn3_fp4attn_cuda','sageattn3_fp4quant_cuda') if hasattr(torch.ops, o)])"
    "nunchaku"      = "import nunchaku, nunchaku._C; from nunchaku import NunchakuFluxTransformer2dModel; print('nunchaku', nunchaku._C.__file__)"
    "flash_attn"    = "import flash_attn, flash_attn_2_cuda; from flash_attn import flash_attn_func; print('flash_attn', flash_attn.__version__, flash_attn_2_cuda.__file__)"
}

Set-Content -Encoding utf8 $Report "verify run $(Get-Date -Format s)"
function Log($s) { Add-Content -Encoding utf8 $Report $s; Write-Host $s }

foreach ($whl in @(Get-ChildItem "$Root\wheels\*.whl")) {
    Log "================ $($whl.Name)"
    Log ("size: {0:N1} MB  sha256: {1}" -f ($whl.Length/1MB), (Get-FileHash $whl -Algorithm SHA256).Hash.ToLower())

    python -m pip install --no-deps --force-reinstall $whl.FullName 2>&1 | Select-Object -Last 1 | ForEach-Object { Log "pip: $_" }
    $pkg = ($whl.Name -split '-')[0]
    $key = @{ "sageattention"="sageattention"; "sageattn3"="sageattn3"; "nunchaku"="nunchaku"; "flash_attn"="flash_attn" }[$pkg]
    $r = python -c $imports[$key] 2>&1
    Log "import ($LASTEXITCODE): $($r -join ' | ')"

    $x = Join-Path "$Root\tmp" ("x_" + $pkg)
    if (Test-Path $x) { Remove-Item -Recurse -Force $x }
    New-Item -ItemType Directory $x | Out-Null
    tar.exe -xf $whl.FullName -C $x
    $tags = Get-ChildItem $x -Recurse -Filter WHEEL | Select-String '^Tag:' | ForEach-Object { $_.Line }
    Log ("WHEEL " + ($tags -join '; '))
    # personal paths anywhere in the wheel (binaries, metadata, sources)
    $leaks = Get-ChildItem $x -Recurse -File | Where-Object {
        Select-String -Path $_.FullName -Pattern $env:USERNAME, 'C:\Users', 'C:/Users' -SimpleMatch -Quiet
    } | ForEach-Object { $_.FullName.Substring($x.Length + 1) }
    Log ("files containing a user-profile path: " + $(if ($leaks) { $leaks -join ', ' } else { 'none' }))
    foreach ($pyd in Get-ChildItem $x -Recurse -Filter *.pyd) {
        Log "--- $($pyd.FullName.Substring($x.Length+1))  ($([int]($pyd.Length/1MB)) MB)"
        $deps = dumpbin /nologo /dependents $pyd.FullName | Where-Object { $_ -match '\.dll' } | ForEach-Object { $_.Trim() }
        Log ("deps: " + ($deps -join ', '))
        if ($deps -match 'cudart64_12|cublas64_12') { Log "!! CUDA 12 DEPENDENCY" }
        $elf = cuobjdump --list-elf $pyd.FullName 2>&1 | ForEach-Object { if ($_ -match 'sm_\d+a?') { $Matches[0] } } | Sort-Object -Unique
        $ptx = cuobjdump --list-ptx $pyd.FullName 2>&1 | ForEach-Object { if ($_ -match 'sm_\d+a?') { $Matches[0] } } | Sort-Object -Unique
        Log ("SASS: " + ($elf -join ' ') + "   PTX: " + ($ptx -join ' '))
    }
}
"report: $Report"
