# One-time setup on Windows (PowerShell 7):
#   .\scripts\setup_windows.ps1 -Backend vulkan
#   .\scripts\setup_windows.ps1 -Backend rocm
#   .\scripts\setup_windows.ps1 -Backend cuda
# Downloads the prebuilt llama.cpp release into bin\<backend>\, clones the matching
# llama.cpp source (for speed_bench.py) and creates a Python venv with its deps.
param(
    [Parameter(Mandatory)][ValidateSet('vulkan', 'rocm', 'cuda')][string]$Backend,
    [string]$Tag = 'b11865'
)
$ErrorActionPreference = 'Stop'
$root = Split-Path $PSScriptRoot -Parent

$release = Invoke-RestMethod "https://api.github.com/repos/ggml-org/llama.cpp/releases/tags/$Tag"
$pattern = @{ vulkan = 'vulkan'; rocm = 'rocm-*'; cuda = 'cuda-13*' }[$Backend]
$assets = @($release.assets | Where-Object { $_.name -like "llama-$Tag-bin-win-$pattern-x64.zip" })
if ($Backend -eq 'cuda') {
    $assets += @($release.assets | Where-Object { $_.name -like "cudart-llama-bin-win-$pattern-x64.zip" })
}
if (-not $assets) { throw "no Windows $Backend build found in release $Tag" }

$binDir = Join-Path $root "bin\$Backend"
# Windows locks DLLs that are in use, so unpacking over a running server fails halfway
$running = Get-Process llama-server -ErrorAction SilentlyContinue | Where-Object { $_.Path -like "$binDir\*" }
if ($running) { throw "llama-server from $binDir is running (a benchmark?); stop it first, then rerun setup" }
New-Item -ItemType Directory -Force $binDir | Out-Null
foreach ($a in $assets) {
    $zip = Join-Path $env:TEMP $a.name
    Write-Host "download $($a.name)"
    Invoke-WebRequest $a.browser_download_url -OutFile $zip
    # GitHub publishes a SHA-256 per release asset; refuse to unpack anything that doesn't match
    $actual = (Get-FileHash $zip -Algorithm SHA256).Hash.ToLower()
    if ($a.digest -and $a.digest -ne "sha256:$actual") {
        Remove-Item $zip
        throw "checksum mismatch for $($a.name): expected $($a.digest), got sha256:$actual"
    }
    if (-not $a.digest) { Write-Warning "no published checksum for $($a.name); not verified" }
    Expand-Archive $zip $binDir -Force
    Remove-Item $zip
}
$server = Get-ChildItem $binDir -Recurse -Filter llama-server.exe | Select-Object -First 1
if (-not $server) { throw "llama-server.exe not found after extracting into $binDir" }
if ($Backend -ne 'rocm') { & $server.FullName --list-devices }  # rocm lists devices once its libraries are in

$src = Join-Path $root 'llama.cpp'
if (-not (Test-Path $src)) {
    git clone --depth 1 --branch $Tag https://github.com/ggml-org/llama.cpp $src
}

$venv = Join-Path $root '.venv'
if (-not (Test-Path $venv)) { python -m venv $venv }
& "$venv\Scripts\python.exe" -m pip install -q -r "$root\requirements.txt"

if ($Backend -eq 'rocm') {
    # The ROCm zip bundles only the HIP runtime; ggml-hip.dll also needs hipBLAS/rocBLAS on PATH.
    # AMD ships them as pip wheels (about 3 GB); run_matrix.py puts them on PATH for rocm runs.
    $rocmVer = [regex]::Match($assets[0].name, 'rocm-(\d+\.\d+)').Groups[1].Value
    Write-Host "install AMD ROCm $rocmVer libraries into the venv (large download)"
    & "$venv\Scripts\python.exe" -m pip install -q --index-url https://stable.repo.amd.com/rocm/whl-next/ "rocm[libraries]==$rocmVer.*"
    $sp = & "$venv\Scripts\python.exe" -c "import sysconfig; print(sysconfig.get_paths()['purelib'])"
    $env:PATH = "$sp\_rocm_sdk_libraries\bin;$sp\_rocm_sdk_core\bin;$env:PATH"
    & $server.FullName --list-devices
}
Write-Host "`nsetup done. Activate the venv with: .venv\Scripts\Activate.ps1"
Write-Host "Pass the device of your graphics card to run_matrix.py, e.g. --server-args `"--device Vulkan0`"."
Write-Host "Device numbers differ between backends: check the list above."
