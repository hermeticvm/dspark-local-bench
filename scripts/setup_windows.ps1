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
New-Item -ItemType Directory -Force $binDir | Out-Null
foreach ($a in $assets) {
    $zip = Join-Path $env:TEMP $a.name
    Write-Host "download $($a.name)"
    Invoke-WebRequest $a.browser_download_url -OutFile $zip
    Expand-Archive $zip $binDir -Force
    Remove-Item $zip
}
$server = Get-ChildItem $binDir -Recurse -Filter llama-server.exe | Select-Object -First 1
if (-not $server) { throw "llama-server.exe not found after extracting into $binDir" }
& $server.FullName --list-devices

$src = Join-Path $root 'llama.cpp'
if (-not (Test-Path $src)) {
    git clone --depth 1 --branch $Tag https://github.com/ggml-org/llama.cpp $src
}

$venv = Join-Path $root '.venv'
if (-not (Test-Path $venv)) { python -m venv $venv }
& "$venv\Scripts\python.exe" -m pip install -q -r "$src\tools\server\bench\speed-bench\requirements.txt" huggingface_hub
Write-Host "`nsetup done. Activate the venv with: .venv\Scripts\Activate.ps1"
