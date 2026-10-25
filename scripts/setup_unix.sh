#!/usr/bin/env bash
# One-time setup on Linux or macOS:
#   ./scripts/setup_unix.sh vulkan     # Linux: AMD, Intel or NVIDIA through Vulkan
#   ./scripts/setup_unix.sh cuda       # Linux: NVIDIA
#   ./scripts/setup_unix.sh rocm       # Linux: AMD (needs ROCm installed on the system)
#   ./scripts/setup_unix.sh metal      # macOS
# Downloads the prebuilt llama.cpp release into bin/<backend>/, clones the matching
# llama.cpp source (for speed_bench.py) and creates a Python venv with its deps.
set -euo pipefail

backend="${1:-}"
tag="${2:-b11865}"
root="$(cd "$(dirname "$0")/.." && pwd)"

case "$(uname -m)" in
    x86_64 | amd64) arch=x64 ;;
    arm64 | aarch64) arch=arm64 ;;
    *) echo "unsupported CPU architecture: $(uname -m)" >&2; exit 1 ;;
esac
case "$(uname -s)-$backend" in
    Darwin-metal) patterns=("llama-$tag-bin-macos-$arch.tar.gz") ;;
    Linux-vulkan) patterns=("llama-$tag-bin-ubuntu-vulkan-$arch.tar.gz") ;;
    Linux-rocm)   patterns=("llama-$tag-bin-ubuntu-rocm-*-$arch.tar.gz") ;;
    Linux-cuda)   patterns=("llama-$tag-bin-ubuntu-cuda-13*-$arch.tar.gz" "cudart-llama-$tag-bin-ubuntu-cuda-13*-$arch.tar.gz") ;;
    *) echo "usage: $0 {vulkan|cuda|rocm} on Linux, $0 metal on macOS  [tag, default b11865]" >&2; exit 1 ;;
esac

# fail before any download if Python can't make a venv (fresh Ubuntu/Debian lacks python3-venv)
if [ ! -d "$root/.venv" ] && ! python3 -c "import ensurepip" 2>/dev/null; then
    echo "python3 can't create virtual environments here; install it first, e.g. sudo apt install python3-venv" >&2
    exit 1
fi

release_json="$(curl -fsSL "https://api.github.com/repos/ggml-org/llama.cpp/releases/tags/$tag")"
bin_dir="$root/bin/$backend"
mkdir -p "$bin_dir"
sha256() { if command -v sha256sum >/dev/null; then sha256sum "$1"; else shasum -a 256 "$1"; fi | cut -d' ' -f1; }
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
for pattern in "${patterns[@]}"; do
    # prints "<url> <sha256 or empty>" for the first matching asset
    read -r url digest <<< "$(python3 -c '
import fnmatch, json, sys
assets = json.load(sys.stdin)["assets"]
hits = [a for a in assets if fnmatch.fnmatch(a["name"], sys.argv[1])]
if hits: print(hits[0]["browser_download_url"], (hits[0].get("digest") or "").removeprefix("sha256:"))' "$pattern" <<< "$release_json")"
    if [ -z "${url:-}" ]; then echo "no $pattern in release $tag" >&2; exit 1; fi
    file="$tmp/$(basename "$url")"
    echo "download $(basename "$url")"
    curl -fsSL "$url" -o "$file"
    # GitHub publishes a SHA-256 per release asset; refuse to unpack anything that doesn't match
    if [ -n "${digest:-}" ]; then
        actual="$(sha256 "$file")"
        if [ "$actual" != "$digest" ]; then echo "checksum mismatch for $(basename "$url"): expected $digest, got $actual" >&2; exit 1; fi
    else
        echo "warning: no published checksum for $(basename "$url"); not verified" >&2
    fi
    tar -xzf "$file" -C "$bin_dir"
done
server="$(find "$bin_dir" -name llama-server -type f | head -n 1)"
if [ -z "$server" ]; then echo "llama-server not found after extracting into $bin_dir" >&2; exit 1; fi
if ! "$server" --list-devices; then
    echo "warning: llama-server didn't start, most likely a missing system library (see 'Linux requirements' in the README)" >&2
fi

if [ ! -d "$root/llama.cpp" ]; then
    git -c advice.detachedHead=false clone -q --depth 1 --branch "$tag" https://github.com/ggml-org/llama.cpp "$root/llama.cpp"
fi
if [ ! -d "$root/.venv" ]; then python3 -m venv "$root/.venv"; fi
"$root/.venv/bin/python" -m pip install -q -r "$root/requirements.txt"
echo
echo "setup done. Activate the venv with: . .venv/bin/activate"
