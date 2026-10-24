# dspark-local-bench

**Does DSpark speculative decoding actually help on *your* GPU, for *one* user?**

[DSpark](https://huggingface.co/papers/2607.05147) is DeepSeek's speculative decoding method. A small
drafter proposes a block of tokens in one pass, with a semi-autoregressive "Markov head", and the target
model verifies them. llama.cpp has supported it since
[#25173](https://github.com/ggml-org/llama.cpp/pull/25173) (`--spec-type draft-dspark`), and Ollama runs it
through `DRAFT` in a Modelfile.

The published numbers come mostly from NVIDIA datacenter or high-end CUDA cards, and the paper's confidence
scheduler is designed for a server under load. This repo measures what a single local user gets:

- every llama.cpp speculative method (none, n-gram, classic draft model, DFlash, DSpark) on the **same model,
  prompts and GPU**, at batch size 1
- a sweep of DSpark's draft length (`--spec-draft-n-max`) against its confidence cutoff
  (`--spec-draft-p-min`), which upstream ships with no tuning data
- the cost to prompt processing (time to first token), which drops sharply when a drafter is loaded
- backends side by side: Vulkan vs ROCm vs CUDA vs Metal

It uses llama.cpp's own [SPEED-Bench client](https://github.com/ggml-org/llama.cpp/tree/master/tools/server/bench/speed-bench),
so the numbers line up with the ones in upstream PRs. **Anyone can run it and submit their GPU's results.**

> **Status: work in progress.** The harness is ready; the first results (RX 7900 GRE, Vulkan + ROCm) are coming.
> See [docs/status-2026-10.md](docs/status-2026-10.md) for the state of DSpark across the ecosystem.

## Quickstart: Windows

PowerShell 7, Python 3.10+, git, about 15 GB of free disk space.

```powershell
git clone https://github.com/juraj-jumic/dspark-local-bench; cd dspark-local-bench
.\scripts\setup_windows.ps1 -Backend vulkan      # or rocm / cuda; prebuilt llama.cpp, no compiler needed
.venv\Scripts\Activate.ps1
python scripts\fetch_models.py configs\qwen3-8b.json

# smoke test, about 2 minutes
python scripts\run_matrix.py --config configs\qwen3-8b.json --backend vulkan --machine rx7900gre-win11 `
    --gpu "AMD Radeon RX 7900 GRE 16GB" --only baseline,dspark-n7-p0 --limit 1

# full matrix, about 1.5-2 h; finished runs are skipped if you stop and restart
python scripts\run_matrix.py --config configs\qwen3-8b.json --backend vulkan --machine rx7900gre-win11 `
    --gpu "AMD Radeon RX 7900 GRE 16GB"
python scripts\summarize.py --write
```

## Quickstart: Linux / macOS

Build llama.cpp at the pinned tag with your backend ([build docs](https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md)),
then point `--server` at the binary:

```bash
git clone https://github.com/juraj-jumic/dspark-local-bench && cd dspark-local-bench
git clone --depth 1 --branch b11865 https://github.com/ggml-org/llama.cpp
cmake -S llama.cpp -B llama.cpp/build -DGGML_CUDA=ON   # or -DGGML_VULKAN=ON / -DGGML_HIP=ON; Metal is on by default on macOS
cmake --build llama.cpp/build --config Release -j
python -m venv .venv && . .venv/bin/activate
pip install -r llama.cpp/tools/server/bench/speed-bench/requirements.txt huggingface_hub
python scripts/fetch_models.py configs/qwen3-8b.json
python scripts/run_matrix.py --config configs/qwen3-8b.json --backend cuda --machine rtx4090-linux \
    --server llama.cpp/build/bin/llama-server
python scripts/summarize.py --write
```

## What gets measured

`configs/qwen3-8b.json` (target Qwen3-8B Q8_0, 8k context, about 11-12.5 GB VRAM):

| run id | method |
|---|---|
| `baseline` | no speculation |
| `ngram-mod` | n-gram self-speculation, no extra model |
| `draft-simple-n{3,5}` | classic small draft model (Qwen3-0.6B) |
| `dflash-n{4,7}` | DFlash block-diffusion drafter |
| `dspark-n{2,3,4,5,7}-p{0,0.3,0.5,0.7,0.9}` | DSpark: draft length × confidence cutoff (`p0` = cutoff off) |
| `dspark-bf16-n{4,7}-p0` | DSpark with an unquantized drafter |

`configs/qwen3-8b-longctx.json` repeats the key runs with fixed 8k-token prompts.

The prompts are SPEED-Bench `qualitative` (coding, math, writing, qa), greedy decoding, Qwen3 thinking
off (the DSpark drafters were trained on non-thinking output), 512 output tokens, one request at a time.

For each run, `summarize.py` reports decode t/s, speedup vs `baseline`, prompt t/s and its ratio to
baseline, draft acceptance rate, and per-category speedups.

Each run is a JSON entry in `configs/`, so adding a model or sweep doesn't need any code. A `sweep` expands
`{placeholders}` across every combination of the listed values.

## Contributing results

1. Run a config on your machine. Use a short, non-identifying `--machine` label such as `rtx3060-win11`;
   it becomes the folder name.
2. Commit `results/<machine>/<config>__<backend>.json` (and the `.md` from `summarize.py --write`).
   Don't commit `runs/`, which holds raw per-request data and server logs and is gitignored.
3. Open a PR. Note anything unusual: power limits, other GPU load, laptop thermals.

## Layout

```
configs/    benchmark matrices (JSON)
scripts/    setup_windows.ps1, fetch_models.py, run_matrix.py, summarize.py
results/    condensed, committed results per machine
docs/       background and ecosystem status
runs/       raw output and server logs (gitignored)
bin/ models/ llama.cpp/   downloaded tools and models (gitignored)
```

## Notes

- Code: MIT, see [LICENSE](LICENSE). Models keep their own licenses (Qwen3: Apache-2.0; the DeepSeek
  drafters: see their model cards).
- SPEED-Bench prompts are covered by NVIDIA's evaluation dataset license. This repo never redistributes
  them; `speed_bench.py` downloads them from Hugging Face, and we publish only aggregate timings.
- Not affiliated with DeepSeek, ggml-org or NVIDIA.
