<!-- markdownlint-disable first-line-h1 no-inline-html -->

<div align="center">
  <img src="assets/banner.svg" width="80%" alt="dspark-local-bench" />
</div>
<hr>
<div align="center">
  <a href="LICENSE"><img alt="License: MIT" src="https://img.shields.io/badge/License-MIT-2a78d6"/></a>
  <a href="https://github.com/ggml-org/llama.cpp/releases/tag/b11865"><img alt="llama.cpp b11865" src="https://img.shields.io/badge/llama.cpp-b11865-555555"/></a>
  <a href="#5-contribute-your-gpu"><img alt="Backends" src="https://img.shields.io/badge/Backends-Vulkan%20%C2%B7%20ROCm%20%C2%B7%20CUDA%20%C2%B7%20Metal-555555"/></a>
  <a href="https://arxiv.org/abs/2607.05147"><img alt="DSpark paper" src="https://img.shields.io/badge/Paper-arXiv%202607.05147-b31b1b"/></a>
  <br>
  <b>Does DSpark speculative decoding make <i>your</i> local LLM faster?</b><br>
  Reproducible single-user benchmarks for llama.cpp on consumer GPUs. Run them on yours and add a row.
</div>

## Table of Contents

1. [Introduction](#1-introduction)
2. [Results](#2-results)
3. [Quick Start](#3-quick-start)
4. [What Gets Measured](#4-what-gets-measured)
5. [Contribute Your GPU](#5-contribute-your-gpu)
6. [Ecosystem Status](#6-ecosystem-status)
7. [License](#7-license)

## 1. Introduction

**Speculative decoding** makes text generation faster without changing the output. A small *draft* model
guesses the next few tokens, and the big model checks all of them in one pass. Every guess it accepts is a
token you didn't have to wait for.

[**DSpark**](https://huggingface.co/papers/2607.05147) is DeepSeek's take on it: the drafter proposes a
whole block of tokens at once, and a trained *confidence head* predicts which of them will survive.
llama.cpp has supported it since July 2026 (`--spec-type draft-dspark`), and Ollama runs it through `DRAFT`
in a Modelfile.

The published speedups come from servers and high-end NVIDIA cards. This project asks the questions a
**single local user** cares about:

- **How much faster is it really** on a consumer GPU, compared with every other speculative method in llama.cpp?
- **Which settings work best:** how many tokens to draft, and whether the confidence cutoff helps or hurts?
- **What does it cost:** how much slower is prompt processing, which you feel as the wait before the first token?

## 2. Results

<!-- RESULTS:START -->
**One card, two backends:** AMD Radeon RX 7900 GRE 16 GB, Qwen3-8B Q8_0, 32 prompts per setting.

| | Vulkan | ROCm |
|---|---|---|
| No speculation | 62 tok/s | 59 tok/s |
| **DSpark, best setting** | 115 tok/s · 1.86× · drafts 3 tokens | **133 tok/s · 2.25× · drafts 7 tokens** |
| DFlash | 95 tok/s · 1.54× | 110 tok/s · 1.86× |
| Small draft model (Qwen3-0.6B) | 84 tok/s · 1.36× | 76 tok/s · 1.29× |

- **DSpark is the fastest method on both backends.** ROCm with DSpark is the fastest setup overall, at
  133 tok/s, 2.25× plain decoding. n-gram lookup does almost nothing on these prompts (1.02×).
- **The best draft length depends on the backend.** Vulkan is fastest with 3-token drafts and drops to 1.63×
  with 7. ROCm is the opposite: fastest with 7 and only 1.85× with 3. The likely reason is that ROCm checks
  a block of drafted tokens more cheaply: it processes prompts 24% faster, while plain decoding is slightly
  slower. **Any fixed setting gives up 13–18% on one backend**, which is the case for drafts that adapt their length.
- **The confidence cutoff helps only a little.** On Vulkan, `--spec-draft-p-min 0.3` adds about 3%, and
  cutoffs of 0.5 and above make it slower. At 0.9 it drops to 1.28×.
- **Keep the drafter at Q8_0.** The BF16 drafter was slower (1.55× vs 1.72× at 4 tokens) and no more accurate.
- **Prompt processing barely suffers:** about 8–9% slower with the drafter loaded, on both backends.
- **The gain depends on the task:** coding gains the most (2.3× on Vulkan, 3.2× on ROCm), Q&A the least (1.5–1.6×).

<p align="center">
  <img src="assets/speedup-rocm.svg" width="90%" alt="Decode speedup per method on ROCm" />
  <br><br>
  <img src="assets/speedup-vulkan.svg" width="90%" alt="Decode speedup per method on Vulkan" />
</p>

The fastest commands on this machine:

```bash
# ROCm: needs the ROCm libraries (setup_windows.ps1 -Backend rocm installs them)
llama-server -m Qwen3-8B-Q8_0.gguf -md dspark-Qwen3-8B-Q8_0.gguf --spec-type draft-dspark --spec-draft-n-max 7 -ngl all -ngld all -fa on --device ROCm1
# Vulkan
llama-server -m Qwen3-8B-Q8_0.gguf -md dspark-Qwen3-8B-Q8_0.gguf --spec-type draft-dspark --spec-draft-n-max 3 --spec-draft-p-min 0.3 -ngl all -ngld all -fa on --device Vulkan0
```

> [!NOTE]
> One GPU, one model and greedy decoding so far, with 8 prompts per category. Treat differences under about 3%
> as noise. The 21-setting quick screen, which includes the cutoff and BF16 results, ran on Vulkan only. ROCm was
> measured at draft lengths 3, 5 and 7. Other GPUs may prefer other settings, which is exactly what
> [contributed results](#5-contribute-your-gpu) will show.
<!-- RESULTS:END -->

Full tables, with per-category speedups and acceptance rates: [Vulkan](results/rx7900gre-win11/qwen3-8b__vulkan.md),
[ROCm](results/rx7900gre-win11/qwen3-8b__rocm.md) and the [Vulkan quick screen](results/rx7900gre-win11/qwen3-8b-screen__vulkan.md).

## 3. Quick Start

You need Python 3.10+, git and about 15 GB of disk. The setup script downloads a prebuilt llama.cpp, so
there's nothing to compile.

**1. Set up** for your system and GPU:

| System | GPU | Command |
|---|---|---|
| Windows (PowerShell 7) | any (AMD, Intel, NVIDIA) | `.\scripts\setup_windows.ps1 -Backend vulkan` |
| Windows (PowerShell 7) | NVIDIA | `.\scripts\setup_windows.ps1 -Backend cuda` |
| Windows (PowerShell 7) | AMD Radeon (tested: RX 7900 GRE) | `.\scripts\setup_windows.ps1 -Backend rocm` (adds about 3 GB of AMD libraries) |
| Linux | any | `./scripts/setup_unix.sh vulkan` |
| Linux | NVIDIA / AMD | `./scripts/setup_unix.sh cuda` or `rocm` (ROCm must be installed on the system) |
| macOS (Apple Silicon) | built-in | `./scripts/setup_unix.sh metal` |

```bash
git clone https://github.com/juraj-jumic/dspark-local-bench
cd dspark-local-bench
# run your setup command from the table, then activate the environment:
.venv\Scripts\Activate.ps1          # Windows
. .venv/bin/activate                # Linux / macOS
python scripts/fetch_models.py configs/qwen3-8b.json
```

**2. Pick your GPU.** The setup prints the devices llama.cpp sees, for example:

```text
Vulkan0: AMD Radeon RX 7900 GRE (16368 MiB)
Vulkan1: AMD Radeon(TM) Graphics          <- integrated GPU, don't use
```

> [!IMPORTANT]
> If an integrated GPU is listed next to your graphics card, pass the card with `--device`, or llama.cpp may
> split the model across both. **The numbering can change between backends:** on the machine above, the
> same card is `Vulkan0` but `ROCm1`. On a Mac, or with a single GPU, leave `--server-args` out.

**3. Run** a two-minute smoke test, then the full matrix (about 2 hours; finished settings are skipped if you restart):

```bash
python scripts/run_matrix.py --config configs/qwen3-8b.json --backend vulkan --machine my-gpu --server-args "--device Vulkan0" --only baseline,dspark-n7-p0 --limit 1
python scripts/run_matrix.py --config configs/qwen3-8b.json --backend vulkan --machine my-gpu --server-args "--device Vulkan0"
python scripts/summarize.py --write
python scripts/plot.py results/my-gpu/qwen3-8b__vulkan.json -o my-chart.svg
```

Use the same `--backend` name you set up (`vulkan`, `cuda`, `rocm` or `metal`): it tells the script which
`bin/` folder to use.

<details>
<summary><b>Short on time?</b> Screen first, then measure the winners</summary>

The full matrix generates about 13,000 tokens for each of its 33 settings, so it takes about 2 hours on a
60 tok/s GPU. A quick screen with 2 prompts per category and 256-token answers is enough to rule out the
clear losers in about 10 minutes:

```bash
python scripts/run_matrix.py --config configs/qwen3-8b.json --backend vulkan --machine my-gpu --server-args "--device Vulkan0" --name qwen3-8b-screen --limit 2 --osl 256 --only "baseline,ngram-mod,draft-simple-*,dflash-*,dspark-n*-p0,dspark-bf16-*"
```

Then run the best few at full size with `--only baseline,<run ids>` (no `--name`, `--limit` or `--osl`).

</details>

## 4. What Gets Measured

All methods run on the same model (Qwen3-8B, Q8_0), the same prompts and the same GPU, one request at a time:

| Method | What drafts the tokens | Run ids |
|---|---|---|
| No speculation | nothing (the reference) | `baseline` |
| n-gram | repeats patterns already in the text | `ngram-mod` |
| Small draft model | Qwen3-0.6B | `draft-simple-n3`, `-n5` |
| DFlash | a block-diffusion drafter | `dflash-n4`, `-n7` |
| **DSpark** | DFlash plus a Markov head and a confidence head | `dspark-n{2,3,4,5,7}-p{0,0.3,0.5,0.7,0.9}` |

In the DSpark ids, `n` is how many tokens are drafted per step and `p` is the confidence cutoff (`0` = off).
The prompts come from [SPEED-Bench](https://huggingface.co/datasets/nvidia/SPEED-Bench), llama.cpp's own
speculative-decoding benchmark: coding, math, writing and Q&A, greedy decoding, with Qwen3's thinking mode
off (the drafters were trained without it).

Every setting is one JSON entry in [configs/](configs/), so adding a model or a sweep needs no code.

## 5. Contribute Your GPU

The goal is a table that answers *"does DSpark help on my GPU?"* for as many cards as possible.

1. Run a config (see [Quick Start](#3-quick-start)) with a short `--machine` label such as `rtx3060-win11`.
2. Commit `results/<machine>/` and open a pull request. Mention anything unusual: power limits, a laptop, other load on the GPU.

> [!IMPORTANT]
> Only commit `results/`. The raw per-request data and server logs in `runs/` stay local; they're gitignored.

NVIDIA and Apple Silicon results are especially welcome. The first results are from AMD.

## 6. Ecosystem Status

- **llama.cpp:** DSpark merged in [#25173](https://github.com/ggml-org/llama.cpp/pull/25173). The confidence
  head is exposed only as a fixed cutoff (`--spec-draft-p-min`). Adaptive draft length exists only as open PRs for MTP.
- **Ollama:** runs DSpark drafters through `DRAFT` in a Modelfile ([ollama#17016](https://github.com/ollama/ollama/issues/17016)).
- **Drafters:** DeepSeek publishes them for Qwen3-4B/8B/14B and Gemma-4-12B, and the community has made about 100 more.

Details and sources: [docs/status-2026-10.md](docs/status-2026-10.md).

## 7. License

The code is [MIT](LICENSE). Models keep their own licenses (Qwen3: Apache-2.0; for the DSpark drafters, see
their model cards). SPEED-Bench prompts are covered by NVIDIA's evaluation dataset license: this repo never
redistributes them and publishes only aggregate timings.

This is an independent project, not affiliated with DeepSeek, ggml-org or NVIDIA. Questions and results go to
[issues](https://github.com/juraj-jumic/dspark-local-bench/issues).
