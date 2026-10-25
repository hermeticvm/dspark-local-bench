# Security

## Reporting a problem

Please report security problems **privately** through
[GitHub's private vulnerability reporting](https://github.com/juraj-jumic/dspark-local-bench/security/advisories/new),
not as a public issue. You'll get an answer within a week.

Security problems in llama.cpp itself go to [llama.cpp's security policy](https://github.com/ggml-org/llama.cpp/security).

## What this project runs on your machine

So you can judge the risk before running anything:

- **Downloads:** the setup scripts download llama.cpp's official release for a pinned tag from
  `github.com/ggml-org/llama.cpp` and **verify each file against the SHA-256 checksum GitHub publishes**
  before unpacking it. Models come from the Hugging Face repos named in `configs/*.json`. Python packages come
  from PyPI at the versions pinned in `requirements.txt`; with `-Backend rocm`, AMD's ROCm libraries also come
  from AMD's package index (`stable.repo.amd.com`).
- **Network:** `llama-server` listens only on `127.0.0.1` (your own machine) while a benchmark runs, and is
  stopped afterwards. Nothing is uploaded, and the scripts send no telemetry.
- **Files:** everything stays inside the project folder. Raw logs (`runs/`) and downloads (`bin/`, `models/`,
  `llama.cpp/`, `.venv/`) are gitignored, so they can't be committed by accident.

## For maintainers reviewing pull requests

- **Treat `configs/` and `scripts/` changes as code.** A config chooses which model repos are downloaded
  and which flags `llama-server` gets, so check every repo name and flag.
- **Result files are data.** CI runs `scripts/validate_results.py` on them, and the scripts escape their text
  before putting it into charts and tables.
