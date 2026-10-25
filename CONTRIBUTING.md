# Contributing

The most useful contribution is **results from a GPU that isn't in the tables yet**. Fixes and new configs
are welcome too.

## Adding your results

1. Follow the [Quick Start](README.md#3-quick-start) and run a config. Use a short `--machine` label such as
   `rtx3060-win11` or `m3pro-macos`. It becomes your folder name, so keep it free of personal details.
2. Optionally, check that speculation doesn't change the output on your machine:
   ```bash
   python scripts/check_lossless.py --config configs/qwen3-8b.json --backend <backend> --machine <label> --run <fastest run id>
   ```
3. Run `python scripts/summarize.py --write` and `python scripts/validate_results.py`. The second one must
   report 0 problems.
4. Open a pull request that adds **only** `results/<machine>/`. In the description, mention anything that
   affects speed: a power limit, a laptop on battery, other programs using the GPU, driver version.

Not comfortable with pull requests? Use the [results issue form](../../issues/new?template=results.yml).

## Good measurement habits

- Close anything else that uses the GPU first: Ollama or LM Studio with a model loaded, games, video.
- Plug laptops in and use the high-performance power mode.
- Don't edit result files by hand. Rerun with `--force` instead.

## Changing code or configs

- Keep the scripts on the Python standard library where possible, and Python 3.10 compatible.
- New config? Give it a `description` and `downloads`, and prefer the official `ggml-org` or model-author
  Hugging Face repos. Configs are reviewed like code, since they decide what gets downloaded and run.
- Test on your machine, and say in the PR which OS and backend you tested on.
