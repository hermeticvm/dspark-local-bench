# Results

One folder per machine; one file per config and backend:

```
results/<machine>/<config>__<backend>.json   written by scripts/run_matrix.py
results/<machine>/<config>__<backend>.md     written by scripts/summarize.py --write
```

Each `.json` has a `meta` block (GPU, backend, llama.cpp version, OS, bench settings) and a `runs` map from
run id to the server args used and the SPEED-Bench per-category summary. It holds no prompts or model output.

Speedups are always computed against the `baseline` run in the same file: same machine, backend and build.

`lossless__<backend>__<run>.json` files come from `scripts/check_lossless.py`. They record how many answers
were identical with and without speculation, and where differing answers first diverge. They hold counts
and positions only, no text.

Run `python scripts/validate_results.py` before committing. CI runs it on every pull request.
