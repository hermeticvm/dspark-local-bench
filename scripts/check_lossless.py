"""Check that speculative decoding doesn't change the output.

With greedy decoding, speculation should only change speed: every drafted token is
verified by the target model. This answers the same SPEED-Bench prompts three times:

  1. no speculation
  2. no speculation again (control: plain decoding must match itself, or nothing below means anything)
  3. with the speculative run under test

and compares the generated texts. Only counts and first-difference positions go to
results/<machine>/lossless__<backend>.json; the texts stay in runs/ (gitignored).

    python scripts/check_lossless.py --config configs/qwen3-8b.json --backend vulkan --machine my-gpu \
        --run dspark-n3-p0.3 --server-args "--device Vulkan0"
"""

from __future__ import annotations

import argparse
import datetime as dt
import importlib.util
import json
import platform
import shlex
import sys
from pathlib import Path

import run_matrix as rm

ROOT = rm.ROOT


def load_prompts(llama_cpp_dir: Path, bench: dict, limit: int) -> list[tuple[str, str, str]]:
    """(sample id, category, first user turn) for the configured SPEED-Bench split."""
    path = llama_cpp_dir / "tools" / "server" / "bench" / "speed-bench" / "speed_bench.py"
    spec = importlib.util.spec_from_file_location("speed_bench", path)
    sb = importlib.util.module_from_spec(spec)
    sys.modules["speed_bench"] = sb  # its dataclasses look the module up while being defined
    spec.loader.exec_module(sb)
    samples = sb.load_samples(argparse.Namespace(bench=bench.get("bench", "qualitative"),
                                                 category=bench.get("categories", "all"), limit=limit))
    return [(s.id, s.category, s.turns[0]) for s in samples]


def generate(server: Path, config: dict, run: dict, args, env: dict, prompts, log_path: Path) -> list[str]:
    cmd = rm.build_server_cmd(server, config, run, args.port, shlex.split(args.server_args))
    with open(log_path, "w", encoding="utf-8", errors="replace") as log:
        proc = rm.start_server(cmd, env, args.port, log)
        try:
            if not rm.wait_healthy(args.port, proc, args.load_timeout):
                sys.exit(f"server for {run['id']} failed to start, see {log_path}")
            rm.warm_up(args.port)
            texts = []
            for i, (_, category, prompt) in enumerate(prompts, 1):
                print(f"\r    {run['id']}: {i}/{len(prompts)} ({category})   ", end="", flush=True)
                texts.append(rm.chat(args.port, prompt, args.max_tokens))
            print()
            return texts
        finally:
            rm.stop_server(proc)


def first_diff(a: str, b: str) -> int | None:
    """Index of the first differing character, or None if identical."""
    if a == b:
        return None
    n = min(len(a), len(b))
    return next((i for i in range(n) if a[i] != b[i]), n)


def compare(ref: list[str], other: list[str], prompts) -> dict:
    diffs = []
    for (sample_id, category, _), a, b in zip(prompts, ref, other):
        i = first_diff(a, b)
        if i is not None:
            diffs.append({"id": sample_id, "category": category, "first_diff_char": i,
                          "first_diff_fraction": round(i / max(len(a), 1), 3), "len_ref": len(a), "len": len(b)})
    return {"identical": len(prompts) - len(diffs), "total": len(prompts), "differences": diffs}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--backend", required=True)
    p.add_argument("--machine", required=True)
    p.add_argument("--run", required=True, help="speculative run id to check, e.g. dspark-n3-p0.3")
    p.add_argument("--server")
    p.add_argument("--server-args", default="")
    p.add_argument("--llama-cpp-dir", default=str(ROOT / "llama.cpp"))
    p.add_argument("--limit", type=int, default=4, help="prompts per category (default 4)")
    p.add_argument("--max-tokens", type=int, default=256)
    p.add_argument("--port", type=int, default=9931)
    p.add_argument("--load-timeout", type=float, default=600)
    args = p.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    config_path = Path(args.config).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    runs = {r["id"]: r for r in rm.expand_runs(config)}
    if args.run not in runs:
        sys.exit(f"unknown run {args.run!r}; see the ids in {args.config}")
    server = Path(args.server) if args.server else rm.find_server(args.backend)
    if server is None:
        sys.exit(f"llama-server not found in bin/{args.backend}/ or on PATH; pass --server")
    env = rm.server_env(args.backend)

    prompts = load_prompts(Path(args.llama_cpp_dir), config["bench"], args.limit)
    raw_dir = ROOT / "runs" / args.machine / args.backend / "lossless"
    raw_dir.mkdir(parents=True, exist_ok=True)
    print(f"{len(prompts)} prompts, {args.max_tokens} tokens each: baseline, baseline again, {args.run}")

    ref = generate(server, config, runs["baseline"], args, env, prompts, raw_dir / "baseline.server.log")
    again = generate(server, config, runs["baseline"], args, env, prompts, raw_dir / "baseline-again.server.log")
    spec = generate(server, config, runs[args.run], args, env, prompts, raw_dir / f"{args.run}.server.log")
    (raw_dir / f"texts__{args.run}.json").write_text(json.dumps(
        {"prompt_ids": [x[0] for x in prompts], "baseline": ref, "baseline_again": again, args.run: spec},
        indent=2, ensure_ascii=False), encoding="utf-8")

    result = {
        "meta": {"machine": args.machine, "backend": args.backend, "llama_cpp_version": rm.server_version(server, env),
                 "config": config_path.stem, "target": config["target"], "run": args.run,
                 "max_tokens": args.max_tokens, "limit": args.limit, "os": platform.platform(),
                 "updated": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")},
        "control": compare(ref, again, prompts),
        "speculative": compare(ref, spec, prompts),
    }
    out = ROOT / "results" / args.machine / f"lossless__{args.backend}__{args.run}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    c, s = result["control"], result["speculative"]
    print(f"control (baseline vs baseline): {c['identical']}/{c['total']} identical")
    print(f"{args.run} vs baseline:  {s['identical']}/{s['total']} identical")
    for d in s["differences"]:
        print(f"    {d['category']:8} {d['id']}: first difference at {d['first_diff_fraction']:.0%} of the text")
    print(f"wrote {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
