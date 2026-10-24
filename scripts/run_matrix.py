"""Run a benchmark matrix: for each run, start llama-server, run SPEED-Bench, stop it.

Each run in the config is one llama-server configuration (baseline, ngram, a draft
model, DSpark with some n-max / p-min ...). Results go to:

  runs/<machine>/<backend>/<config>/<run>.json   raw speed_bench.py output (gitignored)
  results/<machine>/<config>__<backend>.json     condensed record, meant to be committed

Only the Python standard library is used here, so the script runs anywhere
speed_bench.py runs.
"""

from __future__ import annotations

import argparse
import datetime as dt
import fnmatch
import itertools
import json
import platform
import re
import shlex
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def expand_runs(config: dict) -> list[dict]:
    """Expand each run's `sweep` into concrete runs.

    {"id": "dspark-n{n}", "args": ["--spec-draft-n-max", "{n}"], "sweep": {"n": [3, 7]}}
    becomes two runs: dspark-n3 and dspark-n7.
    """
    runs = []
    for run in config["runs"]:
        sweep = run.get("sweep") or {}
        keys = list(sweep)
        for values in itertools.product(*(sweep[k] for k in keys)):
            subst = dict(zip(keys, (str(v) for v in values)))
            runs.append({
                "id": run["id"].format(**subst),
                "draft": run.get("draft"),
                "args": [a.format(**subst) for a in run.get("args", [])],
            })
    ids = [r["id"] for r in runs]
    dupes = {i for i in ids if ids.count(i) > 1}
    if dupes:
        sys.exit(f"duplicate run ids in config: {sorted(dupes)}")
    return runs


def find_server(backend: str) -> Path | None:
    exe = "llama-server.exe" if platform.system() == "Windows" else "llama-server"
    hits = sorted((ROOT / "bin" / backend).rglob(exe))
    if hits:
        return hits[0]
    on_path = shutil.which("llama-server")
    return Path(on_path) if on_path else None


def server_version(server: Path) -> str:
    try:
        out = subprocess.run([str(server), "--version"], capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.TimeoutExpired) as e:
        return f"unknown ({e})"
    text = out.stdout + out.stderr
    m = re.search(r"version:\s*(.+)", text)
    return m.group(1).strip() if m else text.strip().splitlines()[-1] if text.strip() else "unknown"


def detect_gpu() -> str:
    """Best effort; pass --gpu to override."""
    queries = [
        ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
        ["rocm-smi", "--showproductname"],
    ]
    if platform.system() == "Windows":
        queries.append(["powershell", "-NoProfile", "-Command",
                        "(Get-CimInstance Win32_VideoController).Name -join '; '"])
    for cmd in queries:
        if not shutil.which(cmd[0]):
            continue
        try:
            out = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        except (OSError, subprocess.TimeoutExpired):
            continue
        if out.returncode == 0 and out.stdout.strip():
            return " ".join(out.stdout.split())
    return "unknown"


def wait_healthy(port: int, proc: subprocess.Popen, timeout_s: float) -> bool:
    url = f"http://127.0.0.1:{port}/health"
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            return False
        try:
            with urllib.request.urlopen(url, timeout=5) as resp:
                if resp.status == 200:
                    return True
        except (urllib.error.URLError, ConnectionError, TimeoutError):
            pass
        time.sleep(2)
    return False


def warm_up(port: int) -> None:
    """One throwaway request, so the first measured sample doesn't pay for GPU warm-up."""
    body = json.dumps({"messages": [{"role": "user", "content": "Write one sentence about the sea."}],
                       "max_tokens": 32, "temperature": 0}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    try:
        urllib.request.urlopen(req, timeout=120).read()
    except (urllib.error.URLError, ConnectionError, TimeoutError) as e:
        print(f"    warm-up request failed: {e}")


def stop_server(proc: subprocess.Popen) -> None:
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()


def build_server_cmd(server: Path, config: dict, run: dict, port: int, machine_args: list[str] = ()) -> list[str]:
    cmd = [
        str(server),
        "-m", str(ROOT / config["target"]),
        "-ngl", "all", "-fa", "on",
        "-c", str(config.get("context", 8192)),
        "-np", "1",
        "--port", str(port),
        "--jinja",
    ]
    if config.get("chat_template_kwargs"):
        cmd += ["--chat-template-kwargs", json.dumps(config["chat_template_kwargs"])]
    if run["draft"]:
        cmd += ["-md", str(ROOT / run["draft"]), "-ngld", "all"]
    cmd += config.get("server_args", [])
    cmd += list(machine_args)
    cmd += run["args"]
    return cmd


def repo_relative(cmd: list[str]) -> list[str]:
    """Server args with repo paths made relative, so records from different machines compare cleanly."""
    return [Path(a).relative_to(ROOT).as_posix() if a.startswith(str(ROOT)) else a for a in cmd]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True, help="path to a configs/*.json file")
    p.add_argument("--backend", required=True, help="label for the backend, e.g. vulkan, rocm, cuda, metal")
    p.add_argument("--server", help="path to llama-server (default: search bin/<backend>/, then PATH)")
    p.add_argument("--machine", required=True,
                   help="short public label for this machine, e.g. rx7900gre-win11 (becomes the results/ folder name)")
    p.add_argument("--gpu", help="GPU description to record (default: auto-detect)")
    p.add_argument("--llama-cpp-dir", default=str(ROOT / "llama.cpp"), help="llama.cpp checkout that holds speed_bench.py")
    p.add_argument("--only", help="comma-separated run ids or globs to run, e.g. baseline,dspark-*")
    p.add_argument("--limit", type=int, help="override samples per category (use 1-2 for a smoke test)")
    p.add_argument("--server-args", default="",
                   help='extra llama-server args for this machine, e.g. "--device Vulkan0" to keep everything '
                        "on one GPU when an iGPU is also visible (see llama-server --list-devices)")
    p.add_argument("--port", type=int, default=9931)
    p.add_argument("--load-timeout", type=float, default=600, help="seconds to wait for the model to load")
    p.add_argument("--force", action="store_true", help="re-run runs that already have results")
    p.add_argument("--dry-run", action="store_true", help="print the commands and exit")
    args = p.parse_args()

    config_path = Path(args.config).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config_name = config_path.stem
    machine_args = shlex.split(args.server_args)
    runs = expand_runs(config)
    if args.only:
        patterns = [s.strip() for s in args.only.split(",") if s.strip()]
        runs = [r for r in runs if any(fnmatch.fnmatch(r["id"], pat) for pat in patterns)]
        if not runs:
            sys.exit(f"--only {args.only!r} matched no runs")

    server = Path(args.server) if args.server else find_server(args.backend)
    if server is None and not args.dry_run:
        sys.exit(f"llama-server not found in bin/{args.backend}/ or on PATH; pass --server")
    server = server or Path("llama-server")

    speed_bench = Path(args.llama_cpp_dir) / "tools" / "server" / "bench" / "speed-bench" / "speed_bench.py"
    bench = config["bench"]
    limit = args.limit if args.limit is not None else bench.get("limit")
    bench_cmd_base = [
        sys.executable, str(speed_bench),
        "--url", f"127.0.0.1:{args.port}",
        "--bench", bench.get("bench", "qualitative"),
        "--category", bench.get("categories", "all"),
        "--osl", str(bench.get("osl", 512)),
        "--extra-inputs", json.dumps(bench.get("extra_inputs", {"temperature": 0})),
        "--concurrency", "1",
    ]
    if limit:
        bench_cmd_base += ["--limit", str(limit)]

    if args.dry_run:
        for run in runs:
            print(f"# {run['id']}")
            print("  server:", subprocess.list2cmdline(build_server_cmd(server, config, run, args.port, machine_args)))
        print("  bench: ", subprocess.list2cmdline(bench_cmd_base + ["--output", "<raw.json>"]))
        return 0

    if not speed_bench.exists():
        sys.exit(f"{speed_bench} not found; clone llama.cpp (see README) or pass --llama-cpp-dir")
    missing = [f for f in [config["target"]] + [r["draft"] for r in runs if r["draft"]] if not (ROOT / f).exists()]
    if missing:
        sys.exit("missing model files (run scripts/fetch_models.py):\n  " + "\n  ".join(sorted(set(missing))))

    raw_dir = ROOT / "runs" / args.machine / args.backend / config_name
    raw_dir.mkdir(parents=True, exist_ok=True)
    out_path = ROOT / "results" / args.machine / f"{config_name}__{args.backend}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    record = json.loads(out_path.read_text(encoding="utf-8")) if out_path.exists() else {"runs": {}}
    record["meta"] = {
        "machine": args.machine,
        "backend": args.backend,
        "gpu": args.gpu or detect_gpu(),
        "llama_cpp_version": server_version(server),
        "os": platform.platform(),
        "python": platform.python_version(),
        "config": config_name,
        "target": config["target"],
        "context": config.get("context", 8192),
        "bench": {**bench, "limit": limit},
        "updated": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
    }

    print(f"{len(runs)} runs | {record['meta']['gpu']} | {args.backend} | llama.cpp {record['meta']['llama_cpp_version']}")
    for i, run in enumerate(runs, 1):
        raw_path = raw_dir / f"{run['id']}.json"
        if run["id"] in record["runs"] and raw_path.exists() and not args.force:
            print(f"[{i}/{len(runs)}] {run['id']}: already done, skipping (--force to redo)")
            continue

        server_cmd = build_server_cmd(server, config, run, args.port, machine_args)
        log_path = raw_dir / f"{run['id']}.server.log"
        print(f"[{i}/{len(runs)}] {run['id']}: starting server")
        with open(log_path, "w", encoding="utf-8", errors="replace") as log:
            proc = subprocess.Popen(server_cmd, stdout=log, stderr=subprocess.STDOUT)
            try:
                if not wait_healthy(args.port, proc, args.load_timeout):
                    print(f"    server failed to start, see {log_path}")
                    record["runs"][run["id"]] = {"server_args": repo_relative(server_cmd[1:]),
                                                 "error": "server failed to start"}
                    out_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
                    continue
                warm_up(args.port)
                t0 = time.monotonic()
                bench_out = subprocess.run(bench_cmd_base + ["--output", str(raw_path)])
                elapsed = time.monotonic() - t0
            finally:
                stop_server(proc)

        if bench_out.returncode != 0 or not raw_path.exists():
            print(f"    speed_bench.py failed (exit {bench_out.returncode})")
            record["runs"][run["id"]] = {"server_args": repo_relative(server_cmd[1:]),
                                         "error": f"speed_bench exit {bench_out.returncode}"}
        else:
            raw = json.loads(raw_path.read_text(encoding="utf-8"))
            record["runs"][run["id"]] = {
                "server_args": repo_relative(server_cmd[1:]),
                "completed_samples": raw["completed_samples"],
                "failed_samples": raw["failed_samples"],
                "bench_seconds": round(elapsed, 1),
                "summary": raw["summary"],
            }
        out_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")

    print(f"\nwrote {out_path.relative_to(ROOT)}\nnext: python scripts/summarize.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
