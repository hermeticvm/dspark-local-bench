"""Check results/ files before they're committed or merged.

    python scripts/validate_results.py            # every file under results/
    python scripts/validate_results.py FILE...    # specific files

Catches broken or hand-edited files, and personal details that shouldn't be published
(absolute paths, user names in paths, e-mail addresses). Exits non-zero on any problem.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

LABEL = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")
PERSONAL = [
    (re.compile(r"[A-Za-z]:[\\/]"), "absolute Windows path"),
    (re.compile(r"(^|[\s\"'])/(home|Users|root)/"), "absolute home-directory path"),
    (re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"), "e-mail address"),
]
SPEED_KEYS = {"category", "requests", "failed", "avg_prompt_t_s", "avg_pred_t_s", "avg_latency", "accept_rate"}


def check_text(path: Path, text: str) -> list[str]:
    return [f"{path}: contains an {what}" for pattern, what in PERSONAL if pattern.search(text)]


def check_speed(path: Path, data: dict) -> list[str]:
    errs = []
    meta, runs = data.get("meta"), data.get("runs")
    if not isinstance(meta, dict) or not isinstance(runs, dict) or not runs:
        return [f"{path}: needs a 'meta' object and a non-empty 'runs' object"]
    for key in ("machine", "backend", "gpu", "llama_cpp_version", "config", "target", "bench"):
        if key not in meta:
            errs.append(f"{path}: meta.{key} is missing")
    if meta.get("machine") != path.parent.name:
        errs.append(f"{path}: meta.machine {meta.get('machine')!r} doesn't match its folder {path.parent.name!r}")
    if not path.name.endswith(f"__{meta.get('backend')}.json"):
        errs.append(f"{path}: file name should end with __{meta.get('backend')}.json")
    if "baseline" not in runs:
        errs.append(f"{path}: no 'baseline' run, so speedups can't be computed")
    for run_id, run in runs.items():
        if "error" in run:
            continue
        rows = run.get("summary")
        if not isinstance(rows, list) or not any(r.get("category") == "overall" for r in rows):
            errs.append(f"{path}: run {run_id!r} has no summary with an 'overall' row")
            continue
        for row in rows:
            if not SPEED_KEYS <= set(row):
                errs.append(f"{path}: run {run_id!r} row {row.get('category')!r} lacks {sorted(SPEED_KEYS - set(row))}")
            for key in ("avg_pred_t_s", "avg_prompt_t_s"):
                v = row.get(key)
                if v is not None and not (isinstance(v, (int, float)) and 0 < v < 100_000):
                    errs.append(f"{path}: run {run_id!r} {key}={v!r} is not a plausible speed")
        args = run.get("server_args", [])
        if args[:1] != ["-m"] or not str(args[1:2] and args[1]).startswith("models/"):
            errs.append(f"{path}: run {run_id!r} server_args should start with '-m models/...'")
    return errs


def check_lossless(path: Path, data: dict) -> list[str]:
    errs = []
    for key in ("meta", "control", "speculative"):
        if key not in data:
            errs.append(f"{path}: '{key}' is missing")
    for key in ("control", "speculative"):
        part = data.get(key, {})
        if not isinstance(part.get("identical"), int) or not isinstance(part.get("total"), int):
            errs.append(f"{path}: {key} needs integer 'identical' and 'total'")
    return errs


def check_file(path: Path) -> list[str]:
    if not LABEL.match(path.parent.name):
        return [f"{path}: machine folder should be a short lowercase label like rtx3060-win11"]
    if path.suffix == ".md":
        return check_text(path, path.read_text(encoding="utf-8"))
    if path.suffix != ".json":
        return [] if path.name == "README.md" else [f"{path}: unexpected file type in results/"]
    text = path.read_text(encoding="utf-8")
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        return [f"{path}: invalid JSON ({e})"]
    errs = check_text(path, text)
    errs += check_lossless(path, data) if path.name.startswith("lossless__") else check_speed(path, data)
    return errs


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    paths = [Path(p) for p in sys.argv[1:]] or [
        p for p in sorted((ROOT / "results").rglob("*")) if p.is_file() and p.parent != ROOT / "results"]
    errs = [e for p in paths for e in check_file(p)]
    for e in errs:
        print(e)
    print(f"{len(paths)} files checked, {len(errs)} problems")
    return 1 if errs else 0


if __name__ == "__main__":
    sys.exit(main())
