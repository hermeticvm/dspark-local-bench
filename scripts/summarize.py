"""Turn results/**/*.json into markdown tables (and optionally CSV).

Decode speedup is relative to the `baseline` run in the same results file, so it
always compares the same machine, backend and llama.cpp build.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def overall(run: dict) -> dict | None:
    for row in run.get("summary", []):
        if row["category"] == "overall":
            return row
    return None


def fmt(value, kind: str) -> str:
    if value is None:
        return "–"
    if kind == "speed":
        return f"{value:.1f}"
    if kind == "ratio":
        return f"{value:.2f}×"
    if kind == "rate":
        return f"{value:.1%}"
    return str(value)


def table_for(record: dict) -> tuple[str, list[dict]]:
    runs = record["runs"]
    base = overall(runs["baseline"]) if "baseline" in runs else None
    base_by_cat = {r["category"]: r for r in runs.get("baseline", {}).get("summary", [])}
    categories = [r["category"] for r in runs.get("baseline", {}).get("summary", []) if r["category"] != "overall"]

    rows = []
    for run_id, run in runs.items():
        o = overall(run)
        if o is None:
            rows.append({"run": run_id, "error": run.get("error", "no summary")})
            continue
        row = {
            "run": run_id,
            "decode_t_s": o["avg_pred_t_s"],
            "speedup": (o["avg_pred_t_s"] / base["avg_pred_t_s"])
            if base and o["avg_pred_t_s"] and base["avg_pred_t_s"] else None,
            "prompt_t_s": o["avg_prompt_t_s"],
            "prompt_ratio": (o["avg_prompt_t_s"] / base["avg_prompt_t_s"])
            if base and o["avg_prompt_t_s"] and base["avg_prompt_t_s"] else None,
            "accept_rate": o["accept_rate"],
            "latency_s": o["avg_latency"],
            "failed": o["failed"],
        }
        for cat_row in run["summary"]:
            cat = cat_row["category"]
            b = base_by_cat.get(cat)
            if cat != "overall" and b and b["avg_pred_t_s"] and cat_row["avg_pred_t_s"]:
                row[f"speedup_{cat}"] = cat_row["avg_pred_t_s"] / b["avg_pred_t_s"]
        rows.append(row)
    rows.sort(key=lambda r: r.get("decode_t_s") or 0, reverse=True)

    meta = record["meta"]
    lines = [
        f"### {meta['config']} · {meta['backend']} · {meta['gpu']}",
        "",
        f"llama.cpp `{meta['llama_cpp_version']}` · {meta['os']} · target `{meta['target']}` · "
        f"ctx {meta['context']} · bench `{meta['bench'].get('bench')}` "
        f"({meta['bench'].get('categories')}, limit {meta['bench'].get('limit')}, osl {meta['bench'].get('osl')}) · "
        f"updated {meta['updated']}",
        "",
    ]
    header = ["run", "decode t/s", "speedup", "prompt t/s", "prompt vs base", "accept"] + \
             [f"{c} ×" for c in categories] + ["failed"]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "---|" * len(header))
    for r in rows:
        if "error" in r:
            lines.append(f"| {r['run']} | error: {r['error']} |" + " |" * (len(header) - 2))
            continue
        cells = [
            r["run"], fmt(r["decode_t_s"], "speed"), fmt(r["speedup"], "ratio"),
            fmt(r["prompt_t_s"], "speed"), fmt(r["prompt_ratio"], "ratio"), fmt(r["accept_rate"], "rate"),
        ] + [fmt(r.get(f"speedup_{c}"), "ratio") for c in categories] + [str(r["failed"])]
        lines.append("| " + " | ".join(cells) + " |")
    if base is None:
        lines += ["", "_No `baseline` run in this file, so speedups are not computed._"]
    return "\n".join(lines) + "\n", rows


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("paths", nargs="*", help="result files (default: results/**/*.json)")
    p.add_argument("--csv", help="also write all rows to this CSV file")
    p.add_argument("--write", action="store_true", help="write a .md next to each results file")
    args = p.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")  # tables use × and ·; Windows consoles may default to cp1252

    # lossless__*.json are check_lossless.py output, not speed tables
    paths = [Path(x) for x in args.paths] or sorted(
        p for p in (ROOT / "results").rglob("*.json") if not p.name.startswith("lossless__"))
    if not paths:
        print("no results yet: run scripts/run_matrix.py first", file=sys.stderr)
        return 1

    all_rows = []
    for path in paths:
        record = json.loads(path.read_text(encoding="utf-8"))
        md, rows = table_for(record)
        print(md)
        if args.write:
            path.with_suffix(".md").write_text(md, encoding="utf-8")
        meta = record["meta"]
        for r in rows:
            all_rows.append({"machine": meta["machine"], "gpu": meta["gpu"], "backend": meta["backend"],
                             "config": meta["config"], "llama_cpp": meta["llama_cpp_version"], **r})

    if args.csv:
        fields = list(dict.fromkeys(k for r in all_rows for k in r))
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(all_rows)
        print(f"wrote {args.csv}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
