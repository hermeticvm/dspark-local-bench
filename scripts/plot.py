"""Draw the README chart: decode speedup of the best setting per method, as an SVG.

    python scripts/plot.py results/<machine>/<config>__<backend>.json -o assets/speedup.svg

The SVG carries its own light and dark colors (prefers-color-scheme), so it reads on
both GitHub themes. Standard library only.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from html import escape
from pathlib import Path

# (run-id pattern, display name, detail template); first match wins
FAMILIES = [
    (r"^baseline$", "No speculation", "plain decoding"),
    (r"^ngram-mod$", "n-gram", "no extra model"),
    (r"^draft-simple-n(?P<n>\d+)$", "Small draft model", "Qwen3-0.6B · drafts {n} tokens"),
    (r"^dflash-n(?P<n>\d+)$", "DFlash", "drafts {n} tokens"),
    (r"^dspark-bf16-n(?P<n>\d+)-p(?P<p>[\d.]+)$", "DSpark, BF16 drafter", "drafts {n} tokens{cutoff}"),
    # the setting llama.cpp's docs use, shown on its own so the tuned bar has a reference
    (r"^dspark-n(?P<n>7)-p(?P<p>0)$", "DSpark, docs example", "drafts {n} tokens"),
    (r"^dspark-n(?P<n>\d+)-p(?P<p>[\d.]+)$", "DSpark", "drafts {n} tokens{cutoff}"),
]

BACKEND_NAMES = {"vulkan": "Vulkan", "rocm": "ROCm", "cuda": "CUDA", "metal": "Metal", "cpu": "CPU"}

STYLE = """
  .surface { fill: none; }
  .ink { fill: #0b0b0b; } .ink2 { fill: #52514e; } .muted { fill: #6f6d68; }
  .grid { stroke: #e1e0d9; } .axis { stroke: #c3c2b7; }
  .bar { fill: #918f88; } .bar-hi { fill: #2a78d6; }
  @media (prefers-color-scheme: dark) {
    .ink { fill: #f0efec; } .ink2 { fill: #c3c2b7; } .muted { fill: #9a9890; }
    .grid { stroke: #2c2c2a; } .axis { stroke: #383835; }
    .bar { fill: #6b6a64; } .bar-hi { fill: #3987e5; }
  }
  text { font-family: system-ui, -apple-system, "Segoe UI", sans-serif; }
  .num { font-variant-numeric: tabular-nums; }
"""


def overall(run: dict) -> dict | None:
    return next((r for r in run.get("summary", []) if r["category"] == "overall"), None)


def best_per_family(runs: dict) -> list[dict]:
    best: dict[str, dict] = {}
    for run_id, run in runs.items():
        o = overall(run)
        if not o or not o.get("avg_pred_t_s"):
            continue
        for pattern, name, detail in FAMILIES:
            m = re.match(pattern, run_id)
            if not m:
                continue
            g = m.groupdict()
            p = float(g["p"]) if g.get("p") else 0.0
            row = {"name": name, "run": run_id, "t_s": o["avg_pred_t_s"], "accept": o.get("accept_rate"),
                   "detail": detail.format(n=g.get("n"), cutoff=f" · cutoff {p:g}" if p > 0 else "")}
            if name not in best or row["t_s"] > best[name]["t_s"]:
                best[name] = row
            break
    # the docs setting only gets its own bar when tuning beat it; otherwise it is the DSpark bar
    docs = best.pop("DSpark, docs example", None)
    if docs and ("DSpark" not in best or docs["t_s"] >= best["DSpark"]["t_s"]):
        best["DSpark"] = {**docs, "name": "DSpark"}
    elif docs:
        best[docs["name"]] = docs
    return sorted(best.values(), key=lambda r: r["t_s"], reverse=True)


def bar_path(x0: float, y: float, w: float, h: float, r: float = 4) -> str:
    """Bar from the baseline at x0: square at the baseline, 4px rounded at the data end."""
    r = min(r, w / 2, h / 2)
    x1 = x0 + w
    return (f"M{x0:.1f},{y:.1f} H{x1 - r:.1f} Q{x1:.1f},{y:.1f} {x1:.1f},{y + r:.1f} "
            f"V{y + h - r:.1f} Q{x1:.1f},{y + h:.1f} {x1 - r:.1f},{y + h:.1f} H{x0:.1f} Z")


def render(record: dict, rows: list[dict]) -> str:
    base = next((r for r in rows if r["name"] == "No speculation"), None)
    if base is None:
        sys.exit("results have no baseline run; speedups need one")
    for r in rows:
        r["speedup"] = r["t_s"] / base["t_s"]
    fastest = max(rows, key=lambda r: r["speedup"])

    meta = record["meta"]
    build = re.search(r"build (\d+)", meta.get("llama_cpp_version", ""))
    model = Path(meta["target"]).stem
    backend = BACKEND_NAMES.get(meta["backend"], meta["backend"])
    subtitle = (f"{model} · {meta['gpu']} · {backend} · "
                f"llama.cpp {'b' + build.group(1) if build else meta.get('llama_cpp_version', '')} · one user, greedy")

    width, left, right = 760, 250, 130
    top, row_h, bar_h = 92, 48, 22
    plot_w = width - left - right
    max_x = max(2.0, round(fastest["speedup"] * 2 + 0.49) / 2)  # axis end on a clean 0.5 step
    sx = lambda v: left + v / max_x * plot_w
    plot_bottom = top + row_h * len(rows) - (row_h - bar_h) / 2 + 8
    height = round(plot_bottom + 62)
    desc = "; ".join(f"{r['name']} ({r['detail']}): {r['speedup']:.2f}x, {r['t_s']:.0f} tokens/s" for r in rows)

    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
           f'viewBox="0 0 {width} {height}" role="img" aria-labelledby="t d">',
           f"<style>{STYLE}</style>",
           f'<title id="t">Decode speed vs. no speculation</title>',
           f'<desc id="d">{escape(desc)}</desc>',
           f'<text x="0" y="28" class="ink" font-size="20" font-weight="600">Decode speed vs. no speculation</text>',
           f'<text x="0" y="52" class="ink2" font-size="13">{escape(subtitle)}</text>']

    tick = 0.0
    while tick <= max_x + 1e-9:
        x = sx(tick)
        out.append(f'<line x1="{x:.1f}" y1="{top - 8}" x2="{x:.1f}" y2="{plot_bottom:.1f}" class="grid" stroke-width="1"/>')
        out.append(f'<text x="{x:.1f}" y="{plot_bottom + 18:.1f}" class="muted num" font-size="12" '
                   f'text-anchor="middle">{tick:g}×</text>')
        tick += 0.5
    out.append(f'<line x1="{left}" y1="{top - 8}" x2="{left}" y2="{plot_bottom:.1f}" class="axis" stroke-width="1"/>')

    for i, r in enumerate(rows):
        y = top + i * row_h
        out.append(f'<text x="{left - 16}" y="{y + 10}" class="ink" font-size="14" font-weight="600" '
                   f'text-anchor="end">{escape(r["name"])}</text>')
        out.append(f'<text x="{left - 16}" y="{y + 27}" class="muted" font-size="12" '
                   f'text-anchor="end">{escape(r["detail"])}</text>')
        w = sx(r["speedup"]) - left
        cls = "bar-hi" if r is fastest else "bar"
        out.append(f'<path d="{bar_path(left, y, w, bar_h)}" class="{cls}"/>')
        out.append(f'<text x="{left + w + 8:.1f}" y="{y + 16}" class="num" font-size="14">'
                   f'<tspan class="ink" font-weight="600">{r["speedup"]:.2f}×</tspan>'
                   f'<tspan class="muted" font-size="12" dx="6">{r["t_s"]:.0f} tok/s</tspan></text>')

    out.append(f'<text x="0" y="{height - 8}" class="muted" font-size="12">Best setting per method · '
               f'higher is better · accept rates and per-category numbers in results/</text>')
    out.append("</svg>")
    return "\n".join(out) + "\n"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("results", help="a results/<machine>/<config>__<backend>.json file")
    p.add_argument("-o", "--output", default="assets/speedup.svg")
    args = p.parse_args()

    record = json.loads(Path(args.results).read_text(encoding="utf-8"))
    rows = best_per_family(record["runs"])
    svg = render(record, rows)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(svg, encoding="utf-8")
    for r in rows:
        print(f"{r['name']:22} {r['run']:22} {r['speedup']:.2f}x  {r['t_s']:.1f} tok/s")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
