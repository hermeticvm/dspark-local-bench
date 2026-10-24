"""Download the GGUF files a config needs into models/.

    pip install huggingface_hub
    python scripts/fetch_models.py configs/qwen3-8b.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("configs", nargs="+", help="configs/*.json files")
    args = p.parse_args()

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        sys.exit("pip install huggingface_hub")

    wanted = {}
    for c in args.configs:
        for d in json.loads(Path(c).read_text(encoding="utf-8")).get("downloads", []):
            wanted[(d["repo"], d["file"])] = d
    for (repo, file), d in wanted.items():
        dest = ROOT / "models"
        if (dest / file).exists():
            print(f"have  {file}")
            continue
        print(f"fetch {repo} / {file}")
        hf_hub_download(repo_id=repo, filename=file, local_dir=dest)
    return 0


if __name__ == "__main__":
    sys.exit(main())
