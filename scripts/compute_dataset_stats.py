"""Write real dataset statistics (served by GET /api/v1/dataset/stats).

Usage:
    uv run python scripts/compute_dataset_stats.py --dataset data/generated_pcb_1m
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from led_vision.provenance import provenance
from led_vision.synthetic.stats import compute_dataset_stats

REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("artifacts/dataset/stats.json"))
    args = parser.parse_args()

    stats = compute_dataset_stats(args.dataset)
    stats["provenance"] = provenance(REPO_ROOT)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(f"{stats['scenes']:,} scenes / {stats['images']:,} images -> {args.out}")


if __name__ == "__main__":
    main()
