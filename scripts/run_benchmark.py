"""Benchmark every ready detector on a dataset split (served by GET /api/v1/benchmark).

Usage:
    uv run python scripts/run_benchmark.py --dataset data/generated_pcb_1m --split test --size 128
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from led_vision.benchmark.runner import load_samples, run_benchmark
from led_vision.detectors.registry import build_detectors
from led_vision.detectors.service import DetectionService
from led_vision.provenance import provenance

REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--limit", type=int, default=None, help="Random subset size")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--detector-configs", type=Path, default=Path("configs/detectors"))
    parser.add_argument("--out", type=Path, default=Path("artifacts/benchmarks/summary.json"))
    args = parser.parse_args()

    service = DetectionService(build_detectors(args.detector_configs))
    samples = load_samples(args.dataset, args.split, args.size, args.limit, args.seed)
    report = run_benchmark(service, samples)
    report.update({
        "dataset": args.dataset.name,
        "split": args.split,
        "image_size": args.size,
        "subset_seed": args.seed if args.limit else None,
        "provenance": provenance(REPO_ROOT),
    })
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Evaluated {report['evaluated_images']:,} images ({report['missing_images']} missing)")
    for method, r in report["methods"].items():
        o, lat = r["overall"], r["latency"]
        print(f"  {method:9s} acc={o['accuracy']:.3f} f1={o['f1']:.3f} "
              f"falseON={o['false_on']} falseOFF={o['false_off']} p50={lat['p50_ms']:.2f}ms")
    print(f"Report -> {args.out}")


if __name__ == "__main__":
    main()
