"""Grid-search HSV detector thresholds on the TRAIN split.

Usage:
    uv run python scripts/calibrate_hsv.py --dataset data/generated_20k --limit 2000 [--write]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from led_vision.benchmark.runner import load_samples
from led_vision.detectors.hsv import load_hsv_config
from led_vision.training.hsv_calibration import Trial, calibrate_hsv, update_yaml_values


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--limit", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--config", type=Path, default=Path("configs/detectors/hsv.yaml"))
    parser.add_argument("--write", action="store_true", help="Write the best values into --config")
    args = parser.parse_args()

    samples = load_samples(args.dataset, "train", args.size, args.limit, args.seed)
    images = [(s.state == "ON", cv2.imread(str(s.path))) for s in samples]
    images = [(on, img) for on, img in images if img is not None]
    print(f"Calibrating on {len(images)} train images ({sum(on for on, _ in images) / len(images):.0%} ON)")

    def report(t: Trial) -> None:
        print(f"  {t.params} threshold={t.on_threshold} -> {t.accuracy:.4f}")

    result = calibrate_hsv(images, load_hsv_config(args.config), progress=report)
    print(f"\nBest train accuracy {result.best.accuracy:.4f} with:")
    for k, v in result.best_params().items():
        print(f"  {k}: {v}")
    if args.write:
        update_yaml_values(args.config, result.best_params())
        print(f"Updated {args.config}")


if __name__ == "__main__":
    main()
