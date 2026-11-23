"""Grid-search HSV detector thresholds on the TRAIN split and print the best setting.

The detector reports its blob score regardless of the ON threshold, so each grid
point needs one pass over the images; the threshold itself is swept afterwards.

Usage:
    uv run python scripts/calibrate_hsv.py --dataset data/generated_preview --limit 1500
"""
from __future__ import annotations

import argparse
import itertools
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np

from led_vision.benchmark.runner import load_samples
from led_vision.detectors.hsv import HSVDetector, load_hsv_config

GRID = {
    "background_kernel_frac": [0.25, 0.5, 0.9],
    "excess_min": [25.0, 35.0],
    "saturation_weight": [0.25, 0.5],
    "halo_weight": [0.0, 0.5, 1.0, 2.0],
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--limit", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--config", type=Path, default=Path("configs/detectors/hsv.yaml"))
    args = parser.parse_args()

    base = load_hsv_config(args.config)
    samples = load_samples(args.dataset, "train", args.size, args.limit, args.seed)
    images = [(s.state == "ON", cv2.imread(str(s.path))) for s in samples]
    images = [(t, img) for t, img in images if img is not None]
    truth = np.array([t for t, _ in images])
    print(f"Calibrating on {len(images)} train images ({truth.mean():.0%} ON)")

    best: tuple[float, dict[str, float]] = (0.0, {})
    for values in itertools.product(*GRID.values()):
        params = dict(zip(GRID.keys(), values))
        detector = HSVDetector(replace(base, **params))  # type: ignore[arg-type]
        peaks = np.array([detector.detect(img).debug["score"] for _, img in images])
        for threshold in np.arange(30.0, 260.0, 2.5):
            acc = float(((peaks >= threshold) == truth).mean())
            if acc > best[0]:
                best = (acc, {**params, "on_threshold": float(threshold)})
        print(f"  {params} -> best so far {best[0]:.4f}")

    print(f"\nBest train accuracy {best[0]:.4f} with:")
    for k, v in best[1].items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
