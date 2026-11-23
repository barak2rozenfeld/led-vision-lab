"""Save a grid of a detector's mistakes (false ON and false OFF) for error analysis.

Usage:
    uv run python scripts/error_gallery.py --dataset data/generated_preview --split val
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from led_vision.benchmark.runner import load_samples
from led_vision.detectors.registry import build_detectors
from led_vision.detectors.service import DetectionService

TILE = 192


def annotate(img: np.ndarray, lines: list[str], bbox: tuple[float, float, float, float] | None) -> np.ndarray:
    tile = cv2.resize(img, (TILE, TILE), interpolation=cv2.INTER_NEAREST)
    if bbox is not None:
        x, y, w, h = (int(v * TILE) for v in bbox)
        cv2.rectangle(tile, (x, y), (x + w, y + h), (0, 255, 255), 1)
    strip = np.full((14 * len(lines) + 4, TILE, 3), 25, np.uint8)
    for i, line in enumerate(lines):
        cv2.putText(strip, line, (3, 12 + 14 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.36,
                    (230, 230, 230), 1, cv2.LINE_AA)
    return np.vstack([tile, strip])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--method", default="hsv")
    parser.add_argument("--per-kind", type=int, default=12)
    parser.add_argument("--difficulty", default=None, choices=["easy", "medium", "hard", "extreme"])
    parser.add_argument("--detector-configs", type=Path, default=Path("configs/detectors"))
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts/benchmarks"))
    args = parser.parse_args()

    service = DetectionService(build_detectors(args.detector_configs))
    tiles: dict[str, list[np.ndarray]] = {"false_on": [], "false_off": []}
    for s in load_samples(args.dataset, args.split, args.size, None, 0):
        if args.difficulty and s.difficulty != args.difficulty:
            continue
        img = cv2.imread(str(s.path))
        if img is None:
            continue
        r = service.detect(img, args.method)
        kind = "false_on" if (s.state, r.state) == ("OFF", "ON") else \
               "false_off" if (s.state, r.state) == ("ON", "OFF") else None
        if kind is None or len(tiles[kind]) >= args.per_kind:
            continue
        bbox = (r.bbox.x, r.bbox.y, r.bbox.width, r.bbox.height) if r.bbox else None
        tiles[kind].append(annotate(img, [
            f"{s.background} | {s.difficulty}",
            f"truth {s.state} pred {r.state}",
            f"score {r.debug.get('score')} halo {r.debug.get('halo_excess')} thr {r.debug.get('threshold')}",
        ], bbox))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for kind, items in tiles.items():
        if not items:
            print(f"No {kind} errors")
            continue
        while len(items) % 6:
            items.append(np.zeros_like(items[0]))
        grid = np.vstack([np.hstack(items[i:i + 6]) for i in range(0, len(items), 6)])
        out = args.out_dir / f"errors_{args.method}_{kind}.jpg"
        if not cv2.imwrite(str(out), grid):
            raise OSError(f"Failed to write {out}")
        print(f"{kind}: {len(items)} tiles -> {out}")


if __name__ == "__main__":
    main()
