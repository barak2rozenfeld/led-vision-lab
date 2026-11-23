"""Pick a small, diverse set of sample images from a generated dataset.

Copies one ON and one OFF image per background style into ``data/samples/images``
and writes a labelled showcase grid to ``docs/images/sample_grid.jpg``.

Usage:
    uv run python scripts/make_samples.py --dataset data/generated_pcb_1m --size 256
"""
from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

import cv2
import numpy as np

STATES: tuple[str, ...] = ("on", "off")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("data/generated_pcb_1m"))
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--samples-dir", type=Path, default=Path("data/samples/images"))
    parser.add_argument("--grid-path", type=Path, default=Path("docs/images/sample_grid.jpg"))
    return parser


def select_rows(
    metadata: Path, dataset: Path, size: int
) -> dict[tuple[str, str], dict[str, str]]:
    """Return the first single-LED, non-false-positive row per (background, state)
    whose image exists on disk."""
    picked: dict[tuple[str, str], dict[str, str]] = {}
    with metadata.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if int(row["size"]) != size or row["led_count"] != "1":
                continue
            if row["false_positive"] == "True" or row["difficulty"] not in ("easy", "medium"):
                continue
            key = (row["background"], row["state"])
            if key not in picked and (dataset / row["filepath"]).is_file():
                picked[key] = row
    return picked


def copy_samples(
    picked: dict[tuple[str, str], dict[str, str]], dataset: Path, samples_dir: Path
) -> list[tuple[str, str, Path]]:
    samples_dir.mkdir(parents=True, exist_ok=True)
    copied: list[tuple[str, str, Path]] = []
    for (background, state), row in sorted(picked.items()):
        src = dataset / row["filepath"]
        if not src.is_file():
            raise FileNotFoundError(f"Image listed in metadata is missing: {src}")
        dst = samples_dir / f"{background}_{state}_{row['led_color']}.jpg"
        shutil.copyfile(src, dst)
        copied.append((background, state, dst))
    return copied


def build_grid(copied: list[tuple[str, str, Path]], grid_path: Path, tile: int) -> None:
    """Rows = backgrounds, columns = ON / OFF, with a label strip on each tile."""
    by_bg: dict[str, dict[str, Path]] = {}
    for background, state, path in copied:
        by_bg.setdefault(background, {})[state] = path

    label_h = 24
    blank = np.full((tile + label_h, tile, 3), 30, dtype=np.uint8)
    rows: list[np.ndarray] = []
    for background in sorted(by_bg):
        cells: list[np.ndarray] = []
        for state in STATES:
            cell = blank.copy()
            path = by_bg[background].get(state)
            if path is not None:
                img = cv2.imread(str(path))
                if img is None:
                    raise ValueError(f"Could not read sample image: {path}")
                cell[label_h:] = cv2.resize(img, (tile, tile))
            color = (80, 220, 80) if state == "on" else (90, 90, 230)
            cv2.putText(cell, f"{background} | {state.upper()}", (6, 17),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
            cells.append(cell)
        rows.append(np.hstack(cells))

    # Two background-rows side by side keeps the grid closer to square.
    if len(rows) % 2:
        rows.append(np.full_like(rows[0], 30))
    pairs = [np.hstack([rows[i], rows[i + 1]]) for i in range(0, len(rows), 2)]
    grid = np.vstack(pairs)

    grid_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(grid_path), grid, [cv2.IMWRITE_JPEG_QUALITY, 90]):
        raise OSError(f"Failed to write grid image: {grid_path}")


def main() -> None:
    args = build_parser().parse_args()
    metadata = args.dataset / "metadata.csv"
    if not metadata.is_file():
        raise FileNotFoundError(f"metadata.csv not found in {args.dataset}")

    picked = select_rows(metadata, args.dataset, args.size)
    copied = copy_samples(picked, args.dataset, args.samples_dir)
    build_grid(copied, args.grid_path, tile=args.size)
    print(f"Copied {len(copied)} samples to {args.samples_dir}")
    print(f"Grid written to {args.grid_path}")


if __name__ == "__main__":
    main()
