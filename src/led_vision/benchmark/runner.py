"""Run every ready detector over one split of a generated dataset."""
from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ..detectors.service import DetectionService
from .metrics import binary_metrics, latency_summary

BACKGROUND_FAMILY: dict[str, str] = {
    "pcb": "pcb", "blue_pcb": "pcb", "realistic_pcb": "pcb", "aluminum_pcb": "pcb",
    "black_plastic": "plastic", "gray_plastic": "plastic", "white_plastic": "plastic",
    "dark_enclosure": "enclosure", "light_enclosure": "enclosure",
    "metal": "metal", "brushed_metal": "metal",
    "white_studio": "studio", "textured": "studio", "gradient": "studio", "plain": "studio",
}


@dataclass(frozen=True)
class Sample:
    path: Path
    state: str  # "ON" | "OFF"
    difficulty: str
    background: str
    false_positive: bool


def load_samples(
    dataset_dir: Path, split: str, size: int, limit: int | None, seed: int
) -> list[Sample]:
    metadata = dataset_dir / "metadata.csv"
    if not metadata.is_file():
        raise FileNotFoundError(f"metadata.csv not found in {dataset_dir}")
    samples: list[Sample] = []
    with metadata.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["split"] != split or int(row["size"]) != size:
                continue
            samples.append(Sample(
                path=dataset_dir / row["filepath"],
                state=row["state"].upper(),
                difficulty=row["difficulty"],
                background=row["background"],
                false_positive=row["false_positive"] == "True",
            ))
    if not samples:
        raise ValueError(f"No samples for split={split!r} size={size} in {dataset_dir}")
    if limit is not None and limit < len(samples):
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(samples), size=limit, replace=False)
        samples = [samples[i] for i in sorted(idx)]
    return samples


def run_benchmark(
    service: DetectionService, samples: list[Sample]
) -> dict[str, Any]:
    images: list[tuple[Sample, np.ndarray]] = []
    missing = 0
    for s in samples:
        img = cv2.imread(str(s.path))
        if img is None:
            missing += 1
            continue
        images.append((s, img))
    if not images:
        raise ValueError("None of the selected images could be read")

    results: dict[str, Any] = {}
    for method in service.ready_methods():
        truth: list[str] = []
        pred: list[str] = []
        runtimes: list[float] = []
        groups: dict[str, dict[str, tuple[list[str], list[str]]]] = {
            "difficulty": defaultdict(lambda: ([], [])),
            "background_family": defaultdict(lambda: ([], [])),
        }
        fp_scenes = [0, 0]  # [false-positive distractor scenes, of which predicted ON]
        for sample, img in images:
            r = service.detect(img, method)
            truth.append(sample.state)
            pred.append(r.state)
            runtimes.append(r.runtime_ms)
            for key, value in (("difficulty", sample.difficulty),
                               ("background_family", BACKGROUND_FAMILY.get(sample.background, "other"))):
                groups[key][value][0].append(sample.state)
                groups[key][value][1].append(r.state)
            if sample.false_positive:
                fp_scenes[0] += 1
                fp_scenes[1] += r.state == "ON"

        results[method] = {
            "overall": binary_metrics(truth, pred).to_dict(),
            "latency": latency_summary(runtimes),
            "by_difficulty": {k: binary_metrics(*v).to_dict() for k, v in sorted(groups["difficulty"].items())},
            "by_background_family": {
                k: binary_metrics(*v).to_dict() for k, v in sorted(groups["background_family"].items())
            },
            "distractor_scenes": {"count": fp_scenes[0], "predicted_on": fp_scenes[1]},
        }
    return {"evaluated_images": len(images), "missing_images": missing, "methods": results}
