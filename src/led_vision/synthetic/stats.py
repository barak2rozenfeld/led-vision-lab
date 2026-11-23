"""Count what a generated dataset actually contains, from its metadata.csv."""
from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

# Per-scene columns that are tallied. Each scene has one row per export size,
# so scenes are counted once, using the smallest export size.
SCENE_FIELDS: tuple[str, ...] = (
    "state", "split", "difficulty", "background", "group", "led_color", "led_shape",
    "led_lens_type", "led_count", "ambient_type", "glare_type", "occlusion_type",
    "false_positive",
)
CROSS_TABS: tuple[tuple[str, str], ...] = (("difficulty", "state"), ("background", "state"))


def compute_dataset_stats(dataset_dir: Path) -> dict[str, Any]:
    metadata = dataset_dir / "metadata.csv"
    if not metadata.is_file():
        raise FileNotFoundError(f"metadata.csv not found in {dataset_dir}")

    config = _read_config_snapshot(dataset_dir)
    export_sizes = [int(s) for s in config.get("export_sizes", [])]
    images_by_size: Counter[str] = Counter()
    counts: dict[str, Counter[str]] = {f: Counter() for f in SCENE_FIELDS}
    cross: dict[str, Counter[str]] = {f"{a}_x_{b}": Counter() for a, b in CROSS_TABS}

    with metadata.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        missing = set(SCENE_FIELDS) - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"metadata.csv is missing columns: {sorted(missing)}")
        scene_size: str | None = str(min(export_sizes)) if export_sizes else None
        for row in reader:
            images_by_size[row["size"]] += 1
            if scene_size is None:
                scene_size = row["size"]
            if row["size"] != scene_size:
                continue
            for field_name in SCENE_FIELDS:
                counts[field_name][row[field_name]] += 1
            for a, b in CROSS_TABS:
                cross[f"{a}_x_{b}"][f"{row[a]}|{row[b]}"] += 1

    scenes = sum(counts["state"].values())
    if scenes == 0:
        raise ValueError(f"metadata.csv in {dataset_dir} has no rows")

    return {
        "dataset": dataset_dir.name,
        "scenes": scenes,
        "images": sum(images_by_size.values()),
        "images_by_size": dict(sorted(images_by_size.items(), key=lambda kv: int(kv[0]))),
        "config": {
            k: config[k]
            for k in ("seed", "total", "master_size", "export_sizes", "class_balance")
            if k in config
        },
        "counts": {k: dict(v.most_common()) for k, v in counts.items()},
        "cross_tabs": {k: dict(v.most_common()) for k, v in cross.items()},
    }


def _read_config_snapshot(dataset_dir: Path) -> dict[str, Any]:
    path = dataset_dir / "dataset_config.json"
    if not path.is_file():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} does not contain a JSON object")
    return data
