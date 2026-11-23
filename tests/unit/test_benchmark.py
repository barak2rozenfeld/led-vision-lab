from __future__ import annotations

import csv
import json
from pathlib import Path

import cv2
import pytest

from led_vision.benchmark.metrics import binary_metrics, latency_summary
from led_vision.benchmark.runner import load_samples, run_benchmark
from led_vision.detectors.registry import build_detectors
from led_vision.detectors.service import DetectionService
from led_vision.provenance import git_commit, git_dirty, provenance
from led_vision.synthetic.stats import compute_dataset_stats

from .test_detectors import DETECTOR_CONFIGS, dark_led, lit_led

FIELDS = [
    "filepath", "split", "size", "state", "difficulty", "background", "group", "led_color",
    "led_shape", "led_lens_type", "led_count", "ambient_type", "glare_type", "occlusion_type",
    "false_positive",
]


def test_binary_metrics() -> None:
    m = binary_metrics(["ON", "ON", "OFF", "OFF"], ["ON", "OFF", "ON", "OFF"])
    assert m.accuracy == 0.5
    assert (m.false_on, m.false_off) == (1, 1)
    assert m.confusion == {"tp": 1, "tn": 1, "fp": 1, "fn": 1}
    assert m.precision == m.recall == m.f1 == 0.5


def test_binary_metrics_no_positive_predictions() -> None:
    m = binary_metrics(["ON", "OFF"], ["OFF", "OFF"])
    assert m.precision == 0.0 and m.f1 == 0.0


def test_binary_metrics_errors() -> None:
    with pytest.raises(ValueError):
        binary_metrics(["ON"], [])
    with pytest.raises(ValueError):
        binary_metrics([], [])


def test_latency_summary() -> None:
    s = latency_summary([1.0, 2.0, 3.0, 4.0])
    assert s["mean_ms"] == 2.5 and s["max_ms"] == 4.0
    with pytest.raises(ValueError):
        latency_summary([])


@pytest.fixture()
def tiny_dataset(tmp_path: Path) -> Path:
    """4 scenes x 2 export sizes, in the generator's on-disk layout."""
    rows = []
    for i, (state, difficulty) in enumerate([("on", "easy"), ("off", "easy"),
                                             ("on", "hard"), ("off", "hard")]):
        for size in (64, 128):
            rel = Path(f"images/{size}/test/{state}/{i:04d}.png")
            (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
            img = lit_led(size) if state == "on" else dark_led(size)
            assert cv2.imwrite(str(tmp_path / rel), img)
            rows.append({
                "filepath": rel.as_posix(), "split": "test", "size": size, "state": state,
                "difficulty": difficulty, "background": "pcb", "group": "clean_easy",
                "led_color": "red", "led_shape": "round", "led_lens_type": "clear",
                "led_count": 1, "ambient_type": "dark_room", "glare_type": "none",
                "occlusion_type": "none", "false_positive": i == 1,
            })
    with (tmp_path / "metadata.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    (tmp_path / "dataset_config.json").write_text(
        json.dumps({"seed": 7, "total": 4, "export_sizes": [64, 128]}), encoding="utf-8")
    return tmp_path


def test_dataset_stats_counts_scenes_once(tiny_dataset: Path) -> None:
    stats = compute_dataset_stats(tiny_dataset)
    assert stats["scenes"] == 4
    assert stats["images"] == 8
    assert stats["images_by_size"] == {"64": 4, "128": 4}
    assert stats["counts"]["state"] == {"on": 2, "off": 2}
    assert stats["cross_tabs"]["difficulty_x_state"]["easy|on"] == 1
    assert stats["config"]["seed"] == 7


def test_dataset_stats_missing_metadata(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        compute_dataset_stats(tmp_path)


def test_load_samples_filters_and_limits(tiny_dataset: Path) -> None:
    assert len(load_samples(tiny_dataset, "test", 128, None, 0)) == 4
    assert len(load_samples(tiny_dataset, "test", 128, 2, 0)) == 2
    with pytest.raises(ValueError):
        load_samples(tiny_dataset, "train", 128, None, 0)


def test_run_benchmark(tiny_dataset: Path) -> None:
    service = DetectionService(build_detectors(DETECTOR_CONFIGS))
    report = run_benchmark(service, load_samples(tiny_dataset, "test", 128, None, 0))
    hsv = report["methods"]["hsv"]
    assert report["evaluated_images"] == 4 and report["missing_images"] == 0
    assert hsv["overall"]["accuracy"] == 1.0
    assert set(hsv["by_difficulty"]) == {"easy", "hard"}
    assert hsv["distractor_scenes"] == {"count": 1, "predicted_on": 0}


def test_run_benchmark_counts_missing_images(tiny_dataset: Path) -> None:
    samples = load_samples(tiny_dataset, "test", 128, None, 0)
    samples[0].path.unlink()
    service = DetectionService(build_detectors(DETECTOR_CONFIGS))
    assert run_benchmark(service, samples)["missing_images"] == 1


def test_provenance_outside_a_repository(tmp_path: Path) -> None:
    assert git_commit(tmp_path) is None
    assert git_dirty(tmp_path) is None
    p = provenance(tmp_path)
    assert {"generated_at", "git_commit", "git_dirty", "machine", "python"} <= set(p)
