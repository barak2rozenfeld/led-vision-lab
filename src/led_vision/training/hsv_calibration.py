"""Grid-search calibration of the HSV detector. Run on the train split only."""
from __future__ import annotations

import itertools
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from led_vision.detectors.hsv import HSVDetector, HsvConfig

DEFAULT_GRID: dict[str, list[float]] = {
    "background_kernel_frac": [0.25, 0.5, 0.9],
    "excess_min": [25.0, 35.0],
    "saturation_weight": [0.25, 0.5],
    "halo_weight": [0.0, 0.5, 1.0, 2.0],
}
DEFAULT_THRESHOLDS: np.ndarray = np.arange(30.0, 260.0, 2.5)


@dataclass(frozen=True)
class Trial:
    params: dict[str, float]
    on_threshold: float
    accuracy: float


@dataclass(frozen=True)
class CalibrationResult:
    best: Trial
    trials: list[Trial]

    def best_params(self) -> dict[str, float]:
        return {**self.best.params, "on_threshold": self.best.on_threshold}


def calibrate_hsv(
    images: Sequence[tuple[bool, np.ndarray]],
    base: HsvConfig,
    grid: Mapping[str, Sequence[float]] = DEFAULT_GRID,
    thresholds: np.ndarray = DEFAULT_THRESHOLDS,
    progress: Callable[[Trial], None] | None = None,
) -> CalibrationResult:
    """`images` are (is_on, BGR image) pairs.

    The detector reports its score regardless of the ON threshold, so each grid
    point needs one pass over the images and the threshold is swept afterwards.
    """
    if not images:
        raise ValueError("No images to calibrate on")
    unknown = set(grid) - set(HsvConfig.__dataclass_fields__)
    if unknown:
        raise ValueError(f"Grid keys are not HsvConfig fields: {sorted(unknown)}")

    truth = np.array([is_on for is_on, _ in images])
    trials: list[Trial] = []
    for values in itertools.product(*grid.values()):
        params = {k: float(v) for k, v in zip(grid.keys(), values)}
        detector = HSVDetector(replace(base, **params))  # type: ignore[arg-type]
        scores = np.array([detector.detect(img).debug["score"] for _, img in images])
        accuracies = [float(((scores >= t) == truth).mean()) for t in thresholds]
        i = int(np.argmax(accuracies))
        trial = Trial(params=params, on_threshold=float(thresholds[i]), accuracy=accuracies[i])
        trials.append(trial)
        if progress is not None:
            progress(trial)
    return CalibrationResult(best=max(trials, key=lambda t: t.accuracy), trials=trials)


def update_yaml_values(path: Path, values: Mapping[str, float]) -> None:
    """Rewrite `key: value` lines in place, keeping comments and layout."""
    text = path.read_text(encoding="utf-8")
    for key, value in values.items():
        pattern = re.compile(rf"^({re.escape(key)}:\s*)([^\s#]+)", re.MULTILINE)
        text, n = pattern.subn(lambda m: f"{m.group(1)}{value}", text)
        if n != 1:
            raise ValueError(f"Expected exactly one '{key}:' line in {path}, found {n}")
    path.write_text(text, encoding="utf-8")
