"""Central catalogue of detection methods and the detectors that implement them."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from .base import LedDetector
from .hsv import HSVDetector, load_hsv_config

MethodStatus = Literal["ready", "not_trained", "planned"]


@dataclass(frozen=True)
class MethodInfo:
    id: str
    name: str
    family: str
    requires_training: bool


METHOD_CATALOG: tuple[MethodInfo, ...] = (
    MethodInfo("hsv", "HSV Threshold", "Classical CV", requires_training=False),
    MethodInfo("template", "Template Diff", "Classical CV", requires_training=False),
    MethodInfo("knn", "KNN Classifier", "Traditional ML", requires_training=True),
    MethodInfo("cnn", "Small CNN", "Deep Learning", requires_training=True),
    MethodInfo("yolo", "YOLOv8-nano", "Deep Learning (fine-tuned)", requires_training=True),
)


def build_detectors(detector_config_dir: Path) -> dict[str, LedDetector]:
    """Instantiate every implemented detector. Methods not listed here are 'planned'."""
    return {
        "hsv": HSVDetector(load_hsv_config(detector_config_dir / "hsv.yaml")),
    }
