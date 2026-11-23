"""Common detector interface shared by every detection method."""
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

LedState = Literal["ON", "OFF"]


@dataclass(frozen=True)
class BoundingBox:
    """Box normalised to the image: all values in 0..1."""

    x: float
    y: float
    width: float
    height: float


@dataclass(frozen=True)
class Prediction:
    """What a detector computes; timing and method name are added by the base class."""

    state: LedState
    confidence: float
    bbox: BoundingBox | None = None
    debug: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DetectionResult:
    state: LedState
    confidence: float
    runtime_ms: float
    method: str
    bbox: BoundingBox | None = None
    debug: dict[str, Any] = field(default_factory=dict)


class DetectorNotReadyError(RuntimeError):
    """Raised when a detector needs a trained model that is not available."""


class LedDetector(ABC):
    """Base class for all detectors. Subclasses implement :meth:`_predict`."""

    name: str

    @property
    def is_ready(self) -> bool:
        return True

    def detect(self, image: np.ndarray) -> DetectionResult:
        """Run detection on a BGR uint8 image and time it."""
        _validate_image(image)
        if not self.is_ready:
            raise DetectorNotReadyError(f"Detector {self.name!r} is not ready")
        start = time.perf_counter()
        prediction = self._predict(image)
        runtime_ms = (time.perf_counter() - start) * 1000.0
        return DetectionResult(
            state=prediction.state,
            confidence=float(np.clip(prediction.confidence, 0.0, 1.0)),
            runtime_ms=round(runtime_ms, 3),
            method=self.name,
            bbox=prediction.bbox,
            debug=prediction.debug,
        )

    @abstractmethod
    def _predict(self, image: np.ndarray) -> Prediction: ...


def _validate_image(image: np.ndarray) -> None:
    if not isinstance(image, np.ndarray):
        raise TypeError(f"Expected numpy array, got {type(image).__name__}")
    if image.dtype != np.uint8:
        raise ValueError(f"Expected uint8 image, got {image.dtype}")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected HxWx3 BGR image, got shape {image.shape}")
    if image.shape[0] < 8 or image.shape[1] < 8:
        raise ValueError(f"Image too small: {image.shape[1]}x{image.shape[0]}")
