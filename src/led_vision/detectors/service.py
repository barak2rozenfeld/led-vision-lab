"""Detection service: picks detectors by name and runs them. Holds no CV logic."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from .base import DetectionResult, DetectorNotReadyError, LedDetector
from .registry import METHOD_CATALOG, MethodInfo, MethodStatus


class UnknownMethodError(KeyError):
    pass


@dataclass(frozen=True)
class MethodState:
    info: MethodInfo
    status: MethodStatus


class DetectionService:
    def __init__(self, detectors: Mapping[str, LedDetector]) -> None:
        known = {m.id for m in METHOD_CATALOG}
        unknown = set(detectors) - known
        if unknown:
            raise ValueError(f"Detectors not in METHOD_CATALOG: {sorted(unknown)}")
        self._detectors = dict(detectors)

    def methods(self) -> list[MethodState]:
        return [MethodState(info, self._status(info.id)) for info in METHOD_CATALOG]

    def ready_methods(self) -> list[str]:
        return [m.info.id for m in self.methods() if m.status == "ready"]

    def detect(self, image: np.ndarray, method: str) -> DetectionResult:
        if method not in {m.id for m in METHOD_CATALOG}:
            raise UnknownMethodError(method)
        status = self._status(method)
        if status != "ready":
            raise DetectorNotReadyError(f"Method {method!r} is {status.replace('_', ' ')}")
        return self._detectors[method].detect(image)

    def compare(self, image: np.ndarray) -> list[DetectionResult]:
        return [self._detectors[m].detect(image) for m in self.ready_methods()]

    def _status(self, method: str) -> MethodStatus:
        detector = self._detectors.get(method)
        if detector is None:
            return "planned"
        return "ready" if detector.is_ready else "not_trained"
