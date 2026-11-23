"""Response models. They mirror the detector dataclasses one-to-one."""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel

from led_vision.detectors.base import DetectionResult
from led_vision.detectors.service import MethodState


class BoundingBoxOut(BaseModel):
    x: float
    y: float
    width: float
    height: float


class DetectionOut(BaseModel):
    state: Literal["ON", "OFF"]
    confidence: float
    runtime_ms: float
    method: str
    bbox: BoundingBoxOut | None = None
    debug: dict[str, Any] = {}

    @classmethod
    def from_result(cls, r: DetectionResult) -> DetectionOut:
        bbox = None
        if r.bbox is not None:
            bbox = BoundingBoxOut(x=r.bbox.x, y=r.bbox.y, width=r.bbox.width, height=r.bbox.height)
        return cls(
            state=r.state, confidence=r.confidence, runtime_ms=r.runtime_ms,
            method=r.method, bbox=bbox, debug=dict(r.debug),
        )


class CompareOut(BaseModel):
    results: list[DetectionOut]
    skipped: dict[str, str]  # method id -> reason it was not run


class MethodOut(BaseModel):
    id: str
    name: str
    family: str
    requires_training: bool
    status: Literal["ready", "not_trained", "planned"]

    @classmethod
    def from_state(cls, m: MethodState) -> MethodOut:
        return cls(
            id=m.info.id, name=m.info.name, family=m.info.family,
            requires_training=m.info.requires_training, status=m.status,
        )


class HealthOut(BaseModel):
    status: Literal["ok"]
    version: str
    ready_methods: list[str]
