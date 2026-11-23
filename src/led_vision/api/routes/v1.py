"""HTTP routes. Validation and error mapping only: detection lives in DetectionService."""
from __future__ import annotations

import json
import logging
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Annotated, Any

import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile

from led_vision.detectors.base import DetectorNotReadyError
from led_vision.detectors.service import DetectionService, UnknownMethodError
from led_vision.preprocessing.image_io import ImageTooLargeError, InvalidImageError, decode_image

from ..config import ApiConfig
from ..schemas import CompareOut, DetectionOut, HealthOut, MethodOut

log = logging.getLogger("led_vision.api")
router = APIRouter(prefix="/api/v1")

try:
    API_VERSION = version("led-vision-lab")
except PackageNotFoundError:
    API_VERSION = "unknown"


def _service(request: Request) -> DetectionService:
    return request.app.state.service  # type: ignore[no-any-return]


def _config(request: Request) -> ApiConfig:
    return request.app.state.config  # type: ignore[no-any-return]


def _read_image(upload: UploadFile, cfg: ApiConfig) -> np.ndarray:
    if upload.content_type not in cfg.accepted_content_types:
        raise HTTPException(
            415, f"Unsupported content type {upload.content_type!r}; "
                 f"accepted: {', '.join(cfg.accepted_content_types)}",
        )
    data = upload.file.read(cfg.max_upload_bytes + 1)
    if len(data) > cfg.max_upload_bytes:
        raise HTTPException(413, f"Upload exceeds {cfg.max_upload_bytes:,} bytes")
    try:
        return decode_image(data, cfg.max_pixels)
    except ImageTooLargeError as exc:
        raise HTTPException(413, str(exc)) from exc
    except InvalidImageError as exc:
        raise HTTPException(400, str(exc)) from exc


def _read_json_artifact(path: Path, what: str, how: str) -> dict[str, Any]:
    if not path.is_file():
        raise HTTPException(404, f"No {what} yet. Generate it with: {how}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        log.error("Corrupt %s file %s: %s", what, path, exc)
        raise HTTPException(500, f"The {what} file is corrupt") from exc
    if not isinstance(data, dict):
        raise HTTPException(500, f"The {what} file is not a JSON object")
    return data


@router.get("/health", response_model=HealthOut)
def health(request: Request) -> HealthOut:
    return HealthOut(status="ok", version=API_VERSION, ready_methods=_service(request).ready_methods())


@router.get("/methods", response_model=list[MethodOut])
def methods(request: Request) -> list[MethodOut]:
    return [MethodOut.from_state(m) for m in _service(request).methods()]


@router.post("/detect", response_model=DetectionOut)
def detect(
    request: Request,
    image: Annotated[UploadFile, File(description="JPG, PNG, BMP or WebP image")],
    method: Annotated[str, Form(description="Method id, e.g. 'hsv'")] = "hsv",
) -> DetectionOut:
    img = _read_image(image, _config(request))
    try:
        result = _service(request).detect(img, method)
    except UnknownMethodError as exc:
        raise HTTPException(404, f"Unknown method {method!r}") from exc
    except DetectorNotReadyError as exc:
        raise HTTPException(503, str(exc)) from exc
    log.info("detect method=%s state=%s conf=%.3f runtime_ms=%.2f",
             result.method, result.state, result.confidence, result.runtime_ms)
    return DetectionOut.from_result(result)


@router.post("/compare", response_model=CompareOut)
def compare(
    request: Request,
    image: Annotated[UploadFile, File(description="JPG, PNG, BMP or WebP image")],
) -> CompareOut:
    img = _read_image(image, _config(request))
    service = _service(request)
    results = service.compare(img)
    skipped = {m.info.id: m.status for m in service.methods() if m.status != "ready"}
    log.info("compare ran=%s skipped=%s", [r.method for r in results], sorted(skipped))
    return CompareOut(results=[DetectionOut.from_result(r) for r in results], skipped=skipped)


@router.get("/dataset/stats")
def dataset_stats(request: Request) -> dict[str, Any]:
    return _read_json_artifact(
        _config(request).dataset_stats_path, "dataset statistics",
        "uv run python scripts/compute_dataset_stats.py --dataset <dir>",
    )


@router.get("/benchmark")
def benchmark(request: Request) -> dict[str, Any]:
    return _read_json_artifact(
        _config(request).benchmark_path, "benchmark report",
        "uv run python scripts/run_benchmark.py --dataset <dir> --split test",
    )
