from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from led_vision.detectors.base import (
    DetectionResult, DetectorNotReadyError, LedDetector, Prediction,
)
from led_vision.detectors.hsv import HSVDetector, HsvConfig, load_hsv_config
from led_vision.detectors.registry import METHOD_CATALOG, build_detectors
from led_vision.detectors.service import DetectionService, UnknownMethodError
from led_vision.preprocessing.image_io import ImageTooLargeError, InvalidImageError, decode_image

REPO = Path(__file__).resolve().parents[2]
DETECTOR_CONFIGS = REPO / "configs" / "detectors"


def lit_led(size: int = 128, color: tuple[int, int, int] = (40, 40, 255)) -> np.ndarray:
    """Dark board with a saturated LED core and a soft glow around it."""
    img = np.full((size, size, 3), 30, np.uint8)
    yy, xx = np.mgrid[:size, :size]
    d2 = (xx - size // 2) ** 2 + (yy - size // 2) ** 2
    glow = np.exp(-d2 / (2 * (size * 0.12) ** 2))[..., None]
    img = np.clip(img + glow * np.array(color, np.float64) * 0.8, 0, 255).astype(np.uint8)
    cv2.circle(img, (size // 2, size // 2), size // 14, (255, 255, 255), -1)
    return img


def dark_led(size: int = 128) -> np.ndarray:
    img = np.full((size, size, 3), 30, np.uint8)
    cv2.circle(img, (size // 2, size // 2), size // 10, (45, 45, 70), -1)
    return img


class _Fixed(LedDetector):
    name = "fixed"

    def __init__(self, ready: bool = True) -> None:
        self._ready = ready

    @property
    def is_ready(self) -> bool:
        return self._ready

    def _predict(self, image: np.ndarray) -> Prediction:
        return Prediction(state="ON", confidence=1.7)


# --- base interface ---------------------------------------------------------

def test_base_times_and_clips_confidence() -> None:
    r = _Fixed().detect(np.zeros((16, 16, 3), np.uint8))
    assert isinstance(r, DetectionResult)
    assert r.method == "fixed" and r.state == "ON"
    assert r.confidence == 1.0
    assert r.runtime_ms >= 0


@pytest.mark.parametrize("image", [
    np.zeros((16, 16), np.uint8),          # grayscale
    np.zeros((16, 16, 3), np.float32),     # wrong dtype
    np.zeros((4, 4, 3), np.uint8),         # too small
])
def test_base_rejects_bad_images(image: np.ndarray) -> None:
    with pytest.raises(ValueError):
        _Fixed().detect(image)


def test_base_refuses_when_not_ready() -> None:
    with pytest.raises(DetectorNotReadyError):
        _Fixed(ready=False).detect(np.zeros((16, 16, 3), np.uint8))


# --- HSV detector -----------------------------------------------------------

@pytest.fixture(scope="module")
def hsv() -> HSVDetector:
    return HSVDetector(load_hsv_config(DETECTOR_CONFIGS / "hsv.yaml"))


@pytest.mark.parametrize("color", [(40, 40, 255), (40, 255, 40), (255, 255, 255)])
def test_hsv_detects_lit_led(hsv: HSVDetector, color: tuple[int, int, int]) -> None:
    r = hsv.detect(lit_led(color=color))
    assert r.state == "ON"
    assert r.bbox is not None
    cx, cy = r.bbox.x + r.bbox.width / 2, r.bbox.y + r.bbox.height / 2
    assert abs(cx - 0.5) < 0.1 and abs(cy - 0.5) < 0.1


@pytest.mark.parametrize("image", [dark_led(), np.full((128, 128, 3), 200, np.uint8)])
def test_hsv_says_off_without_light(hsv: HSVDetector, image: np.ndarray) -> None:
    r = hsv.detect(image)
    assert r.state == "OFF"
    assert r.bbox is None


def test_hsv_handles_any_input_size(hsv: HSVDetector) -> None:
    assert hsv.detect(cv2.resize(lit_led(), (640, 360))).state == "ON"


def test_hsv_debug_explains_decision(hsv: HSVDetector) -> None:
    debug = hsv.detect(lit_led()).debug
    assert {"score", "peak_score", "halo_excess", "threshold"} <= set(debug)
    assert debug["score"] >= debug["threshold"]


def test_hsv_config_rejects_unknown_keys(tmp_path: Path) -> None:
    path = tmp_path / "hsv.yaml"
    path.write_text("v_min: 80\nnot_a_key: 1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="not_a_key"):
        load_hsv_config(path)


def test_hsv_config_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_hsv_config(tmp_path / "nope.yaml")


def test_hsv_defaults_are_valid() -> None:
    assert HSVDetector(HsvConfig()).detect(dark_led()).state == "OFF"


# --- registry and service ---------------------------------------------------

def test_service_lists_every_catalogued_method() -> None:
    service = DetectionService(build_detectors(DETECTOR_CONFIGS))
    states = {m.info.id: m.status for m in service.methods()}
    assert set(states) == {m.id for m in METHOD_CATALOG}
    assert states["hsv"] == "ready"
    assert service.ready_methods() == ["hsv"]


def test_service_errors() -> None:
    service = DetectionService(build_detectors(DETECTOR_CONFIGS))
    with pytest.raises(UnknownMethodError):
        service.detect(lit_led(), "magic")
    with pytest.raises(DetectorNotReadyError):
        service.detect(lit_led(), "yolo")


def test_service_rejects_uncatalogued_detector() -> None:
    with pytest.raises(ValueError):
        DetectionService({"fixed": _Fixed()})


def test_service_compare_runs_ready_methods_only() -> None:
    service = DetectionService(build_detectors(DETECTOR_CONFIGS))
    assert [r.method for r in service.compare(lit_led())] == ["hsv"]


# --- image decoding ---------------------------------------------------------

def test_decode_roundtrip() -> None:
    ok, buf = cv2.imencode(".png", lit_led())
    assert ok
    assert decode_image(buf.tobytes(), max_pixels=10**6).shape == (128, 128, 3)


def test_decode_errors() -> None:
    with pytest.raises(InvalidImageError):
        decode_image(b"", max_pixels=10**6)
    with pytest.raises(InvalidImageError):
        decode_image(b"definitely not an image", max_pixels=10**6)
    _, buf = cv2.imencode(".png", lit_led())
    with pytest.raises(ImageTooLargeError):
        decode_image(buf.tobytes(), max_pixels=100)
