from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from led_vision.api.app import create_app
from led_vision.api.config import ApiConfig, load_api_config

from ..unit.test_detectors import DETECTOR_CONFIGS, REPO, dark_led, lit_led


def png(image: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", image)
    assert ok
    return buf.tobytes()


@pytest.fixture()
def artifacts(tmp_path: Path) -> tuple[Path, Path]:
    return tmp_path / "stats.json", tmp_path / "summary.json"


@pytest.fixture()
def client(artifacts: tuple[Path, Path]) -> TestClient:
    stats, bench = artifacts
    cfg = ApiConfig(
        max_upload_bytes=200_000, max_pixels=1_000_000,
        detector_config_dir=DETECTOR_CONFIGS, dataset_stats_path=stats, benchmark_path=bench,
    )
    return TestClient(create_app(cfg))


def post(client: TestClient, path: str, data: bytes, content_type: str = "image/png",
         **form: str) -> object:
    return client.post(path, files={"image": ("led.png", data, content_type)}, data=form)


def test_health(client: TestClient) -> None:
    body = client.get("/api/v1/health").json()
    assert body["status"] == "ok"
    assert body["ready_methods"] == ["hsv"]


def test_methods_report_status(client: TestClient) -> None:
    methods = {m["id"]: m["status"] for m in client.get("/api/v1/methods").json()}
    assert methods == {"hsv": "ready", "template": "planned", "knn": "planned",
                       "cnn": "planned", "yolo": "planned"}


@pytest.mark.parametrize("image, state", [(lit_led(), "ON"), (dark_led(), "OFF")])
def test_detect(client: TestClient, image: np.ndarray, state: str) -> None:
    res = post(client, "/api/v1/detect", png(image), method="hsv")
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["state"] == state and body["method"] == "hsv"
    assert 0.0 <= body["confidence"] <= 1.0
    assert (body["bbox"] is not None) == (state == "ON")


def test_detect_defaults_to_hsv(client: TestClient) -> None:
    assert post(client, "/api/v1/detect", png(lit_led())).json()["method"] == "hsv"


def test_compare(client: TestClient) -> None:
    body = post(client, "/api/v1/compare", png(lit_led())).json()
    assert [r["method"] for r in body["results"]] == ["hsv"]
    assert body["skipped"]["yolo"] == "planned"


@pytest.mark.parametrize("data, content_type, form, status", [
    (b"GIF89a", "image/gif", {}, 415),
    (b"x" * 200_001, "image/png", {}, 413),
    (b"not really a png", "image/png", {}, 400),
    (b"", "image/png", {}, 400),
    (None, "image/png", {"method": "magic"}, 404),
    (None, "image/png", {"method": "yolo"}, 503),
], ids=["wrong_type", "too_big", "corrupt", "empty", "unknown_method", "not_trained"])
def test_detect_errors(client: TestClient, data: bytes | None, content_type: str,
                       form: dict[str, str], status: int) -> None:
    res = post(client, "/api/v1/detect", data if data is not None else png(lit_led()),
               content_type, **form)
    assert res.status_code == status, res.text
    assert isinstance(res.json()["detail"], str)


def test_too_many_pixels_is_413(client: TestClient) -> None:
    big = np.zeros((1200, 1000, 3), np.uint8)
    assert post(client, "/api/v1/detect", png(big)).status_code == 413


def test_missing_image_field_is_422(client: TestClient) -> None:
    assert client.post("/api/v1/detect", data={"method": "hsv"}).status_code == 422


def test_artifacts_404_until_generated(client: TestClient) -> None:
    for path in ("/api/v1/dataset/stats", "/api/v1/benchmark"):
        res = client.get(path)
        assert res.status_code == 404
        assert "scripts/" in res.json()["detail"]


def test_artifacts_served(client: TestClient, artifacts: tuple[Path, Path]) -> None:
    stats, bench = artifacts
    stats.write_text(json.dumps({"scenes": 4}), encoding="utf-8")
    bench.write_text(json.dumps({"methods": {}}), encoding="utf-8")
    assert client.get("/api/v1/dataset/stats").json() == {"scenes": 4}
    assert client.get("/api/v1/benchmark").json() == {"methods": {}}


def test_corrupt_artifact_is_500(client: TestClient, artifacts: tuple[Path, Path]) -> None:
    artifacts[0].write_text("{broken", encoding="utf-8")
    assert client.get("/api/v1/dataset/stats").status_code == 500


def test_repo_api_config_loads() -> None:
    cfg = load_api_config(REPO / "configs" / "api.yaml")
    assert cfg.max_upload_bytes == 10 * 1024 * 1024
    assert "image/jpeg" in cfg.accepted_content_types


def test_api_config_rejects_unknown_keys(tmp_path: Path) -> None:
    path = tmp_path / "api.yaml"
    path.write_text("port: 1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="port"):
        load_api_config(path)
