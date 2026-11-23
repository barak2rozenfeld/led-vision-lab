"""API configuration loaded from configs/api.yaml."""
from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path

import yaml

DEFAULT_API_CONFIG = Path("configs/api.yaml")


@dataclass(frozen=True)
class ApiConfig:
    max_upload_bytes: int = 10 * 1024 * 1024
    max_pixels: int = 16_000_000
    accepted_content_types: tuple[str, ...] = ("image/jpeg", "image/png", "image/bmp", "image/webp")
    detector_config_dir: Path = Path("configs/detectors")
    dataset_stats_path: Path = Path("artifacts/dataset/stats.json")
    benchmark_path: Path = Path("artifacts/benchmarks/summary.json")
    cors_origins: tuple[str, ...] = field(default_factory=tuple)


_PATH_FIELDS = {"detector_config_dir", "dataset_stats_path", "benchmark_path"}
_TUPLE_FIELDS = {"accepted_content_types", "cors_origins"}


def load_api_config(path: Path) -> ApiConfig:
    if not path.is_file():
        raise FileNotFoundError(f"API config not found: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"{path} must contain a mapping")
    known = {f.name for f in fields(ApiConfig)}
    unknown = set(raw) - known
    if unknown:
        raise ValueError(f"Unknown keys in {path}: {sorted(unknown)}")
    values: dict[str, object] = {}
    for key, value in raw.items():
        if key in _PATH_FIELDS:
            values[key] = Path(value)
        elif key in _TUPLE_FIELDS:
            values[key] = tuple(value)
        else:
            values[key] = value
    cfg = ApiConfig(**values)  # type: ignore[arg-type]
    if cfg.max_upload_bytes <= 0 or cfg.max_pixels <= 0:
        raise ValueError("max_upload_bytes and max_pixels must be positive")
    return cfg
