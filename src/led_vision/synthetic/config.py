"""Dataset generation configuration — loaded from YAML or used with defaults."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class LedConfig:
    colors: list[str] = field(default_factory=lambda: [
        "red", "green", "yellow", "blue", "white", "orange",
        "purple", "cyan", "magenta", "amber", "warm_white",
        "cool_white", "lime_green", "uv_violet",
    ])
    shapes: list[str] = field(default_factory=lambda: ["round", "rectangular", "square"])
    size_range: tuple[float, float] = (0.08, 0.35)
    position_margin: float = 0.08


@dataclass
class BackgroundWeights:
    plain: float = 0.25
    gradient: float = 0.15
    pcb: float = 0.30
    metal: float = 0.20
    noisy: float = 0.10


@dataclass
class VariationConfig:
    blur_range: tuple[float, float] = (0.0, 3.0)
    noise_sigma_range: tuple[float, float] = (0.0, 15.0)
    camera_angle_range: tuple[float, float] = (-10.0, 10.0)
    ambient_factor_range: tuple[float, float] = (0.5, 1.5)
    brightness_range: tuple[float, float] = (0.7, 1.3)


@dataclass
class SplitRatios:
    train: float = 0.70
    val: float = 0.15
    test: float = 0.15

    def __post_init__(self) -> None:
        total = self.train + self.val + self.test
        if abs(total - 1.0) > 1e-4:
            raise ValueError(f"Split ratios must sum to 1.0, got {total:.4f}")


@dataclass
class DatasetConfig:
    seed: int = 42
    total: int = 100
    output_dir: Path = field(default_factory=lambda: Path("data/generated"))
    sizes: list[int] = field(default_factory=lambda: [128])
    image_format: str = "jpg"
    image_quality: int = 95
    split: SplitRatios = field(default_factory=SplitRatios)
    on_ratio: float = 0.50
    led: LedConfig = field(default_factory=LedConfig)
    backgrounds: BackgroundWeights = field(default_factory=BackgroundWeights)
    variation: VariationConfig = field(default_factory=VariationConfig)


def load_config(path: Path) -> DatasetConfig:
    """Load a DatasetConfig from a YAML file."""
    with path.open(encoding="utf-8") as f:
        data: dict[str, Any] = yaml.safe_load(f)

    cfg = DatasetConfig()

    if "seed" in data:
        cfg.seed = int(data["seed"])
    if "total" in data:
        cfg.total = int(data["total"])
    if "output_dir" in data:
        cfg.output_dir = Path(data["output_dir"])
    if "image_format" in data:
        cfg.image_format = str(data["image_format"])
    if "image_quality" in data:
        cfg.image_quality = int(data["image_quality"])
    if "on_ratio" in data:
        cfg.on_ratio = float(data["on_ratio"])
    if "sizes" in data:
        cfg.sizes = [int(s) for s in data["sizes"]]

    if "split" in data:
        s = data["split"]
        cfg.split = SplitRatios(
            train=float(s.get("train", 0.70)),
            val=float(s.get("val", 0.15)),
            test=float(s.get("test", 0.15)),
        )

    if "led" in data:
        led = data["led"]
        size_range = led.get("size_range", list(cfg.led.size_range))
        cfg.led = LedConfig(
            colors=led.get("colors", cfg.led.colors),
            shapes=led.get("shapes", cfg.led.shapes),
            size_range=(float(size_range[0]), float(size_range[1])),
            position_margin=float(led.get("position_margin", cfg.led.position_margin)),
        )

    if "backgrounds" in data:
        bg = data["backgrounds"]
        cfg.backgrounds = BackgroundWeights(
            plain=float(bg.get("plain", 0.25)),
            gradient=float(bg.get("gradient", 0.15)),
            pcb=float(bg.get("pcb", 0.30)),
            metal=float(bg.get("metal", 0.20)),
            noisy=float(bg.get("noisy", 0.10)),
        )

    if "variation" in data:
        v = data["variation"]
        cfg.variation = VariationConfig(
            blur_range=_pair(v.get("blur_range", list(cfg.variation.blur_range))),
            noise_sigma_range=_pair(v.get("noise_sigma_range", list(cfg.variation.noise_sigma_range))),
            camera_angle_range=_pair(v.get("camera_angle_range", list(cfg.variation.camera_angle_range))),
            ambient_factor_range=_pair(v.get("ambient_factor_range", list(cfg.variation.ambient_factor_range))),
            brightness_range=_pair(v.get("brightness_range", list(cfg.variation.brightness_range))),
        )

    return cfg


def _pair(value: list[float]) -> tuple[float, float]:
    return (float(value[0]), float(value[1]))
