"""Dataset generation configuration — v2 full parameter space."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


# ---------------------------------------------------------------------------
# Sub-configs
# ---------------------------------------------------------------------------

@dataclass
class SplitRatios:
    train: float = 0.70
    val: float = 0.15
    test: float = 0.15

    def __post_init__(self) -> None:
        if abs(self.train + self.val + self.test - 1.0) > 1e-4:
            raise ValueError(f"Split ratios must sum to 1.0")


@dataclass
class LedConfig:
    colors: list[str] = field(default_factory=lambda: [
        "red", "green", "yellow", "blue", "white", "orange",
        "purple", "cyan", "magenta", "amber", "warm_white",
        "cool_white", "lime_green", "uv_violet",
    ])
    shapes: list[str] = field(default_factory=lambda: [
        "round", "square", "rectangular", "dome", "dome_side", "smd",
    ])
    lens_types: list[str] = field(default_factory=lambda: ["clear", "diffused"])
    brightness_range: tuple[float, float] = (0.05, 1.0)
    size_range: tuple[float, float] = (0.04, 0.22)
    position_margin: float = 0.05
    glow_radius_range: tuple[float, float] = (0.6, 3.0)
    halo_intensity_range: tuple[float, float] = (0.1, 1.0)


@dataclass
class MultiLedConfig:
    single: float = 0.60
    two: float = 0.18
    four: float = 0.12
    row: float = 0.07
    grid: float = 0.03
    mixed_states: bool = True


@dataclass
class BackgroundWeights:
    black_plastic: float   = 0.09
    gray_plastic: float    = 0.08
    white_plastic: float   = 0.05
    dark_enclosure: float  = 0.07
    light_enclosure: float = 0.06
    pcb: float             = 0.09
    blue_pcb: float        = 0.05
    realistic_pcb: float   = 0.09
    metal: float           = 0.06
    brushed_metal: float   = 0.05
    white_studio: float    = 0.05
    aluminum_pcb: float    = 0.05
    textured: float        = 0.05
    gradient: float        = 0.05
    plain: float           = 0.06


@dataclass
class AmbientWeights:
    dark_room: float      = 0.10
    normal_indoor: float  = 0.35
    bright_office: float  = 0.15
    directional: float    = 0.15
    low_light: float      = 0.10
    warm: float           = 0.05
    cool: float           = 0.05
    backlight: float      = 0.05


@dataclass
class GlareWeights:
    none: float               = 0.50
    small_highlight: float    = 0.20
    strong_glare: float       = 0.10
    reflection_near: float    = 0.10
    reflection_overlap: float = 0.05
    glossy: float             = 0.05


@dataclass
class OcclusionWeights:
    none: float        = 0.75
    cable: float       = 0.08
    label: float       = 0.07
    shadow: float      = 0.05
    panel_edge: float  = 0.05


@dataclass
class FalsePositiveConfig:
    enabled: bool = True
    rate: float = 0.05
    types: list[str] = field(default_factory=lambda: [
        "colored_screw", "reflection_spot", "printed_circle",
        "colored_button", "display_pixel", "specular_highlight",
    ])


@dataclass
class CameraConfig:
    noise_sigma_range: tuple[float, float]   = (0.0, 20.0)
    blur_sigma_range: tuple[float, float]    = (0.0, 3.0)
    motion_blur_prob: float                  = 0.10
    motion_blur_length_range: tuple[int, int]= (3, 15)
    chromatic_shift_range: tuple[float, float]= (0.0, 3.0)
    gamma_range: tuple[float, float]         = (0.7, 1.5)
    white_balance_shift: float               = 0.10
    jpeg_artifact_prob: float                = 0.10
    jpeg_quality_range: tuple[int, int]      = (60, 85)


@dataclass
class DifficultyWeights:
    easy: float    = 0.25
    medium: float  = 0.35
    hard: float    = 0.25
    extreme: float = 0.15


@dataclass
class GroupWeights:
    clean_easy: float            = 0.20
    lighting_variation: float    = 0.15
    camera_variation: float      = 0.15
    geometry_perspective: float  = 0.10
    glare_reflections: float     = 0.10
    blur_noise: float            = 0.10
    difficult_on_off: float      = 0.10
    false_positives: float       = 0.05
    occlusion: float             = 0.05


# ---------------------------------------------------------------------------
# Root config
# ---------------------------------------------------------------------------

@dataclass
class DatasetConfig:
    seed: int = 42
    total: int = 100
    master_size: int = 512
    export_sizes: list[int] = field(default_factory=lambda: [64, 128, 224, 256])
    output_dir: Path = field(default_factory=lambda: Path("data/generated"))
    image_format: str = "jpg"
    image_quality: int = 95
    n_workers: int = 1              # parallel worker processes (1 = single-threaded)

    # balanced | imbalanced_80_20 | imbalanced_95_5 | rare_fault
    class_balance: str = "balanced"

    split: SplitRatios       = field(default_factory=SplitRatios)
    led: LedConfig           = field(default_factory=LedConfig)
    multi_led: MultiLedConfig= field(default_factory=MultiLedConfig)
    backgrounds: BackgroundWeights = field(default_factory=BackgroundWeights)
    ambient: AmbientWeights  = field(default_factory=AmbientWeights)
    glare: GlareWeights      = field(default_factory=GlareWeights)
    occlusion: OcclusionWeights = field(default_factory=OcclusionWeights)
    false_positive: FalsePositiveConfig = field(default_factory=FalsePositiveConfig)
    camera: CameraConfig     = field(default_factory=CameraConfig)
    difficulty: DifficultyWeights = field(default_factory=DifficultyWeights)
    groups: GroupWeights     = field(default_factory=GroupWeights)


# ---------------------------------------------------------------------------
# YAML loader
# ---------------------------------------------------------------------------

def load_config(path: Path) -> DatasetConfig:
    with path.open(encoding="utf-8") as f:
        d: dict[str, Any] = yaml.safe_load(f)

    cfg = DatasetConfig()

    for key in ("seed", "total", "master_size", "image_format", "image_quality",
                "class_balance", "n_workers"):
        if key in d:
            setattr(cfg, key, d[key])

    if "output_dir" in d:
        cfg.output_dir = Path(d["output_dir"])
    if "export_sizes" in d:
        cfg.export_sizes = [int(s) for s in d["export_sizes"]]

    if "split" in d:
        s = d["split"]
        cfg.split = SplitRatios(
            train=float(s.get("train", 0.70)),
            val=float(s.get("val", 0.15)),
            test=float(s.get("test", 0.15)),
        )

    if "led" in d:
        v = d["led"]
        cfg.led = LedConfig(
            colors=v.get("colors", cfg.led.colors),
            shapes=v.get("shapes", cfg.led.shapes),
            lens_types=v.get("lens_types", cfg.led.lens_types),
            brightness_range=_pair(v.get("brightness_range", cfg.led.brightness_range)),
            size_range=_pair(v.get("size_range", cfg.led.size_range)),
            position_margin=float(v.get("position_margin", cfg.led.position_margin)),
            glow_radius_range=_pair(v.get("glow_radius_range", cfg.led.glow_radius_range)),
            halo_intensity_range=_pair(v.get("halo_intensity_range", cfg.led.halo_intensity_range)),
        )

    if "multi_led" in d:
        v = d["multi_led"]
        cfg.multi_led = MultiLedConfig(
            single=float(v.get("single", 0.60)),
            two=float(v.get("two", 0.18)),
            four=float(v.get("four", 0.12)),
            row=float(v.get("row", 0.07)),
            grid=float(v.get("grid", 0.03)),
            mixed_states=bool(v.get("mixed_states", True)),
        )

    if "backgrounds" in d:
        v = d["backgrounds"]
        cfg.backgrounds = BackgroundWeights(**{
            k: float(v.get(k, getattr(cfg.backgrounds, k)))
            for k in BackgroundWeights.__dataclass_fields__
        })

    if "ambient" in d:
        v = d["ambient"]
        cfg.ambient = AmbientWeights(**{
            k: float(v.get(k, getattr(cfg.ambient, k)))
            for k in AmbientWeights.__dataclass_fields__
        })

    if "glare" in d:
        v = d["glare"]
        cfg.glare = GlareWeights(**{
            k: float(v.get(k, getattr(cfg.glare, k)))
            for k in GlareWeights.__dataclass_fields__
        })

    if "occlusion" in d:
        v = d["occlusion"]
        cfg.occlusion = OcclusionWeights(**{
            k: float(v.get(k, getattr(cfg.occlusion, k)))
            for k in OcclusionWeights.__dataclass_fields__
        })

    if "false_positive" in d:
        v = d["false_positive"]
        cfg.false_positive = FalsePositiveConfig(
            enabled=bool(v.get("enabled", True)),
            rate=float(v.get("rate", 0.05)),
            types=v.get("types", cfg.false_positive.types),
        )

    if "camera" in d:
        v = d["camera"]
        cfg.camera = CameraConfig(
            noise_sigma_range=_pair(v.get("noise_sigma_range", cfg.camera.noise_sigma_range)),
            blur_sigma_range=_pair(v.get("blur_sigma_range", cfg.camera.blur_sigma_range)),
            motion_blur_prob=float(v.get("motion_blur_prob", cfg.camera.motion_blur_prob)),
            motion_blur_length_range=tuple(v.get("motion_blur_length_range", list(cfg.camera.motion_blur_length_range))),
            chromatic_shift_range=_pair(v.get("chromatic_shift_range", cfg.camera.chromatic_shift_range)),
            gamma_range=_pair(v.get("gamma_range", cfg.camera.gamma_range)),
            white_balance_shift=float(v.get("white_balance_shift", cfg.camera.white_balance_shift)),
            jpeg_artifact_prob=float(v.get("jpeg_artifact_prob", cfg.camera.jpeg_artifact_prob)),
            jpeg_quality_range=tuple(v.get("jpeg_quality_range", list(cfg.camera.jpeg_quality_range))),
        )

    if "difficulty" in d:
        v = d["difficulty"]
        cfg.difficulty = DifficultyWeights(**{
            k: float(v.get(k, getattr(cfg.difficulty, k)))
            for k in DifficultyWeights.__dataclass_fields__
        })

    if "groups" in d:
        v = d["groups"]
        cfg.groups = GroupWeights(**{
            k: float(v.get(k, getattr(cfg.groups, k)))
            for k in GroupWeights.__dataclass_fields__
        })

    return cfg


def _pair(value: Any) -> tuple[float, float]:
    return (float(value[0]), float(value[1]))
