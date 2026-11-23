"""Scene builder — difficulty-driven composition with multi-LED, FP, and effects."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .backgrounds import BackgroundGenerator
from .config import DatasetConfig
from .effects import (
    apply_ambient,
    apply_chromatic_aberration,
    apply_false_positive,
    apply_gamma,
    apply_gaussian_blur,
    apply_glare,
    apply_jpeg_artifacts,
    apply_motion_blur,
    apply_noise,
    apply_occlusion,
    apply_rotation,
    apply_white_balance,
)
from .led import LEDRenderer, LedSpec


# ---------------------------------------------------------------------------
# Difficulty parameter bounds
# ---------------------------------------------------------------------------

_DIFFICULTY_PARAMS: dict[str, dict[str, Any]] = {
    "easy": {
        "brightness_range": (0.65, 1.0),
        "blur_range":       (0.0, 0.5),
        "noise_range":      (0.0, 5.0),
        "glare_types":      ["none"],
        "occlusion_types":  ["none"],
        "ambient_range":    (0.75, 1.25),
        "motion_blur_prob": 0.0,
        "chromatic_range":  (0.0, 0.5),
        "gamma_range":      (0.90, 1.10),
        "wb_shift":         0.03,
        "jpeg_prob":        0.0,
    },
    "medium": {
        "brightness_range": (0.35, 0.75),
        "blur_range":       (0.3, 2.0),
        "noise_range":      (3.0, 12.0),
        "glare_types":      ["none", "none", "small_highlight"],
        "occlusion_types":  ["none", "none", "shadow"],
        "ambient_range":    (0.45, 1.30),
        "motion_blur_prob": 0.08,
        "chromatic_range":  (0.0, 1.5),
        "gamma_range":      (0.75, 1.40),
        "wb_shift":         0.08,
        "jpeg_prob":        0.10,
    },
    "hard": {
        "brightness_range": (0.12, 0.40),
        "blur_range":       (1.5, 3.5),
        "noise_range":      (10.0, 20.0),
        "glare_types":      ["none", "small_highlight", "strong_glare", "reflection_near"],
        "occlusion_types":  ["none", "cable", "shadow", "label"],
        "ambient_range":    (0.15, 0.80),
        "motion_blur_prob": 0.25,
        "chromatic_range":  (1.0, 3.0),
        "gamma_range":      (0.60, 1.60),
        "wb_shift":         0.12,
        "jpeg_prob":        0.20,
    },
    "extreme": {
        "brightness_range": (0.03, 0.18),
        "blur_range":       (3.0, 6.0),
        "noise_range":      (15.0, 35.0),
        "glare_types":      ["strong_glare", "reflection_overlap", "glossy"],
        "occlusion_types":  ["cable", "label", "shadow", "panel_edge"],
        "ambient_range":    (0.05, 0.40),
        "motion_blur_prob": 0.45,
        "chromatic_range":  (2.0, 4.0),
        "gamma_range":      (0.50, 1.90),
        "wb_shift":         0.18,
        "jpeg_prob":        0.35,
    },
}


# ---------------------------------------------------------------------------
# Scene result
# ---------------------------------------------------------------------------

@dataclass
class SceneResult:
    """Everything produced for one generated scene."""
    image: np.ndarray                 # master-size rendered image (uint8 BGR)
    state: str                        # "on" | "off"
    is_false_positive: bool
    led_count: int
    leds: list[dict[str, Any]]        # per-LED metadata
    led_positions: list[tuple[int, int, int]]  # (cx, cy, r) in master pixels
    background: str
    ambient_type: str
    ambient_factor: float
    view_angle: float
    glare_type: str
    occlusion_type: str
    blur_sigma: float
    noise_sigma: float
    motion_blur: bool
    motion_blur_angle: float | None
    motion_blur_length: int | None
    chromatic_shift: float
    gamma: float
    white_balance_shift: float
    jpeg_quality: int | None
    difficulty: str
    group: str
    seed: int
    fp_type: str | None = None


# ---------------------------------------------------------------------------
# Scene builder
# ---------------------------------------------------------------------------

class SceneBuilder:
    """Assembles one fully-rendered scene from config + difficulty level."""

    def __init__(self, config: DatasetConfig, rng: np.random.Generator) -> None:
        self._cfg = config
        self._rng = rng
        self._bg = BackgroundGenerator(rng)
        self._led = LEDRenderer(rng)

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def build(
        self,
        target_state: str,  # "on" | "off"
        difficulty: str,
        group: str,
        scene_seed: int,
    ) -> SceneResult:
        params = _DIFFICULTY_PARAMS[difficulty]
        size = self._cfg.master_size
        cfg = self._cfg

        # --- False positive decision ---
        is_fp = (
            target_state == "off"
            and cfg.false_positive.enabled
            and float(self._rng.random()) < cfg.false_positive.rate
        )

        # --- Background ---
        bg_style = _weighted_choice(self._rng, _dataclass_weights(cfg.backgrounds))
        canvas = self._bg.generate(bg_style, size)

        led_positions: list[tuple[int, int, int]] = []
        led_meta: list[dict[str, Any]] = []

        if is_fp:
            fp_type = str(self._rng.choice(cfg.false_positive.types))
            canvas, fp_cx, fp_cy, fp_r = apply_false_positive(canvas, fp_type, self._rng)
            led_positions.append((fp_cx, fp_cy, fp_r))
        else:
            fp_type = None
            # --- LED count ---
            led_count_label = _weighted_choice(
                self._rng, _dataclass_weights(cfg.multi_led,
                                              exclude=("mixed_states",))
            )
            led_count = {"single": 1, "two": 2, "four": 4, "row": 4, "grid": 4}.get(
                led_count_label, 1
            )

            # --- Position layout ---
            positions_xy = _layout_positions(
                self._rng, led_count, led_count_label, size, cfg.led.position_margin
            )

            for i, (px, py) in enumerate(positions_xy):
                # The scene label is "on" if any LED is lit, so an "off" scene must
                # keep every LED off; only "on" scenes may mix states.
                if i == 0 or not cfg.multi_led.mixed_states or target_state == "off":
                    led_state = target_state
                else:
                    led_state = str(self._rng.choice(["on", "off"]))

                color = str(self._rng.choice(cfg.led.colors))
                shape = str(self._rng.choice(cfg.led.shapes))
                lens = str(self._rng.choice(cfg.led.lens_types))
                brightness = float(self._rng.uniform(*params["brightness_range"]))
                size_pct = float(self._rng.uniform(*cfg.led.size_range))
                glow_r = float(self._rng.uniform(*cfg.led.glow_radius_range))
                halo_i = float(self._rng.uniform(*cfg.led.halo_intensity_range))

                spec = LedSpec(
                    state=led_state,
                    color=color,
                    shape=shape,
                    lens_type=lens,
                    brightness=brightness,
                    size_pct=size_pct,
                    pos_x=px,
                    pos_y=py,
                    glow_radius=glow_r,
                    halo_intensity=halo_i,
                )
                canvas, cx, cy, r = self._led.render(canvas, spec)
                led_positions.append((cx, cy, r))
                led_meta.append({
                    "state": led_state,
                    "color": color,
                    "shape": shape,
                    "lens_type": lens,
                    "brightness": round(brightness, 3),
                    "size_pct": round(size_pct, 3),
                    "pos_x": round(px, 3),
                    "pos_y": round(py, 3),
                    "glow_radius": round(glow_r, 3),
                    "halo_intensity": round(halo_i, 3),
                    "cx_master": cx,
                    "cy_master": cy,
                    "radius_master": r,
                })

        # --- Ambient lighting ---
        ambient_type = _weighted_choice(self._rng, _dataclass_weights(cfg.ambient))
        ambient_factor = float(self._rng.uniform(*params["ambient_range"]))
        canvas = apply_ambient(canvas, ambient_type, ambient_factor, self._rng)

        # --- Glare (only when there is something glowing) ---
        on_positions = [p for p, m in zip(led_positions, led_meta)
                        if m.get("state", "off") == "on"] if not is_fp else []
        glare_type = str(self._rng.choice(params["glare_types"]))
        if on_positions:
            canvas = apply_glare(canvas, glare_type, on_positions, self._rng)
        else:
            glare_type = "none"

        # --- Occlusion ---
        occlusion_type = str(self._rng.choice(params["occlusion_types"]))
        canvas = apply_occlusion(canvas, occlusion_type, led_positions, self._rng)

        # --- Camera effects ---
        view_angle = float(self._rng.uniform(-12, 12))
        canvas = apply_rotation(canvas, view_angle)

        blur_sigma = float(self._rng.uniform(*params["blur_range"]))
        canvas = apply_gaussian_blur(canvas, blur_sigma)

        noise_sigma = float(self._rng.uniform(*params["noise_range"]))
        canvas = apply_noise(canvas, noise_sigma, self._rng)

        motion_blur = float(self._rng.random()) < params["motion_blur_prob"]
        motion_angle: float | None = None
        motion_length: int | None = None
        if motion_blur:
            motion_angle = float(self._rng.uniform(0, 180))
            lo, hi = cfg.camera.motion_blur_length_range
            motion_length = int(self._rng.integers(lo, hi + 1))
            canvas = apply_motion_blur(canvas, motion_angle, motion_length)

        chrom_lo, chrom_hi = params["chromatic_range"]
        chromatic_shift = float(self._rng.uniform(chrom_lo, chrom_hi))
        canvas = apply_chromatic_aberration(canvas, chromatic_shift)

        gam_lo, gam_hi = params["gamma_range"]
        gamma = float(self._rng.uniform(gam_lo, gam_hi))
        canvas = apply_gamma(canvas, gamma)

        wb_shift = float(params["wb_shift"])
        white_balance_shift = float(self._rng.uniform(0, wb_shift))
        canvas = apply_white_balance(canvas, white_balance_shift, self._rng)

        jpeg_quality: int | None = None
        if float(self._rng.random()) < params["jpeg_prob"]:
            lo_q, hi_q = cfg.camera.jpeg_quality_range
            jpeg_quality = int(self._rng.integers(lo_q, hi_q + 1))
            canvas = apply_jpeg_artifacts(canvas, jpeg_quality)

        # --- Final state label ---
        if is_fp:
            final_state = "off"
        else:
            final_state = "on" if any(m["state"] == "on" for m in led_meta) else "off"

        return SceneResult(
            image=canvas,
            state=final_state,
            is_false_positive=is_fp,
            led_count=1 if is_fp else len(led_meta),
            leds=led_meta,
            led_positions=led_positions,
            background=bg_style,
            ambient_type=ambient_type,
            ambient_factor=round(ambient_factor, 3),
            view_angle=round(view_angle, 2),
            glare_type=glare_type,
            occlusion_type=occlusion_type,
            blur_sigma=round(blur_sigma, 3),
            noise_sigma=round(noise_sigma, 3),
            motion_blur=motion_blur,
            motion_blur_angle=round(motion_angle, 2) if motion_angle is not None else None,
            motion_blur_length=motion_length,
            chromatic_shift=round(chromatic_shift, 3),
            gamma=round(gamma, 3),
            white_balance_shift=round(white_balance_shift, 4),
            jpeg_quality=jpeg_quality,
            difficulty=difficulty,
            group=group,
            seed=scene_seed,
            fp_type=fp_type,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _dataclass_weights(obj: Any, exclude: tuple[str, ...] = ()) -> dict[str, float]:
    """Extract {field_name: float_value} from a dataclass, skipping *exclude*."""
    from dataclasses import fields as dc_fields
    return {
        f.name: float(getattr(obj, f.name))
        for f in dc_fields(obj)
        if f.name not in exclude and isinstance(getattr(obj, f.name), (int, float))
    }


def _weighted_choice(rng: np.random.Generator, weights: dict[str, float]) -> str:
    names = list(weights.keys())
    probs = np.array(list(weights.values()), dtype=np.float64)
    probs /= probs.sum()
    idx = int(rng.choice(len(names), p=probs))
    return names[idx]


def _layout_positions(
    rng: np.random.Generator,
    count: int,
    layout: str,
    size: int,
    margin: float,
) -> list[tuple[float, float]]:
    lo, hi = margin, 1.0 - margin
    if count == 1:
        return [(float(rng.uniform(lo, hi)), float(rng.uniform(lo, hi)))]

    if layout == "row":
        xs = np.linspace(lo + 0.05, hi - 0.05, count)
        y = float(rng.uniform(lo + 0.1, hi - 0.1))
        return [(float(x), y) for x in xs]

    if layout == "grid":
        n = int(np.ceil(np.sqrt(count)))
        xs = np.linspace(lo + 0.05, hi - 0.05, n)
        ys = np.linspace(lo + 0.05, hi - 0.05, n)
        pts = [(float(x), float(y)) for y in ys for x in xs]
        return pts[:count]

    # Random non-overlapping positions for "two" / "four"
    positions: list[tuple[float, float]] = []
    for _ in range(count * 20):
        if len(positions) >= count:
            break
        x = float(rng.uniform(lo, hi))
        y = float(rng.uniform(lo, hi))
        if all(np.sqrt((x - px) ** 2 + (y - py) ** 2) > 0.20 for px, py in positions):
            positions.append((x, y))
    while len(positions) < count:
        positions.append((float(rng.uniform(lo, hi)), float(rng.uniform(lo, hi))))
    return positions
