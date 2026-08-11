"""LED indicator renderer — v2 (brightness, lens type, multi-LED)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import cv2

# BGR tuples for each named LED color
LED_COLORS_BGR: dict[str, tuple[int, int, int]] = {
    "red":        (0,   20,  220),
    "green":      (20,  220, 20),
    "yellow":     (0,   230, 230),
    "blue":       (240, 60,  10),
    "white":      (255, 255, 255),
    "orange":     (0,   140, 255),
    "purple":     (200, 0,   200),
    "cyan":       (255, 220, 20),
    "magenta":    (220, 0,   220),
    "amber":      (0,   165, 255),
    "warm_white": (200, 230, 255),
    "cool_white": (255, 250, 220),
    "lime_green": (50,  255, 80),
    "uv_violet":  (255, 0,   160),
}

ALL_SHAPES = ("round", "square", "rectangular", "dome", "dome_side", "smd")


@dataclass
class LedSpec:
    """Per-LED rendering specification."""
    state: str          # "on" | "off"
    color: str          # key in LED_COLORS_BGR
    shape: str
    lens_type: str      # "clear" | "diffused"
    brightness: float   # 0.0–1.0
    size_pct: float     # fraction of image dimension
    pos_x: float        # 0.0–1.0 (center x)
    pos_y: float        # 0.0–1.0 (center y)
    glow_radius: float  # glow spread multiplier (0.6–3.0)
    halo_intensity: float  # 0.1–1.0

    @property
    def color_bgr(self) -> tuple[int, int, int]:
        return LED_COLORS_BGR.get(self.color, (255, 255, 255))


class LEDRenderer:
    def __init__(self, rng: np.random.Generator) -> None:
        self._rng = rng

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def render(
        self,
        image: np.ndarray,
        spec: LedSpec,
    ) -> tuple[np.ndarray, int, int, int]:
        """Draw one LED onto *image* (modified in-place copy).

        Returns ``(image, cx_px, cy_px, radius_px)`` in image pixel coords.
        """
        h, w = image.shape[:2]
        radius = max(4, int(min(w, h) * spec.size_pct / 2))
        cx = int(w * spec.pos_x)
        cy = int(h * spec.pos_y)

        method = {
            "round":       self._round,
            "square":      self._square,
            "rectangular": self._rectangular,
            "dome":        self._dome_topdown,
            "dome_side":   self._dome_side,
            "smd":         self._smd,
        }[spec.shape]

        result = method(image.copy(), spec, cx, cy, radius)
        return result, cx, cy, radius

    def render_multiple(
        self,
        image: np.ndarray,
        specs: list[LedSpec],
    ) -> tuple[np.ndarray, list[tuple[int, int, int]]]:
        """Render a list of LEDs. Returns (image, [(cx, cy, r), ...])."""
        positions: list[tuple[int, int, int]] = []
        canvas = image.copy()
        for spec in specs:
            canvas, cx, cy, r = self.render(canvas, spec)
            positions.append((cx, cy, r))
        return canvas, positions

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def _apply_glow(
        self,
        image: np.ndarray,
        cx: int,
        cy: int,
        radius: int,
        color_bgr: tuple[int, int, int],
        glow_radius: float,
        halo_intensity: float,
        lens_type: str,
    ) -> np.ndarray:
        glow_r = int(radius * (1 + glow_radius))
        glow_layer = np.zeros_like(image, dtype=np.float32)
        cv2.circle(glow_layer, (cx, cy), glow_r, color_bgr, -1)
        sigma = glow_r * (0.6 if lens_type == "clear" else 1.2)
        k = max(3, int(sigma * 4) | 1)
        glow_layer = cv2.GaussianBlur(glow_layer, (k, k), sigma)
        result = np.clip(image.astype(np.float32) + glow_layer * halo_intensity, 0, 255)
        return result.astype(np.uint8)

    def _apply_specular(
        self, image: np.ndarray, cx: int, cy: int, radius: int, lens_type: str
    ) -> np.ndarray:
        if lens_type == "diffused":
            return image
        hl_cx = cx - radius // 3 + int(self._rng.integers(-2, 3))
        hl_cy = cy - radius // 3 + int(self._rng.integers(-2, 3))
        hl_r = max(1, radius // 4)
        spec = np.zeros_like(image, dtype=np.float32)
        cv2.circle(spec, (hl_cx, hl_cy), hl_r, (255, 255, 255), -1)
        k = max(3, hl_r | 1)
        spec = cv2.GaussianBlur(spec, (k, k), hl_r * 0.5)
        return np.clip(image.astype(np.float32) + spec * 0.85, 0, 255).astype(np.uint8)

    def _scale_color(
        self, color_bgr: tuple[int, int, int], brightness: float
    ) -> tuple[int, int, int]:
        return tuple(int(c * brightness) for c in color_bgr)  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Shape renderers
    # ------------------------------------------------------------------

    def _round(
        self, img: np.ndarray, spec: LedSpec, cx: int, cy: int, r: int
    ) -> np.ndarray:
        if spec.state == "on":
            color = self._scale_color(spec.color_bgr, spec.brightness)
            img = self._apply_glow(img, cx, cy, r, color, spec.glow_radius,
                                   spec.halo_intensity, spec.lens_type)
            cv2.circle(img, (cx, cy), r, color, -1)
            img = self._apply_specular(img, cx, cy, r, spec.lens_type)
        else:
            housing = self._scale_color(spec.color_bgr, 0.08 + spec.brightness * 0.05)
            cv2.circle(img, (cx, cy), r, housing, -1)
            cv2.circle(img, (cx, cy), r, (30, 30, 30), 1)
        return img

    def _square(
        self, img: np.ndarray, spec: LedSpec, cx: int, cy: int, r: int
    ) -> np.ndarray:
        x1, y1, x2, y2 = cx - r, cy - r, cx + r, cy + r
        if spec.state == "on":
            color = self._scale_color(spec.color_bgr, spec.brightness)
            img = self._apply_glow(img, cx, cy, r, color, spec.glow_radius,
                                   spec.halo_intensity, spec.lens_type)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, -1)
            img = self._apply_specular(img, cx, cy, r, spec.lens_type)
        else:
            housing = self._scale_color(spec.color_bgr, 0.08 + spec.brightness * 0.05)
            cv2.rectangle(img, (x1, y1), (x2, y2), housing, -1)
            cv2.rectangle(img, (x1, y1), (x2, y2), (30, 30, 30), 1)
        return img

    def _rectangular(
        self, img: np.ndarray, spec: LedSpec, cx: int, cy: int, r: int
    ) -> np.ndarray:
        aspect = float(self._rng.uniform(1.6, 3.0))
        rx = max(2, int(r * aspect))
        ry = max(2, r)
        x1, y1, x2, y2 = cx - rx, cy - ry, cx + rx, cy + ry
        if spec.state == "on":
            color = self._scale_color(spec.color_bgr, spec.brightness)
            img = self._apply_glow(img, cx, cy, max(rx, ry), color, spec.glow_radius,
                                   spec.halo_intensity, spec.lens_type)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, -1)
            img = self._apply_specular(img, cx, cy, ry, spec.lens_type)
        else:
            housing = self._scale_color(spec.color_bgr, 0.08 + spec.brightness * 0.05)
            cv2.rectangle(img, (x1, y1), (x2, y2), housing, -1)
            cv2.rectangle(img, (x1, y1), (x2, y2), (30, 30, 30), 1)
        return img

    def _dome_topdown(
        self, img: np.ndarray, spec: LedSpec, cx: int, cy: int, r: int
    ) -> np.ndarray:
        # Copper pad ring around LED
        pad_color = (0, int(self._rng.integers(90, 140)), int(self._rng.integers(140, 200)))
        cv2.circle(img, (cx, cy), int(r * 1.35), pad_color, -1)
        cv2.circle(img, (cx, cy), int(r * 1.35), (20, 60, 60), 1)
        if spec.state == "on":
            color = self._scale_color(spec.color_bgr, spec.brightness)
            if spec.lens_type == "diffused":
                img = self._apply_glow(img, cx, cy, r, color, spec.glow_radius * 1.4,
                                       spec.halo_intensity, spec.lens_type)
            else:
                img = self._apply_glow(img, cx, cy, r, color, spec.glow_radius,
                                       spec.halo_intensity, spec.lens_type)
            cv2.circle(img, (cx, cy), r, color, -1)
            img = self._apply_specular(img, cx, cy, r, spec.lens_type)
        else:
            housing = self._scale_color(spec.color_bgr, 0.06 + spec.brightness * 0.04)
            cv2.circle(img, (cx, cy), r, housing, -1)
            cv2.circle(img, (cx, cy), r, (15, 15, 15), 1)
        return img

    def _dome_side(
        self, img: np.ndarray, spec: LedSpec, cx: int, cy: int, r: int
    ) -> np.ndarray:
        # Side-profile: oval body, flat base, two leads
        body_h = int(r * 1.5)
        body_w = r
        # Housing body (cylinder seen from side)
        housing_base = (40, 40, 40)
        cv2.ellipse(img, (cx, cy), (body_w, body_h // 2), 0, 180, 360, housing_base, -1)
        cv2.rectangle(img, (cx - body_w, cy), (cx + body_w, cy + body_h // 2), housing_base, -1)
        # Leads
        lead_len = int(r * 0.8)
        lead_color = (120, 120, 140)
        spacing = max(2, body_w // 2)
        cv2.line(img, (cx - spacing, cy + body_h // 2),
                 (cx - spacing, cy + body_h // 2 + lead_len), lead_color, max(1, r // 8))
        cv2.line(img, (cx + spacing, cy + body_h // 2),
                 (cx + spacing, cy + body_h // 2 + lead_len), lead_color, max(1, r // 8))
        if spec.state == "on":
            color = self._scale_color(spec.color_bgr, spec.brightness)
            if spec.lens_type == "diffused":
                lens_color = tuple(min(255, c + 40) for c in color)
                cv2.ellipse(img, (cx, cy), (body_w, body_h // 2), 0, 180, 360,
                            lens_color, -1)  # type: ignore[arg-type]
            else:
                cv2.ellipse(img, (cx, cy), (body_w, body_h // 2), 0, 180, 360, color, -1)
            img = self._apply_glow(img, cx, cy - body_h // 4, body_w, color,
                                   spec.glow_radius, spec.halo_intensity, spec.lens_type)
            img = self._apply_specular(img, cx - body_w // 3, cy - body_h // 4,
                                       max(2, body_w // 3), spec.lens_type)
        else:
            off_color = self._scale_color(spec.color_bgr, 0.07 + spec.brightness * 0.04)
            cv2.ellipse(img, (cx, cy), (body_w, body_h // 2), 0, 180, 360, off_color, -1)
        return img

    def _smd(
        self, img: np.ndarray, spec: LedSpec, cx: int, cy: int, r: int
    ) -> np.ndarray:
        # Surface-mount: flat rectangle with silver pads on sides
        pkg_w = int(r * 1.8)
        pkg_h = int(r * 1.0)
        pkg_color = (35, 35, 35)
        cv2.rectangle(img, (cx - pkg_w, cy - pkg_h), (cx + pkg_w, cy + pkg_h), pkg_color, -1)
        cv2.rectangle(img, (cx - pkg_w, cy - pkg_h), (cx + pkg_w, cy + pkg_h), (60, 60, 60), 1)
        # Silver pads
        pad_w = max(2, pkg_w // 4)
        pad_color = (190, 190, 200)
        cv2.rectangle(img, (cx - pkg_w, cy - pkg_h + 2),
                      (cx - pkg_w + pad_w, cy + pkg_h - 2), pad_color, -1)
        cv2.rectangle(img, (cx + pkg_w - pad_w, cy - pkg_h + 2),
                      (cx + pkg_w, cy + pkg_h - 2), pad_color, -1)
        # Lens window
        lens_r = max(2, pkg_h - 2)
        if spec.state == "on":
            color = self._scale_color(spec.color_bgr, spec.brightness)
            if spec.lens_type == "diffused":
                lens_r_big = int(lens_r * 1.3)
                img = self._apply_glow(img, cx, cy, lens_r_big, color, spec.glow_radius * 1.2,
                                       spec.halo_intensity, spec.lens_type)
                cv2.circle(img, (cx, cy), lens_r_big, color, -1)
            else:
                img = self._apply_glow(img, cx, cy, lens_r, color, spec.glow_radius,
                                       spec.halo_intensity, spec.lens_type)
                cv2.circle(img, (cx, cy), lens_r, color, -1)
            img = self._apply_specular(img, cx, cy, lens_r, spec.lens_type)
        else:
            off_color = self._scale_color(spec.color_bgr, 0.06 + spec.brightness * 0.04)
            cv2.circle(img, (cx, cy), lens_r, off_color, -1)
            cv2.circle(img, (cx, cy), lens_r, (55, 55, 55), 1)
        return img
