"""LED shape renderer for synthetic image generation."""
from __future__ import annotations

import numpy as np
import cv2

# BGR color definitions for all 14 LED colors
LED_COLORS_BGR: dict[str, tuple[int, int, int]] = {
    "red":        (30,  30,  255),
    "green":      (30,  255, 30),
    "yellow":     (0,   230, 255),
    "blue":       (255, 80,  30),
    "white":      (255, 255, 255),
    "orange":     (0,   130, 255),
    "purple":     (220, 0,   200),
    "cyan":       (255, 220, 0),
    "magenta":    (200, 0,   255),
    "amber":      (0,   160, 255),
    "warm_white": (160, 215, 255),
    "cool_white": (255, 230, 210),
    "lime_green": (0,   255, 130),
    "uv_violet":  (255, 0,   140),
}

_GLOW_INTENSITY = 0.75
_OFF_DIM_FACTOR = 0.15
_OFF_RIM_FACTOR = 0.28
_HIGHLIGHT_SIZE_RATIO = 0.28
_HIGHLIGHT_OFFSET_RATIO = 0.28


class LEDRenderer:
    """Draws LED indicators onto images with realistic ON/OFF appearance."""

    def __init__(self, rng: np.random.Generator) -> None:
        self.rng = rng

    def render(
        self,
        image: np.ndarray,
        state: str,
        color_name: str,
        shape: str,
        size_pct: float,
        pos_x: float,
        pos_y: float,
    ) -> np.ndarray:
        """
        Draw an LED on *image* and return a new image.

        Args:
            image:      BGR uint8 input image.
            state:      "on" or "off".
            color_name: Key from LED_COLORS_BGR.
            shape:      "round", "square", or "rectangular".
            size_pct:   LED size as fraction of min(width, height).
            pos_x:      Horizontal centre position (0.0–1.0).
            pos_y:      Vertical centre position (0.0–1.0).
        """
        if color_name not in LED_COLORS_BGR:
            raise ValueError(f"Unknown LED color: {color_name!r}")
        if shape not in ("round", "square", "rectangular"):
            raise ValueError(f"Unknown LED shape: {shape!r}")
        if state not in ("on", "off"):
            raise ValueError(f"Unknown LED state: {state!r}")

        h, w = image.shape[:2]
        color = LED_COLORS_BGR[color_name]
        radius = max(2, int(min(w, h) * size_pct / 2))
        cx = int(w * pos_x)
        cy = int(h * pos_y)

        if state == "on":
            return self._draw_on(image, shape, cx, cy, radius, color)
        return self._draw_off(image, shape, cx, cy, radius, color)

    # ------------------------------------------------------------------
    # ON / OFF drawing
    # ------------------------------------------------------------------

    def _draw_on(
        self,
        image: np.ndarray,
        shape: str,
        cx: int, cy: int, radius: int,
        color: tuple[int, int, int],
    ) -> np.ndarray:
        result = image.astype(np.float32)

        # Glow halo: draw shape into blank layer, then blur heavily
        glow_layer = np.zeros_like(result)
        self._fill_shape(glow_layer, shape, cx, cy, radius, color)
        glow_sigma = max(2.0, radius * 0.9)
        ksize = int(glow_sigma * 6) | 1
        glow_blurred = cv2.GaussianBlur(glow_layer, (ksize, ksize), glow_sigma)
        result = np.clip(result + glow_blurred * _GLOW_INTENSITY, 0, 255)

        result_u8 = result.astype(np.uint8)

        # LED body
        self._fill_shape(result_u8, shape, cx, cy, radius, color)

        # Specular highlight (small bright spot in upper-left of LED)
        hl_r = max(1, int(radius * _HIGHLIGHT_SIZE_RATIO))
        hl_off = max(1, int(radius * _HIGHLIGHT_OFFSET_RATIO))
        self._fill_shape(result_u8, shape, cx - hl_off, cy - hl_off, hl_r, (255, 255, 255))

        return result_u8

    def _draw_off(
        self,
        image: np.ndarray,
        shape: str,
        cx: int, cy: int, radius: int,
        color: tuple[int, int, int],
    ) -> np.ndarray:
        result = image.copy()

        dim = tuple(max(0, int(c * _OFF_DIM_FACTOR)) for c in color)
        self._fill_shape(result, shape, cx, cy, radius, dim)

        # Subtle rim to show the LED housing
        rim = tuple(max(0, int(c * _OFF_RIM_FACTOR)) for c in color)
        thickness = max(1, radius // 15)
        self._outline_shape(result, shape, cx, cy, radius, rim, thickness)

        return result

    # ------------------------------------------------------------------
    # Shape primitives
    # ------------------------------------------------------------------

    def _fill_shape(
        self,
        image: np.ndarray,
        shape: str,
        cx: int, cy: int, radius: int,
        color: tuple[int, ...],
    ) -> None:
        c = tuple(int(v) for v in color)
        if shape == "round":
            cv2.circle(image, (cx, cy), max(1, radius), c, -1)
        elif shape == "square":
            r = max(1, radius)
            cv2.rectangle(image, (cx - r, cy - r), (cx + r, cy + r), c, -1)
        elif shape == "rectangular":
            hw = max(1, int(radius * 1.7))
            hh = max(1, int(radius * 0.65))
            cv2.rectangle(image, (cx - hw, cy - hh), (cx + hw, cy + hh), c, -1)

    def _outline_shape(
        self,
        image: np.ndarray,
        shape: str,
        cx: int, cy: int, radius: int,
        color: tuple[int, ...],
        thickness: int,
    ) -> None:
        c = tuple(int(v) for v in color)
        if shape == "round":
            cv2.circle(image, (cx, cy), max(1, radius), c, thickness)
        elif shape == "square":
            r = max(1, radius)
            cv2.rectangle(image, (cx - r, cy - r), (cx + r, cy + r), c, thickness)
        elif shape == "rectangular":
            hw = max(1, int(radius * 1.7))
            hh = max(1, int(radius * 0.65))
            cv2.rectangle(image, (cx - hw, cy - hh), (cx + hw, cy + hh), c, thickness)
