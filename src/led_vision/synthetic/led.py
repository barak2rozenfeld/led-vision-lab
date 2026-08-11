"""LED shape renderer for synthetic image generation.

Shapes
------
round       : flat filled circle (simple baseline)
square      : flat filled square
rectangular : flat filled rectangle
dome        : top-down view of through-hole LED — copper pad + translucent dome lens
dome_side   : side-profile of through-hole LED — dome body + flange + wire leads
smd         : top-down surface-mount LED chip — ceramic package + die
"""
from __future__ import annotations

import cv2
import numpy as np

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

ALL_SHAPES = ("round", "square", "rectangular", "dome", "dome_side", "smd")

_GLOW_INTENSITY = 0.75
_OFF_DIM_FACTOR = 0.15
_OFF_RIM_FACTOR = 0.28

# Copper / metal colors (BGR)
_COPPER = (30, 110, 185)
_SOLDER = (180, 185, 185)
_SILVER = (190, 195, 195)
_LEAD_WIRE = (160, 162, 162)
_PCB_HOLE = (8, 15, 8)
_SMD_CERAMIC = (220, 220, 215)
_SMD_PAD = (170, 175, 175)


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
        """Draw an LED on *image* and return a new image (uint8 BGR)."""
        if color_name not in LED_COLORS_BGR:
            raise ValueError(f"Unknown LED color: {color_name!r}")
        if shape not in ALL_SHAPES:
            raise ValueError(f"Unknown LED shape: {shape!r}")
        if state not in ("on", "off"):
            raise ValueError(f"Unknown LED state: {state!r}")

        h, w = image.shape[:2]
        color = LED_COLORS_BGR[color_name]
        radius = max(2, int(min(w, h) * size_pct / 2))
        cx = int(w * pos_x)
        cy = int(h * pos_y)

        dispatch = {
            "round":       self._simple_shape,
            "square":      self._simple_shape,
            "rectangular": self._simple_shape,
            "dome":        self._dome_topdown,
            "dome_side":   self._dome_side,
            "smd":         self._smd,
        }
        return dispatch[shape](image, state, shape, cx, cy, radius, color)

    # ------------------------------------------------------------------
    # Simple flat shapes (round / square / rectangular)
    # ------------------------------------------------------------------

    def _simple_shape(
        self, image: np.ndarray, state: str, shape: str,
        cx: int, cy: int, radius: int, color: tuple[int, int, int],
    ) -> np.ndarray:
        if state == "on":
            return self._draw_on_simple(image, shape, cx, cy, radius, color)
        return self._draw_off_simple(image, shape, cx, cy, radius, color)

    def _draw_on_simple(self, image, shape, cx, cy, radius, color):
        result = image.astype(np.float32)
        glow = np.zeros_like(result)
        self._fill(glow, shape, cx, cy, radius, color)
        sigma = max(2.0, radius * 0.9)
        k = int(sigma * 6) | 1
        glow_blur = cv2.GaussianBlur(glow, (k, k), sigma)
        result = np.clip(result + glow_blur * _GLOW_INTENSITY, 0, 255).astype(np.uint8)
        self._fill(result, shape, cx, cy, radius, color)
        hl_r = max(1, int(radius * 0.28))
        hl_off = max(1, int(radius * 0.28))
        self._fill(result, shape, cx - hl_off, cy - hl_off, hl_r, (255, 255, 255))
        return result

    def _draw_off_simple(self, image, shape, cx, cy, radius, color):
        result = image.copy()
        dim = tuple(max(0, int(c * _OFF_DIM_FACTOR)) for c in color)
        self._fill(result, shape, cx, cy, radius, dim)
        rim = tuple(max(0, int(c * _OFF_RIM_FACTOR)) for c in color)
        self._outline(result, shape, cx, cy, radius, rim, max(1, radius // 15))
        return result

    # ------------------------------------------------------------------
    # Dome — top-down view (through-hole LED on PCB)
    # ------------------------------------------------------------------

    def _dome_topdown(
        self, image: np.ndarray, state: str, _shape: str,
        cx: int, cy: int, radius: int, color: tuple[int, int, int],
    ) -> np.ndarray:
        result = image.copy()
        pad_r = max(radius + 3, int(radius * 1.45))

        # Copper mounting pad
        cv2.circle(result, (cx, cy), pad_r, _COPPER, -1)
        # Solder ring
        cv2.circle(result, (cx, cy), pad_r, _SOLDER, max(1, pad_r // 5))
        # Through-hole
        hole_r = max(1, radius // 5)
        cv2.circle(result, (cx, cy), hole_r, _PCB_HOLE, -1)

        if state == "on":
            # Glow halo
            glow = np.zeros_like(result, dtype=np.float32)
            cv2.circle(glow, (cx, cy), radius, color, -1)
            sigma = max(2.0, radius * 1.0)
            k = int(sigma * 6) | 1
            glow_blur = cv2.GaussianBlur(glow, (k, k), sigma)
            result = np.clip(result.astype(np.float32) + glow_blur * _GLOW_INTENSITY, 0, 255).astype(np.uint8)

            # LED dome body
            cv2.circle(result, (cx, cy), radius, color, -1)

            # Bright diffuse center
            inner_r = max(1, int(radius * 0.55))
            bright = tuple(min(255, int(c * 1.4) + 40) for c in color)
            cv2.circle(result, (cx, cy), inner_r, bright, -1)

            # Specular highlight
            hl_r = max(1, int(radius * 0.22))
            hl_off = max(1, int(radius * 0.32))
            cv2.circle(result, (cx - hl_off, cy - hl_off), hl_r, (255, 255, 255), -1)

        else:
            # Dark translucent dome
            dim = tuple(max(0, int(c * _OFF_DIM_FACTOR)) for c in color)
            cv2.circle(result, (cx, cy), radius, dim, -1)

            # Cathode flat marker (LED polarity indicator — flat spot on one side)
            marker_angle = float(self.rng.uniform(0, 360))
            marker_r = int(radius * 0.85)
            mx = cx + int(marker_r * 0.75 * np.cos(np.radians(marker_angle)))
            my = cy + int(marker_r * 0.75 * np.sin(np.radians(marker_angle)))
            flat_r = max(1, int(radius * 0.22))
            flat_c = tuple(max(0, int(c * 0.08)) for c in color)
            cv2.circle(result, (mx, my), flat_r, flat_c, -1)

            # Subtle rim
            rim = tuple(max(0, int(c * _OFF_RIM_FACTOR)) for c in color)
            cv2.circle(result, (cx, cy), radius, rim, max(1, radius // 12))

        return result

    # ------------------------------------------------------------------
    # Dome — side profile view (isolated component / studio)
    # ------------------------------------------------------------------

    def _dome_side(
        self, image: np.ndarray, state: str, _shape: str,
        cx: int, cy: int, radius: int, color: tuple[int, int, int],
    ) -> np.ndarray:
        result = image.copy()

        # Component geometry (all relative to radius)
        dome_r       = radius
        body_h       = max(4, int(radius * 0.55))
        flange_w     = max(radius + 2, int(radius * 1.25))
        flange_h     = max(2, int(radius * 0.18))
        lead_len     = max(8, int(radius * 1.8))
        lead_thick   = max(1, int(radius * 0.08))
        lead_sep     = max(2, int(radius * 0.30))

        # Layout
        dome_cy   = cy
        body_top  = dome_cy
        body_bot  = body_top + body_h
        flange_y1 = body_bot
        flange_y2 = body_bot + flange_h
        lead_top  = flange_y2
        lead_bot  = lead_top + lead_len

        if state == "on":
            # Glow emanating from dome
            glow = np.zeros_like(result, dtype=np.float32)
            cv2.ellipse(glow, (cx, dome_cy), (dome_r, dome_r), 0, 180, 360, color, -1)
            sigma = max(2.0, radius * 0.9)
            k = int(sigma * 6) | 1
            glow_blur = cv2.GaussianBlur(glow, (k, k), sigma)
            result = np.clip(result.astype(np.float32) + glow_blur * _GLOW_INTENSITY, 0, 255).astype(np.uint8)

            # Dome (upper hemisphere)
            cv2.ellipse(result, (cx, dome_cy), (dome_r, dome_r), 0, 180, 360, color, -1)

            # Bright inner glow in dome
            inner_rx = max(1, int(dome_r * 0.55))
            inner_ry = max(1, int(dome_r * 0.45))
            bright = tuple(min(255, int(c * 1.4) + 40) for c in color)
            cv2.ellipse(result, (cx, dome_cy - dome_r // 4), (inner_rx, inner_ry), 0, 180, 360, bright, -1)

            # Specular highlight
            hl_r = max(1, dome_r // 5)
            cv2.circle(result, (cx - dome_r // 3, dome_cy - dome_r // 2), hl_r, (255, 255, 255), -1)

        else:
            # Dim dome (OFF)
            dim = tuple(max(0, int(c * _OFF_DIM_FACTOR)) for c in color)
            cv2.ellipse(result, (cx, dome_cy), (dome_r, dome_r), 0, 180, 360, dim, -1)
            rim = tuple(max(0, int(c * _OFF_RIM_FACTOR)) for c in color)
            cv2.ellipse(result, (cx, dome_cy), (dome_r, dome_r), 0, 180, 360, rim, max(1, dome_r // 10))

        # Body (cylindrical)
        body_color = tuple(max(0, int(c * (_OFF_DIM_FACTOR + 0.05))) for c in color)
        cv2.rectangle(result, (cx - dome_r, body_top), (cx + dome_r, body_bot), body_color, -1)
        cv2.rectangle(result, (cx - dome_r, body_top), (cx + dome_r, body_bot), _SILVER, max(1, dome_r // 12))

        # Flange
        cv2.rectangle(result, (cx - flange_w, flange_y1), (cx + flange_w, flange_y2), _SILVER, -1)

        # Wire leads
        l1x = cx - lead_sep
        l2x = cx + lead_sep
        cv2.line(result, (l1x, lead_top), (l1x, lead_bot), _LEAD_WIRE, lead_thick)
        cv2.line(result, (l2x, lead_top), (l2x, lead_bot), _LEAD_WIRE, lead_thick)

        return result

    # ------------------------------------------------------------------
    # SMD LED — top-down view
    # ------------------------------------------------------------------

    def _smd(
        self, image: np.ndarray, state: str, _shape: str,
        cx: int, cy: int, radius: int, color: tuple[int, int, int],
    ) -> np.ndarray:
        result = image.copy()

        pkg_w  = max(4, int(radius * 1.8))   # package width  (x axis)
        pkg_h  = max(3, int(radius * 1.1))   # package height (y axis)
        pad_w  = max(2, int(radius * 0.4))   # metal pad width on each end
        die_r  = max(1, int(radius * 0.35))  # LED die radius

        # Ceramic package body
        cv2.rectangle(
            result,
            (cx - pkg_w, cy - pkg_h),
            (cx + pkg_w, cy + pkg_h),
            _SMD_CERAMIC, -1,
        )
        cv2.rectangle(
            result,
            (cx - pkg_w, cy - pkg_h),
            (cx + pkg_w, cy + pkg_h),
            (180, 180, 175), max(1, radius // 10),
        )

        # Metal connection pads (left and right)
        for side in (-1, 1):
            pad_x1 = cx + side * pkg_w - side * pad_w
            pad_x2 = cx + side * pkg_w
            if pad_x1 > pad_x2:
                pad_x1, pad_x2 = pad_x2, pad_x1
            cv2.rectangle(result, (pad_x1, cy - pkg_h), (pad_x2, cy + pkg_h), _SMD_PAD, -1)

        if state == "on":
            # Glow
            glow = np.zeros_like(result, dtype=np.float32)
            cv2.circle(glow, (cx, cy), die_r, color, -1)
            sigma = max(2.0, radius * 0.9)
            k = int(sigma * 6) | 1
            glow_blur = cv2.GaussianBlur(glow, (k, k), sigma)
            result = np.clip(result.astype(np.float32) + glow_blur * _GLOW_INTENSITY * 1.2, 0, 255).astype(np.uint8)

            # Die (bright)
            bright = tuple(min(255, int(c * 1.3) + 50) for c in color)
            cv2.circle(result, (cx, cy), die_r, bright, -1)

            # Specular
            hl_r = max(1, die_r // 3)
            cv2.circle(result, (cx - die_r // 2, cy - die_r // 2), hl_r, (255, 255, 255), -1)

        else:
            # Die (off — very dark)
            dim = tuple(max(0, int(c * 0.10)) for c in color)
            cv2.circle(result, (cx, cy), die_r, dim, -1)
            cv2.circle(result, (cx, cy), die_r, (120, 120, 118), max(1, die_r // 4))

        return result

    # ------------------------------------------------------------------
    # Primitive helpers
    # ------------------------------------------------------------------

    def _fill(self, image, shape, cx, cy, radius, color):
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

    def _outline(self, image, shape, cx, cy, radius, color, thickness):
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
