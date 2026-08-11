"""Procedural background generators — v2 (15 styles)."""
from __future__ import annotations

import numpy as np
import cv2

BACKGROUND_STYLES = (
    "black_plastic", "gray_plastic", "white_plastic",
    "dark_enclosure", "light_enclosure",
    "pcb", "blue_pcb", "realistic_pcb",
    "metal", "brushed_metal", "aluminum_pcb",
    "white_studio", "textured", "gradient", "plain",
)


class BackgroundGenerator:
    def __init__(self, rng: np.random.Generator) -> None:
        self._rng = rng

    def generate(self, style: str, size: int) -> np.ndarray:
        if style not in BACKGROUND_STYLES:
            raise ValueError(f"Unknown background style {style!r}. Choose from {BACKGROUND_STYLES}")
        method = getattr(self, f"_{style}")
        return method(size)

    # ------------------------------------------------------------------
    # Plastic / enclosure styles
    # ------------------------------------------------------------------

    def _black_plastic(self, size: int) -> np.ndarray:
        base_v = int(self._rng.integers(10, 35))
        img = np.full((size, size, 3), base_v, dtype=np.float32)
        noise = self._rng.normal(0, 3, (size, size)).astype(np.float32)
        for ch in range(3):
            img[:, :, ch] = np.clip(img[:, :, ch] + noise, 0, 255)
        # Subtle glossy highlight near top-left
        hl_cx = int(size * self._rng.uniform(0.05, 0.35))
        hl_cy = int(size * self._rng.uniform(0.05, 0.35))
        hl_r = int(size * self._rng.uniform(0.08, 0.20))
        glow = np.zeros((size, size), dtype=np.float32)
        cv2.circle(glow, (hl_cx, hl_cy), hl_r, 1.0, -1)
        k = max(3, (hl_r | 1))
        glow = cv2.GaussianBlur(glow, (k, k), hl_r * 0.4)
        for ch in range(3):
            img[:, :, ch] = np.clip(img[:, :, ch] + glow * self._rng.uniform(20, 50), 0, 255)
        return img.astype(np.uint8)

    def _gray_plastic(self, size: int) -> np.ndarray:
        base_v = int(self._rng.integers(80, 145))
        img = np.full((size, size, 3), base_v, dtype=np.float32)
        # Matte texture: medium noise
        noise = self._rng.normal(0, 6, (size, size)).astype(np.float32)
        # Slightly warmer or cooler plastic tint
        tint = self._rng.choice(["warm", "cool", "neutral"])
        for ch in range(3):
            shift = {"warm": [0, 0, 4], "cool": [4, 0, 0], "neutral": [0, 0, 0]}[tint][ch]
            img[:, :, ch] = np.clip(img[:, :, ch] + noise + shift, 0, 255)
        # Faint surface imperfections
        n_scratches = int(self._rng.integers(2, 8))
        for _ in range(n_scratches):
            x1 = int(self._rng.integers(0, size))
            y1 = int(self._rng.integers(0, size))
            x2 = int(self._rng.integers(0, size))
            y2 = int(self._rng.integers(0, size))
            scratch_v = int(self._rng.integers(base_v - 20, base_v + 20))
            cv2.line(img.astype(np.uint8), (x1, y1), (x2, y2), (scratch_v,) * 3, 1)
        return np.clip(img, 0, 255).astype(np.uint8)

    def _white_plastic(self, size: int) -> np.ndarray:
        base_v = int(self._rng.integers(210, 245))
        img = np.full((size, size, 3), base_v, dtype=np.float32)
        noise = self._rng.normal(0, 4, (size, size)).astype(np.float32)
        # Slight yellowing variation
        yellowing = self._rng.uniform(0, 8)
        for ch in range(3):
            yellow_shift = [0, 0, yellowing][ch]
            img[:, :, ch] = np.clip(img[:, :, ch] + noise + yellow_shift, 0, 255)
        return img.astype(np.uint8)

    def _dark_enclosure(self, size: int) -> np.ndarray:
        # Very dark blue-black plastic electronics box
        b_base = int(self._rng.integers(5, 22))
        g_base = int(self._rng.integers(5, 20))
        r_base = int(self._rng.integers(3, 18))
        img = np.zeros((size, size, 3), dtype=np.float32)
        img[:, :, 0] = b_base
        img[:, :, 1] = g_base
        img[:, :, 2] = r_base
        noise = self._rng.normal(0, 2.5, (size, size)).astype(np.float32)
        for ch in range(3):
            img[:, :, ch] = np.clip(img[:, :, ch] + noise, 0, 255)
        # Venting slot pattern
        n_slots = int(self._rng.integers(3, 9))
        for i in range(n_slots):
            y = int((i + 0.5) * size / n_slots)
            slot_v = min(255, b_base + int(self._rng.integers(3, 12)))
            cv2.line(img.astype(np.uint8), (10, y), (size - 10, y), (slot_v,) * 3, 1)
        return np.clip(img, 0, 255).astype(np.uint8)

    def _light_enclosure(self, size: int) -> np.ndarray:
        # Light beige/gray electronics enclosure
        base_b = int(self._rng.integers(155, 195))
        base_g = int(self._rng.integers(155, 195))
        base_r = int(self._rng.integers(150, 190))
        img = np.zeros((size, size, 3), dtype=np.float32)
        img[:, :, 0] = base_b
        img[:, :, 1] = base_g
        img[:, :, 2] = base_r
        noise = self._rng.normal(0, 5, (size, size)).astype(np.float32)
        for ch in range(3):
            img[:, :, ch] = np.clip(img[:, :, ch] + noise, 0, 255)
        # Subtle panel lines
        if self._rng.random() > 0.5:
            line_v = max(0, base_b - 20)
            midline = size // 2
            cv2.line(img.astype(np.uint8), (0, midline), (size, midline), (line_v,) * 3, 1)
        return np.clip(img, 0, 255).astype(np.uint8)

    # ------------------------------------------------------------------
    # PCB styles
    # ------------------------------------------------------------------

    def _pcb(self, size: int) -> np.ndarray:
        base_g = int(self._rng.integers(20, 55))
        base_b = int(self._rng.integers(15, 40))
        img = np.zeros((size, size, 3), dtype=np.float32)
        img[:, :, 0] = base_b
        img[:, :, 1] = base_g
        noise = self._rng.normal(0, 3, (size, size)).astype(np.float32)
        for ch in range(3):
            img[:, :, ch] = np.clip(img[:, :, ch] + noise, 0, 255)
        self._add_pcb_traces(img, size, trace_color=(0, int(self._rng.integers(60, 100)), 0))
        return np.clip(img, 0, 255).astype(np.uint8)

    def _blue_pcb(self, size: int) -> np.ndarray:
        base_b = int(self._rng.integers(60, 110))
        img = np.zeros((size, size, 3), dtype=np.float32)
        img[:, :, 0] = base_b
        img[:, :, 1] = int(base_b * 0.35)
        noise = self._rng.normal(0, 3, (size, size)).astype(np.float32)
        for ch in range(3):
            img[:, :, ch] = np.clip(img[:, :, ch] + noise, 0, 255)
        self._add_pcb_traces(img, size, trace_color=(int(base_b + 40), int(base_b * 0.5), 0))
        return np.clip(img, 0, 255).astype(np.uint8)

    def _realistic_pcb(self, size: int) -> np.ndarray:
        # Detailed PCB with ICs, resistors, capacitors
        base_g = int(self._rng.integers(18, 48))
        img = np.zeros((size, size, 3), dtype=np.float32)
        img[:, :, 1] = base_g
        noise = self._rng.normal(0, 2.5, (size, size)).astype(np.float32)
        for ch in range(3):
            img[:, :, ch] = np.clip(img[:, :, ch] + noise, 0, 255)
        self._add_pcb_traces(img, size, trace_color=(0, int(base_g + 40), 0))
        # IC packages
        n_ics = int(self._rng.integers(1, 4))
        for _ in range(n_ics):
            ix = int(self._rng.integers(5, size - 30))
            iy = int(self._rng.integers(5, size - 20))
            iw = int(self._rng.integers(15, 35))
            ih = int(self._rng.integers(10, 25))
            cv2.rectangle(img.astype(np.uint8), (ix, iy), (ix + iw, iy + ih), (30, 30, 30), -1)
            cv2.rectangle(img.astype(np.uint8), (ix, iy), (ix + iw, iy + ih), (50, 50, 50), 1)
        # SMD resistors/caps
        n_components = int(self._rng.integers(3, 10))
        for _ in range(n_components):
            cx = int(self._rng.integers(2, size - 10))
            cy = int(self._rng.integers(2, size - 8))
            w_comp = int(self._rng.integers(4, 10))
            h_comp = int(self._rng.integers(2, 5))
            comp_v = int(self._rng.integers(80, 160))
            cv2.rectangle(img.astype(np.uint8), (cx, cy), (cx + w_comp, cy + h_comp),
                          (comp_v, comp_v, comp_v), -1)
        return np.clip(img, 0, 255).astype(np.uint8)

    def _add_pcb_traces(self, img: np.ndarray, size: int, trace_color: tuple[int, int, int]) -> None:
        n_horiz = int(self._rng.integers(3, 8))
        for _ in range(n_horiz):
            y = int(self._rng.integers(0, size))
            x1, x2 = sorted(self._rng.integers(0, size, size=2).tolist())
            cv2.line(img.astype(np.uint8), (x1, y), (x2, y), trace_color, 1)
        n_vert = int(self._rng.integers(3, 8))
        for _ in range(n_vert):
            x = int(self._rng.integers(0, size))
            y1, y2 = sorted(self._rng.integers(0, size, size=2).tolist())
            cv2.line(img.astype(np.uint8), (x, y1), (x, y2), trace_color, 1)
        # Pads
        n_pads = int(self._rng.integers(4, 12))
        for _ in range(n_pads):
            px = int(self._rng.integers(0, size))
            py = int(self._rng.integers(0, size))
            pad_r = int(self._rng.integers(2, 6))
            pad_v = int(self._rng.integers(160, 200))
            cv2.circle(img.astype(np.uint8), (px, py), pad_r, (pad_v, pad_v - 20, 0), -1)

    # ------------------------------------------------------------------
    # Metal styles
    # ------------------------------------------------------------------

    def _metal(self, size: int) -> np.ndarray:
        base_v = int(self._rng.integers(100, 165))
        img = np.full((size, size, 3), base_v, dtype=np.float32)
        # Brushed horizontal texture
        for y in range(size):
            streak = float(self._rng.normal(0, 8))
            img[y, :, :] = np.clip(img[y, :, :] + streak, 0, 255)
        # Specular highlight band
        hl_y = int(size * self._rng.uniform(0.1, 0.4))
        hl_w = int(size * self._rng.uniform(0.05, 0.15))
        hl_band = np.zeros((size, size), dtype=np.float32)
        hl_band[max(0, hl_y - hl_w // 2):hl_y + hl_w // 2, :] = 1.0
        k = max(3, (hl_w | 1))
        hl_band = cv2.GaussianBlur(hl_band, (1, k), hl_w * 0.4)
        for ch in range(3):
            img[:, :, ch] = np.clip(img[:, :, ch] + hl_band * 60, 0, 255)
        return img.astype(np.uint8)

    def _brushed_metal(self, size: int) -> np.ndarray:
        base_v = int(self._rng.integers(80, 150))
        img = np.full((size, size, 3), base_v, dtype=np.float32)
        # Strong directional brushing
        n_streaks = int(self._rng.integers(size // 4, size // 2))
        for _ in range(n_streaks):
            y = int(self._rng.integers(0, size))
            streak_v = float(self._rng.normal(0, 18))
            streak_len = int(self._rng.integers(size // 4, size))
            x_start = int(self._rng.integers(0, size))
            x_end = min(size, x_start + streak_len)
            img[y, x_start:x_end, :] = np.clip(img[y, x_start:x_end, :] + streak_v, 0, 255)
        # Slight blue tint (steel)
        img[:, :, 0] = np.clip(img[:, :, 0] + self._rng.uniform(0, 10), 0, 255)
        img[:, :, 2] = np.clip(img[:, :, 2] - self._rng.uniform(0, 10), 0, 255)
        # Multiple highlight bands
        n_hl = int(self._rng.integers(1, 4))
        for _ in range(n_hl):
            hl_y = int(self._rng.integers(0, size))
            hl_w = max(1, int(size * self._rng.uniform(0.01, 0.05)))
            for dy in range(-hl_w // 2, hl_w // 2 + 1):
                y_pos = hl_y + dy
                if 0 <= y_pos < size:
                    img[y_pos, :, :] = np.clip(img[y_pos, :, :] + 40, 0, 255)
        return img.astype(np.uint8)

    def _aluminum_pcb(self, size: int) -> np.ndarray:
        base_v = int(self._rng.integers(140, 190))
        img = np.full((size, size, 3), base_v, dtype=np.float32)
        # Aluminum substrate (slightly warm)
        img[:, :, 2] = np.clip(img[:, :, 2] - 8, 0, 255)
        noise = self._rng.normal(0, 5, (size, size)).astype(np.float32)
        for ch in range(3):
            img[:, :, ch] = np.clip(img[:, :, ch] + noise, 0, 255)
        # PCB traces on metal
        self._add_pcb_traces(img, size, trace_color=(180, 140, 60))
        # Anodized patches
        n_patches = int(self._rng.integers(1, 4))
        for _ in range(n_patches):
            px = int(self._rng.integers(0, size))
            py = int(self._rng.integers(0, size))
            patch_v = int(self._rng.integers(base_v - 20, base_v + 20))
            cv2.circle(img.astype(np.uint8), (px, py), int(self._rng.integers(4, 12)),
                       (patch_v,) * 3, -1)
        return np.clip(img, 0, 255).astype(np.uint8)

    # ------------------------------------------------------------------
    # Studio / special styles
    # ------------------------------------------------------------------

    def _white_studio(self, size: int) -> np.ndarray:
        base_v = int(self._rng.integers(230, 252))
        img = np.full((size, size, 3), base_v, dtype=np.float32)
        # Vignette
        cx, cy = size / 2, size / 2
        y_idx, x_idx = np.ogrid[:size, :size]
        dist = np.sqrt((x_idx - cx) ** 2 + (y_idx - cy) ** 2) / (size * 0.7)
        vignette = np.clip(dist, 0, 1) ** 2
        for ch in range(3):
            img[:, :, ch] = np.clip(img[:, :, ch] - vignette * self._rng.uniform(20, 60), 0, 255)
        # Subtle shadow beneath subject area
        shadow_y = int(size * self._rng.uniform(0.55, 0.75))
        shadow = np.zeros((size, size), dtype=np.float32)
        for y in range(shadow_y, min(size, shadow_y + size // 5)):
            alpha = (y - shadow_y) / (size // 5)
            shadow[y, :] = alpha * self._rng.uniform(10, 30)
        for ch in range(3):
            img[:, :, ch] = np.clip(img[:, :, ch] - shadow, 0, 255)
        return img.astype(np.uint8)

    def _textured(self, size: int) -> np.ndarray:
        texture_type = self._rng.choice(["concrete", "fabric", "rubber", "foam"])
        if texture_type == "concrete":
            base_v = int(self._rng.integers(90, 160))
            img = np.full((size, size, 3), base_v, dtype=np.float32)
            noise = self._rng.normal(0, 15, (size, size)).astype(np.float32)
            for ch in range(3):
                img[:, :, ch] = np.clip(img[:, :, ch] + noise, 0, 255)
            k = 5
            img = cv2.GaussianBlur(img, (k, k), 1.5)
        elif texture_type == "fabric":
            base_v = int(self._rng.integers(40, 180))
            img = np.full((size, size, 3), base_v, dtype=np.float32)
            for y in range(0, size, 2):
                img[y, :, :] = np.clip(img[y, :, :] + 8, 0, 255)
            for x in range(0, size, 2):
                img[:, x, :] = np.clip(img[:, x, :] + 5, 0, 255)
        elif texture_type == "rubber":
            base_v = int(self._rng.integers(15, 50))
            img = np.full((size, size, 3), base_v, dtype=np.float32)
            noise = self._rng.normal(0, 4, (size, size)).astype(np.float32)
            for ch in range(3):
                img[:, :, ch] = np.clip(img[:, :, ch] + noise, 0, 255)
        else:  # foam
            base_v = int(self._rng.integers(150, 230))
            img = np.full((size, size, 3), base_v, dtype=np.float32)
            noise = self._rng.normal(0, 10, (size, size)).astype(np.float32)
            for ch in range(3):
                img[:, :, ch] = np.clip(img[:, :, ch] + noise, 0, 255)
            k = 9
            img = cv2.GaussianBlur(img, (k, k), 3.0)
        return np.clip(img, 0, 255).astype(np.uint8)

    def _gradient(self, size: int) -> np.ndarray:
        direction = self._rng.choice(["horizontal", "vertical", "diagonal", "radial"])
        c1 = self._rng.integers(0, 256, size=3).astype(np.float32)
        c2 = self._rng.integers(0, 256, size=3).astype(np.float32)
        img = np.zeros((size, size, 3), dtype=np.float32)
        if direction == "horizontal":
            t = np.linspace(0, 1, size, dtype=np.float32)[np.newaxis, :, np.newaxis]
            img[:] = c1 * (1 - t) + c2 * t
        elif direction == "vertical":
            t = np.linspace(0, 1, size, dtype=np.float32)[:, np.newaxis, np.newaxis]
            img[:] = c1 * (1 - t) + c2 * t
        elif direction == "diagonal":
            tx = np.linspace(0, 1, size, dtype=np.float32)[np.newaxis, :, np.newaxis]
            ty = np.linspace(0, 1, size, dtype=np.float32)[:, np.newaxis, np.newaxis]
            t = (tx + ty) / 2
            img[:] = c1 * (1 - t) + c2 * t
        else:  # radial
            cx, cy = size / 2, size / 2
            y_idx, x_idx = np.ogrid[:size, :size]
            dist = np.sqrt((x_idx - cx) ** 2 + (y_idx - cy) ** 2)
            t = np.clip(dist / (size * 0.7), 0, 1)[:, :, np.newaxis]
            img[:] = c1 * (1 - t) + c2 * t
        return np.clip(img, 0, 255).astype(np.uint8)

    def _plain(self, size: int) -> np.ndarray:
        color = self._rng.integers(0, 256, size=3).astype(np.uint8)
        img = np.full((size, size, 3), color, dtype=np.uint8)
        return img
