"""Procedural background generators for synthetic LED images."""
from __future__ import annotations

import numpy as np
import cv2

# PCB colors in BGR
_PCB_BASE = (25, 80, 20)
_PCB_COPPER = (30, 110, 185)
_PCB_SILK = (220, 225, 220)
_PCB_HOLE = (8, 15, 8)

BACKGROUND_STYLES = ("plain", "gradient", "pcb", "metal", "noisy")


class BackgroundGenerator:
    """Generates synthetic background images in various styles."""

    def __init__(self, rng: np.random.Generator) -> None:
        self.rng = rng

    def generate(self, style: str, size: int) -> np.ndarray:
        """Return a (size, size, 3) uint8 BGR image for the given style."""
        generators = {
            "plain": self._plain,
            "gradient": self._gradient,
            "pcb": self._pcb,
            "metal": self._metal,
            "noisy": self._noisy,
        }
        if style not in generators:
            raise ValueError(f"Unknown background style: {style!r}. Choose from {BACKGROUND_STYLES}")
        return generators[style](size)

    # ------------------------------------------------------------------
    # Individual style implementations
    # ------------------------------------------------------------------

    def _plain(self, size: int) -> np.ndarray:
        mode = self.rng.choice(["dark", "mid", "light"])
        if mode == "dark":
            base = int(self.rng.integers(5, 55))
        elif mode == "mid":
            base = int(self.rng.integers(80, 160))
        else:
            base = int(self.rng.integers(180, 240))

        tint = self.rng.integers(-18, 19, size=3)
        color = np.clip([base + tint[0], base + tint[1], base + tint[2]], 0, 255).astype(np.uint8)
        return np.full((size, size, 3), color, dtype=np.uint8)

    def _gradient(self, size: int) -> np.ndarray:
        c1 = self.rng.integers(10, 200, size=3).astype(np.float32)
        c2 = self.rng.integers(10, 200, size=3).astype(np.float32)

        direction = self.rng.choice(["horizontal", "vertical", "diagonal"])
        if direction == "horizontal":
            t = np.linspace(0, 1, size, dtype=np.float32)[np.newaxis, :, np.newaxis]
        elif direction == "vertical":
            t = np.linspace(0, 1, size, dtype=np.float32)[:, np.newaxis, np.newaxis]
        else:
            tx = np.linspace(0, 1, size, dtype=np.float32)
            ty = np.linspace(0, 1, size, dtype=np.float32)
            t = ((tx[np.newaxis, :] + ty[:, np.newaxis]) / 2)[:, :, np.newaxis]

        image = np.broadcast_to((1 - t) * c1 + t * c2, (size, size, 3)).copy()
        return np.clip(image, 0, 255).astype(np.uint8)

    def _pcb(self, size: int) -> np.ndarray:
        image = np.full((size, size, 3), _PCB_BASE, dtype=np.uint8)

        # Subtle surface texture
        noise = self.rng.integers(-8, 9, (size, size, 3), dtype=np.int16)
        image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        # Copper traces
        n_traces = int(self.rng.integers(4, 20))
        for _ in range(n_traces):
            horizontal = bool(self.rng.random() > 0.5)
            thickness = max(1, int(size * self.rng.uniform(0.003, 0.015)))
            jitter = self.rng.integers(-20, 21, size=3)
            color = tuple(int(np.clip(_PCB_COPPER[i] + jitter[i], 0, 255)) for i in range(3))
            if horizontal:
                y = int(self.rng.integers(0, size))
                cv2.line(image, (0, y), (size, y), color, thickness)
            else:
                x = int(self.rng.integers(0, size))
                cv2.line(image, (x, 0), (x, size), color, thickness)

        # Solder pads with through-holes
        n_pads = int(self.rng.integers(2, 14))
        for _ in range(n_pads):
            cx = int(self.rng.integers(0, size))
            cy = int(self.rng.integers(0, size))
            pad_r = max(2, int(size * self.rng.uniform(0.01, 0.04)))
            jitter = self.rng.integers(-15, 16, size=3)
            pad_color = tuple(int(np.clip(_PCB_COPPER[i] + jitter[i], 0, 255)) for i in range(3))
            cv2.circle(image, (cx, cy), pad_r, pad_color, -1)
            cv2.circle(image, (cx, cy), max(1, pad_r // 2), _PCB_HOLE, -1)

        # Silkscreen component outlines
        n_components = int(self.rng.integers(1, 7))
        outline_thickness = max(1, size // 200)
        for _ in range(n_components):
            x = int(self.rng.integers(0, max(1, size - size // 4)))
            y = int(self.rng.integers(0, max(1, size - size // 4)))
            w = int(self.rng.integers(size // 12, size // 3))
            h = int(self.rng.integers(size // 12, size // 3))
            cv2.rectangle(image, (x, y), (x + w, y + h), _PCB_SILK, outline_thickness)

        return image

    def _metal(self, size: int) -> np.ndarray:
        base = int(self.rng.integers(130, 195))
        image = np.empty((size, size, 3), dtype=np.uint8)

        # Brushed horizontal lines — slight per-row brightness variation
        for y in range(size):
            val = int(np.clip(base + self.rng.integers(-12, 13), 0, 255))
            image[y, :] = (val, val, val)

        # Vertical sheen gradient
        sheen = np.linspace(0.88, 1.12, size, dtype=np.float32)
        for x in range(size):
            image[:, x] = np.clip(image[:, x].astype(np.float32) * sheen[x], 0, 255).astype(np.uint8)

        # Specular highlight band
        hl_x = int(self.rng.integers(size // 4, 3 * size // 4))
        hl_w = int(size * self.rng.uniform(0.05, 0.18))
        for dx in range(-hl_w, hl_w + 1):
            x = hl_x + dx
            if 0 <= x < size:
                alpha = 1.0 - abs(dx) / max(1, hl_w)
                image[:, x] = np.clip(image[:, x].astype(np.float32) + alpha * 28, 0, 255).astype(np.uint8)

        return image

    def _noisy(self, size: int) -> np.ndarray:
        base = self.rng.integers(20, 120, size=3).astype(np.float32)
        image = np.full((size, size, 3), base, dtype=np.float32)

        # Multi-scale Perlin-like noise via upsampled random layers
        for scale in [1, 2, 4, 8]:
            s = max(1, size // scale)
            layer = self.rng.random((s, s, 3)).astype(np.float32)
            if s != size:
                layer = cv2.resize(layer, (size, size), interpolation=cv2.INTER_LINEAR)
            image += layer * (40.0 / scale)

        return np.clip(image, 0, 255).astype(np.uint8)
