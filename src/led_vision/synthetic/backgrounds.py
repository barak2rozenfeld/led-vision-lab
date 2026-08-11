"""Procedural background generators for synthetic LED images."""
from __future__ import annotations

import numpy as np
import cv2

# PCB colors (BGR)
_PCB_BASE     = (25,  80,  20)
_PCB_COPPER   = (30,  110, 185)
_PCB_SILK     = (220, 225, 220)
_PCB_HOLE     = (8,   15,  8)
_BLUE_PCB_BASE = (90, 55, 25)   # blue PCB board color
_ALU_BASE     = (215, 215, 210)  # aluminum substrate

BACKGROUND_STYLES = (
    "plain", "gradient", "pcb", "blue_pcb", "realistic_pcb",
    "metal", "noisy", "white_studio", "aluminum_pcb",
)


class BackgroundGenerator:
    """Generates synthetic background images in various styles."""

    def __init__(self, rng: np.random.Generator) -> None:
        self.rng = rng

    def generate(self, style: str, size: int) -> np.ndarray:
        """Return a (size, size, 3) uint8 BGR image for the given style."""
        generators = {
            "plain":        self._plain,
            "gradient":     self._gradient,
            "pcb":          self._pcb,
            "blue_pcb":     self._blue_pcb,
            "realistic_pcb":self._realistic_pcb,
            "metal":        self._metal,
            "noisy":        self._noisy,
            "white_studio": self._white_studio,
            "aluminum_pcb": self._aluminum_pcb,
        }
        if style not in generators:
            raise ValueError(
                f"Unknown background style: {style!r}. Choose from {BACKGROUND_STYLES}"
            )
        return generators[style](size)

    # ------------------------------------------------------------------
    # Plain
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

    # ------------------------------------------------------------------
    # Gradient
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # PCB (green, basic)
    # ------------------------------------------------------------------

    def _pcb(self, size: int) -> np.ndarray:
        return self._draw_pcb_board(size, _PCB_BASE)

    # ------------------------------------------------------------------
    # Blue PCB (hobby/Arduino style)
    # ------------------------------------------------------------------

    def _blue_pcb(self, size: int) -> np.ndarray:
        return self._draw_pcb_board(size, _BLUE_PCB_BASE)

    # ------------------------------------------------------------------
    # Shared PCB board drawing logic
    # ------------------------------------------------------------------

    def _draw_pcb_board(self, size: int, base_color: tuple) -> np.ndarray:
        image = np.full((size, size, 3), base_color, dtype=np.uint8)

        # Subtle surface noise
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

        # Solder pads + through-holes
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
        outline_t = max(1, size // 200)
        for _ in range(n_components):
            x = int(self.rng.integers(0, max(1, size - size // 4)))
            y = int(self.rng.integers(0, max(1, size - size // 4)))
            w = int(self.rng.integers(size // 12, size // 3))
            h = int(self.rng.integers(size // 12, size // 3))
            cv2.rectangle(image, (x, y), (x + w, y + h), _PCB_SILK, outline_t)

        return image

    # ------------------------------------------------------------------
    # Realistic PCB — detailed board with ICs, resistors, capacitors, labels
    # ------------------------------------------------------------------

    def _realistic_pcb(self, size: int) -> np.ndarray:
        base_color = _PCB_BASE if bool(self.rng.random() > 0.4) else _BLUE_PCB_BASE
        image = np.full((size, size, 3), base_color, dtype=np.uint8)

        noise = self.rng.integers(-6, 7, (size, size, 3), dtype=np.int16)
        image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        # Routed copper traces (L-shaped routing)
        n_routes = int(self.rng.integers(5, 16))
        for _ in range(n_routes):
            x1 = int(self.rng.integers(0, size))
            y1 = int(self.rng.integers(0, size))
            x_mid = int(self.rng.integers(0, size))
            y2 = int(self.rng.integers(0, size))
            t = max(1, int(size * self.rng.uniform(0.003, 0.012)))
            jitter = self.rng.integers(-15, 16, size=3)
            c = tuple(int(np.clip(_PCB_COPPER[i] + jitter[i], 0, 255)) for i in range(3))
            cv2.line(image, (x1, y1), (x_mid, y1), c, t)
            cv2.line(image, (x_mid, y1), (x_mid, y2), c, t)

        # IC packages (DIP or SOIC style rectangles with legs)
        n_ics = int(self.rng.integers(0, 4))
        for _ in range(n_ics):
            ic_w = int(self.rng.integers(size // 8, size // 3))
            ic_h = int(self.rng.integers(size // 12, size // 6))
            x = int(self.rng.integers(0, max(1, size - ic_w)))
            y = int(self.rng.integers(0, max(1, size - ic_h)))
            # IC body (dark gray)
            cv2.rectangle(image, (x, y), (x + ic_w, y + ic_h), (40, 40, 40), -1)
            cv2.rectangle(image, (x, y), (x + ic_w, y + ic_h), _PCB_SILK, max(1, size // 200))
            # Pin notch
            notch_cx = x + ic_w // 2
            cv2.circle(image, (notch_cx, y), max(2, ic_h // 8), (55, 55, 55), -1)
            # Pin legs on top and bottom
            n_pins = max(2, ic_w // max(1, size // 20))
            pin_spacing = max(2, ic_w // max(1, n_pins))
            for p in range(n_pins):
                px = x + p * pin_spacing + pin_spacing // 2
                pad_r = max(1, size // 80)
                jitter = self.rng.integers(-10, 11, size=3)
                pad_c = tuple(int(np.clip(_PCB_COPPER[i] + jitter[i], 0, 255)) for i in range(3))
                cv2.circle(image, (px, y - 3), pad_r, pad_c, -1)
                cv2.circle(image, (px, y + ic_h + 3), pad_r, pad_c, -1)

        # Resistors (small colored rectangles with silk bands)
        n_res = int(self.rng.integers(2, 10))
        for _ in range(n_res):
            r_w = int(self.rng.integers(size // 18, size // 8))
            r_h = max(3, size // 28)
            rx = int(self.rng.integers(0, max(1, size - r_w)))
            ry = int(self.rng.integers(0, max(1, size - r_h)))
            rotate = bool(self.rng.random() > 0.5)
            if rotate:
                r_w, r_h = r_h, r_w
            cv2.rectangle(image, (rx, ry), (rx + r_w, ry + r_h), (160, 155, 140), -1)
            # Pads on both ends
            for side in [0, 1]:
                px = rx + side * r_w
                py = ry + r_h // 2
                pad_r = max(1, r_h // 2)
                cv2.circle(image, (px, py), pad_r, _PCB_COPPER, -1)

        # Capacitors (small circles/cylinders)
        n_caps = int(self.rng.integers(1, 6))
        for _ in range(n_res):  # noqa (intentionally using n_res loop count)
            pass
        for _ in range(n_caps):
            cap_r = max(3, int(size * self.rng.uniform(0.015, 0.04)))
            cx_c = int(self.rng.integers(cap_r, size - cap_r))
            cy_c = int(self.rng.integers(cap_r, size - cap_r))
            cap_color_top = (50, 55, 100) if bool(self.rng.random() > 0.5) else (30, 100, 30)
            cv2.circle(image, (cx_c, cy_c), cap_r, cap_color_top, -1)
            cv2.circle(image, (cx_c, cy_c), cap_r, _PCB_SILK, max(1, cap_r // 5))
            # Pad underneath
            cv2.circle(image, (cx_c, cy_c), max(1, cap_r + 2), _PCB_COPPER, max(1, size // 100))

        # Component labels (silkscreen text)
        n_labels = int(self.rng.integers(3, 10))
        label_prefixes = ["D", "R", "C", "U", "Q", "L", "J"]
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = max(0.2, size / 800)
        font_t = max(1, size // 300)
        for i in range(n_labels):
            prefix = label_prefixes[int(self.rng.integers(0, len(label_prefixes)))]
            num = int(self.rng.integers(1, 12))
            label = f"{prefix}{num}"
            lx = int(self.rng.integers(0, max(1, size - size // 5)))
            ly = int(self.rng.integers(size // 20, size - size // 20))
            cv2.putText(image, label, (lx, ly), font, font_scale, _PCB_SILK, font_t, cv2.LINE_AA)

        # Through-hole LED mounting pads
        n_led_pads = int(self.rng.integers(1, 5))
        for _ in range(n_led_pads):
            lx = int(self.rng.integers(0, size))
            ly = int(self.rng.integers(0, size))
            r = max(3, int(size * self.rng.uniform(0.012, 0.035)))
            cv2.circle(image, (lx, ly), r + 2, _PCB_COPPER, -1)
            cv2.circle(image, (lx, ly), r, _PCB_HOLE, -1)

        return image

    # ------------------------------------------------------------------
    # Metal panel (brushed aluminum)
    # ------------------------------------------------------------------

    def _metal(self, size: int) -> np.ndarray:
        base = int(self.rng.integers(130, 195))
        image = np.empty((size, size, 3), dtype=np.uint8)
        for y in range(size):
            val = int(np.clip(base + self.rng.integers(-12, 13), 0, 255))
            image[y, :] = (val, val, val)
        sheen = np.linspace(0.88, 1.12, size, dtype=np.float32)
        for x in range(size):
            image[:, x] = np.clip(image[:, x].astype(np.float32) * sheen[x], 0, 255).astype(np.uint8)
        hl_x = int(self.rng.integers(size // 4, 3 * size // 4))
        hl_w = int(size * self.rng.uniform(0.05, 0.18))
        for dx in range(-hl_w, hl_w + 1):
            x = hl_x + dx
            if 0 <= x < size:
                alpha = 1.0 - abs(dx) / max(1, hl_w)
                image[:, x] = np.clip(image[:, x].astype(np.float32) + alpha * 28, 0, 255).astype(np.uint8)
        return image

    # ------------------------------------------------------------------
    # Noisy texture (multi-scale)
    # ------------------------------------------------------------------

    def _noisy(self, size: int) -> np.ndarray:
        base = self.rng.integers(20, 120, size=3).astype(np.float32)
        image = np.full((size, size, 3), base, dtype=np.float32)
        for scale in [1, 2, 4, 8]:
            s = max(1, size // scale)
            layer = self.rng.random((s, s, 3)).astype(np.float32)
            if s != size:
                layer = cv2.resize(layer, (size, size), interpolation=cv2.INTER_LINEAR)
            image += layer * (40.0 / scale)
        return np.clip(image, 0, 255).astype(np.uint8)

    # ------------------------------------------------------------------
    # White studio — clean light background for isolated component shots
    # ------------------------------------------------------------------

    def _white_studio(self, size: int) -> np.ndarray:
        # Base: very light gray to white
        base_val = int(self.rng.integers(230, 252))
        image = np.full((size, size, 3), (base_val, base_val, base_val), dtype=np.float32)

        # Very slight warm or cool tint
        tint_b = float(self.rng.uniform(-8, 8))
        tint_g = float(self.rng.uniform(-5, 5))
        tint_r = float(self.rng.uniform(-8, 8))
        image[:, :, 0] = np.clip(image[:, :, 0] + tint_b, 0, 255)
        image[:, :, 1] = np.clip(image[:, :, 1] + tint_g, 0, 255)
        image[:, :, 2] = np.clip(image[:, :, 2] + tint_r, 0, 255)

        # Subtle vignette (slightly darker corners)
        cx, cy = size / 2, size / 2
        max_dist = np.sqrt(cx ** 2 + cy ** 2)
        y_idx, x_idx = np.ogrid[:size, :size]
        dist = np.sqrt((x_idx - cx) ** 2 + (y_idx - cy) ** 2)
        vignette = 1.0 - 0.12 * (dist / max_dist) ** 2
        for ch in range(3):
            image[:, :, ch] *= vignette

        # Faint soft shadow under component area (central slight darkening)
        shadow_r = size // 3
        shadow = np.zeros((size, size), dtype=np.float32)
        cv2.circle(shadow, (size // 2, size // 2 + size // 8), shadow_r, 1.0, -1)
        shadow = cv2.GaussianBlur(shadow, (0, 0), shadow_r * 0.4)
        image[:, :, 0] -= shadow * 8
        image[:, :, 1] -= shadow * 8
        image[:, :, 2] -= shadow * 8

        return np.clip(image, 0, 255).astype(np.uint8)

    # ------------------------------------------------------------------
    # Aluminum PCB — white/silver SMD panel (like OSRAM/LED strip boards)
    # ------------------------------------------------------------------

    def _aluminum_pcb(self, size: int) -> np.ndarray:
        # White-ish aluminum substrate
        base = int(self.rng.integers(200, 230))
        image = np.full((size, size, 3), (base, base, base - 5), dtype=np.uint8)

        # Subtle texture lines (rolled aluminum grain)
        for y in range(0, size, max(1, size // 60)):
            val = int(np.clip(base + self.rng.integers(-4, 5), 0, 255))
            thickness = max(1, size // 300)
            cv2.line(image, (0, y), (size, y), (val, val, val - 3), thickness)

        # Board outline
        margin = max(2, size // 30)
        cv2.rectangle(image, (margin, margin), (size - margin, size - margin), (140, 142, 142), max(1, size // 150))

        # Fine copper traces
        n_traces = int(self.rng.integers(3, 12))
        trace_color = (140, 155, 180)  # light copper on aluminum
        for _ in range(n_traces):
            x1 = int(self.rng.integers(margin, size - margin))
            y1 = int(self.rng.integers(margin, size - margin))
            x2 = int(self.rng.integers(margin, size - margin))
            y2 = int(self.rng.integers(margin, size - margin))
            t = max(1, int(size * self.rng.uniform(0.002, 0.008)))
            cv2.line(image, (x1, y1), (x2, y2), trace_color, t)

        # SMD LED pads (grid pattern)
        n_cols = int(self.rng.integers(2, 5))
        n_rows = int(self.rng.integers(1, 4))
        cell_w = (size - 2 * margin) // max(1, n_cols)
        cell_h = (size - 2 * margin) // max(1, n_rows)

        for row in range(n_rows):
            for col in range(n_cols):
                px = margin + col * cell_w + cell_w // 2
                py = margin + row * cell_h + cell_h // 2
                pad_w = max(3, cell_w // 4)
                pad_h = max(2, cell_h // 6)
                pad_color = (160, 165, 165)
                # Left pad
                cv2.rectangle(image, (px - pad_w - pad_w, py - pad_h), (px - pad_w, py + pad_h), pad_color, -1)
                # Right pad
                cv2.rectangle(image, (px + pad_w, py - pad_h), (px + pad_w + pad_w, py + pad_h), pad_color, -1)
                # Center die area (square)
                die_r = max(2, min(pad_w, pad_h))
                cv2.rectangle(image, (px - die_r, py - die_r), (px + die_r, py + die_r), (190, 192, 190), -1)

        # + / - polarity labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = max(0.2, size / 1000)
        cv2.putText(image, "+", (margin + 2, size - margin - 4), font, font_scale, (80, 80, 80), 1, cv2.LINE_AA)
        cv2.putText(image, "-", (size - margin - 12, size - margin - 4), font, font_scale, (80, 80, 80), 1, cv2.LINE_AA)

        return image
