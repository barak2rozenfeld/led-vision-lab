"""Post-processing effects pipeline — ambient, glare, occlusion, camera, false positives."""
from __future__ import annotations

import numpy as np
import cv2


# ---------------------------------------------------------------------------
# Ambient lighting
# ---------------------------------------------------------------------------

def apply_ambient(
    image: np.ndarray,
    ambient_type: str,
    ambient_factor: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Apply global ambient lighting style to the image."""
    img = image.astype(np.float32)
    h, w = img.shape[:2]

    if ambient_type == "dark_room":
        img *= ambient_factor * 0.28

    elif ambient_type == "low_light":
        img *= ambient_factor * 0.50

    elif ambient_type == "normal_indoor":
        img *= ambient_factor

    elif ambient_type == "bright_office":
        img *= min(1.5, ambient_factor * 1.30)

    elif ambient_type == "warm":
        img[:, :, 2] = np.clip(img[:, :, 2] * min(1.3, ambient_factor * 1.15), 0, 255)
        img[:, :, 0] = np.clip(img[:, :, 0] * max(0.75, ambient_factor * 0.85), 0, 255)
        img[:, :, 1] = np.clip(img[:, :, 1] * ambient_factor, 0, 255)

    elif ambient_type == "cool":
        img[:, :, 0] = np.clip(img[:, :, 0] * min(1.3, ambient_factor * 1.15), 0, 255)
        img[:, :, 2] = np.clip(img[:, :, 2] * max(0.75, ambient_factor * 0.85), 0, 255)
        img[:, :, 1] = np.clip(img[:, :, 1] * ambient_factor, 0, 255)

    elif ambient_type == "directional":
        direction = rng.choice(["left", "right", "top", "bottom"])
        t = np.linspace(0.25, 1.0, w if direction in ("left", "right") else h, dtype=np.float32)
        if direction == "right":
            t = t[::-1]
        if direction in ("left", "right"):
            gradient = t[np.newaxis, :, np.newaxis]
        elif direction == "top":
            gradient = t[:, np.newaxis, np.newaxis]
        else:
            t = t[::-1]
            gradient = t[:, np.newaxis, np.newaxis]
        img = img * gradient * ambient_factor

    elif ambient_type == "backlight":
        cx, cy = w / 2, h / 2
        y_idx, x_idx = np.ogrid[:h, :w]
        dist = np.sqrt((x_idx - cx) ** 2 + (y_idx - cy) ** 2)
        max_dist = np.sqrt(cx ** 2 + cy ** 2)
        backlight = (dist / (max_dist + 1e-6)) * 90.0
        for ch in range(3):
            img[:, :, ch] += backlight
        img *= ambient_factor

    return np.clip(img, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Glare / reflections
# ---------------------------------------------------------------------------

def apply_glare(
    image: np.ndarray,
    glare_type: str,
    led_positions: list[tuple[int, int, int]],
    rng: np.random.Generator,
) -> np.ndarray:
    """Add glare or reflection effects around LED positions."""
    if glare_type == "none" or not led_positions:
        return image

    result = image.copy()

    for cx, cy, r in led_positions:
        if glare_type == "small_highlight":
            hl_r = max(2, r // 2)
            hl_x = cx + int(rng.integers(-r // 2, r // 2 + 1))
            hl_y = cy + int(rng.integers(-r // 2, r // 2 + 1))
            result = _add_bloom(result, hl_x, hl_y, hl_r, (255, 255, 255), 0.8, 0.8)

        elif glare_type == "strong_glare":
            bloom_r = max(4, r * 3)
            result = _add_bloom(result, cx, cy, bloom_r, (255, 255, 255), 0.6, 1.2)

        elif glare_type == "reflection_near":
            angle = rng.uniform(0, 360)
            dist = r * rng.uniform(1.5, 3.0)
            ref_x = int(cx + dist * np.cos(np.radians(angle)))
            ref_y = int(cy + dist * np.sin(np.radians(angle)))
            ref_r = max(2, int(r * rng.uniform(0.3, 0.8)))
            result = _add_bloom(result, ref_x, ref_y, ref_r, (200, 210, 255), 0.7, 0.6)

        elif glare_type == "reflection_overlap":
            angle = rng.uniform(0, 360)
            dist = r * rng.uniform(0.2, 0.9)
            ref_x = int(cx + dist * np.cos(np.radians(angle)))
            ref_y = int(cy + dist * np.sin(np.radians(angle)))
            ref_r = max(2, int(r * rng.uniform(0.5, 1.3)))
            result = _add_bloom(result, ref_x, ref_y, ref_r, (200, 220, 255), 0.6, 0.8)

        elif glare_type == "glossy":
            h_img, w_img = image.shape[:2]
            n_specs = int(rng.integers(2, 7))
            for _ in range(n_specs):
                sx = int(rng.integers(0, w_img))
                sy = int(rng.integers(0, h_img))
                sr = max(2, int(rng.integers(2, max(3, r))))
                result = _add_bloom(result, sx, sy, sr, (255, 255, 255), 0.5, 0.5)

    return result


def _add_bloom(
    image: np.ndarray,
    cx: int, cy: int, radius: int,
    color: tuple[int, int, int],
    sigma_factor: float,
    intensity: float,
) -> np.ndarray:
    glow = np.zeros_like(image, dtype=np.float32)
    cv2.circle(glow, (cx, cy), radius, color, -1)
    sigma = max(1.0, radius * sigma_factor)
    k = max(3, int(sigma * 4) | 1)
    glow = cv2.GaussianBlur(glow, (k, k), sigma)
    return np.clip(image.astype(np.float32) + glow * intensity, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Occlusion
# ---------------------------------------------------------------------------

def apply_occlusion(
    image: np.ndarray,
    occlusion_type: str,
    led_positions: list[tuple[int, int, int]],
    rng: np.random.Generator,
) -> np.ndarray:
    """Add occlusion over or near LED positions."""
    if occlusion_type == "none" or not led_positions:
        return image

    result = image.copy()
    cx, cy, r = led_positions[0]  # target the primary LED
    h, w = image.shape[:2]

    if occlusion_type == "cable":
        angle = rng.uniform(0, 180)
        thickness = max(3, r // 3)
        diag = int(np.sqrt(w ** 2 + h ** 2))
        dx = int(np.cos(np.radians(angle)) * diag)
        dy = int(np.sin(np.radians(angle)) * diag)
        start = (cx - dx, cy - dy)
        end = (cx + dx, cy + dy)
        cable_b = int(rng.integers(10, 55))
        cv2.line(result, start, end, (cable_b,) * 3, thickness)
        # Thin highlight on cable top edge
        highlight_b = min(255, cable_b + 30)
        n = max(1, thickness // 3)
        cv2.line(result, start, end, (highlight_b,) * 3, n)

    elif occlusion_type == "label":
        lx1 = cx - int(r * rng.uniform(0.5, 2.0))
        ly1 = cy - int(r * rng.uniform(0.3, 1.0))
        lx2 = cx + int(r * rng.uniform(0.5, 1.5))
        ly2 = cy + int(r * rng.uniform(0.3, 1.0))
        label_v = int(rng.integers(200, 252))
        overlay = result.copy()
        cv2.rectangle(overlay, (lx1, ly1), (lx2, ly2), (label_v,) * 3, -1)
        # Label border
        cv2.rectangle(overlay, (lx1, ly1), (lx2, ly2), (label_v - 30,) * 3, 1)
        alpha = float(rng.uniform(0.55, 0.90))
        result = cv2.addWeighted(overlay, alpha, result, 1 - alpha, 0)

    elif occlusion_type == "shadow":
        shadow_intensity = float(rng.uniform(0.30, 0.70))
        direction = rng.choice(["top", "bottom", "left", "right"])
        shadow_map = np.zeros((h, w), dtype=np.float32)
        half = (h if direction in ("top", "bottom") else w) // 2
        if direction == "top":
            for y in range(half):
                shadow_map[y, :] = shadow_intensity * (1 - y / half)
        elif direction == "bottom":
            for y in range(half, h):
                shadow_map[y, :] = shadow_intensity * ((y - half) / half)
        elif direction == "left":
            for x in range(half):
                shadow_map[:, x] = shadow_intensity * (1 - x / half)
        else:
            for x in range(half, w):
                shadow_map[:, x] = shadow_intensity * ((x - half) / half)
        result = np.clip(
            result.astype(np.float32) * (1 - shadow_map[:, :, np.newaxis]),
            0, 255,
        ).astype(np.uint8)

    elif occlusion_type == "panel_edge":
        side = rng.choice(["top", "bottom", "left", "right"])
        edge_v = int(rng.integers(25, 90))
        thickness = max(r, int(r * rng.uniform(1.0, 2.2)))
        if side == "top":
            result[:thickness, :] = edge_v
        elif side == "bottom":
            result[h - thickness:, :] = edge_v
        elif side == "left":
            result[:, :thickness] = edge_v
        else:
            result[:, w - thickness:] = edge_v

    return result


# ---------------------------------------------------------------------------
# False positives
# ---------------------------------------------------------------------------

def apply_false_positive(
    image: np.ndarray,
    fp_type: str,
    rng: np.random.Generator,
) -> tuple[np.ndarray, int, int, int]:
    """Draw a false-positive object that looks like an LED (but isn't).

    Returns ``(image, cx, cy, radius)`` for metadata.
    """
    h, w = image.shape[:2]
    margin = 0.12
    cx = int(w * float(rng.uniform(margin, 1 - margin)))
    cy = int(h * float(rng.uniform(margin, 1 - margin)))
    size = max(4, int(min(w, h) * float(rng.uniform(0.03, 0.10))))
    result = image.copy()

    if fp_type == "colored_screw":
        metal_v = int(rng.integers(110, 175))
        cv2.circle(result, (cx, cy), size, (metal_v, metal_v, metal_v), -1)
        cv2.circle(result, (cx, cy), size, (70, 70, 70), 2)
        slot = max(1, size // 4)
        cv2.line(result, (cx - size // 2, cy), (cx + size // 2, cy), (50, 50, 50), slot)
        cv2.line(result, (cx, cy - size // 2), (cx, cy + size // 2), (50, 50, 50), slot)

    elif fp_type == "reflection_spot":
        result = _add_bloom(result, cx, cy, size, (255, 255, 255), 0.8, 0.9)

    elif fp_type == "printed_circle":
        r = int(rng.integers(30, 230))
        g = int(rng.integers(30, 230))
        b = int(rng.integers(30, 230))
        cv2.circle(result, (cx, cy), size, (b, g, r), -1)

    elif fp_type == "colored_button":
        base_b = int(rng.integers(40, 210))
        base_g = int(rng.integers(40, 210))
        base_r = int(rng.integers(40, 210))
        cv2.circle(result, (cx, cy), size, (base_b, base_g, base_r), -1)
        rim_b = max(0, base_b - 40)
        rim_g = max(0, base_g - 40)
        rim_r = max(0, base_r - 40)
        cv2.circle(result, (cx, cy), size, (rim_b, rim_g, rim_r), 2)
        hl_r = max(1, size // 4)
        hl_b = min(255, base_b + 60)
        hl_g = min(255, base_g + 60)
        hl_r_v = min(255, base_r + 60)
        cv2.circle(result, (cx - size // 4, cy - size // 4), hl_r,
                   (hl_b, hl_g, hl_r_v), -1)

    elif fp_type == "display_pixel":
        half = max(2, size // 3)
        r = int(rng.integers(120, 256))
        g = int(rng.integers(120, 256))
        b = int(rng.integers(120, 256))
        cv2.rectangle(result, (cx - half, cy - half), (cx + half, cy + half), (b, g, r), -1)

    elif fp_type == "specular_highlight":
        axes = (size, max(2, size // 3))
        angle = int(rng.integers(0, 180))
        glow = np.zeros_like(result, dtype=np.float32)
        cv2.ellipse(glow, (cx, cy), axes, angle, 0, 360, (255, 255, 255), -1)
        sigma = size * 0.5
        k = max(3, int(sigma * 4) | 1)
        glow = cv2.GaussianBlur(glow, (k, k), sigma)
        result = np.clip(result.astype(np.float32) + glow * 0.75, 0, 255).astype(np.uint8)

    return result, cx, cy, size


# ---------------------------------------------------------------------------
# Camera effects
# ---------------------------------------------------------------------------

def apply_gaussian_blur(image: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return image
    k = max(3, int(sigma * 4) | 1)
    return cv2.GaussianBlur(image, (k, k), sigma)


def apply_motion_blur(image: np.ndarray, angle_deg: float, length: int) -> np.ndarray:
    if length < 2:
        return image
    kernel = np.zeros((length, length), dtype=np.float32)
    center = length // 2
    cv2.line(kernel, (0, center), (length - 1, center), 1.0, 1)
    M = cv2.getRotationMatrix2D((center, center), angle_deg, 1.0)
    kernel = cv2.warpAffine(kernel, M, (length, length))
    total = kernel.sum()
    if total > 0:
        kernel /= total
    return cv2.filter2D(image, -1, kernel)


def apply_noise(image: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    if sigma <= 0:
        return image
    noise = rng.normal(0, sigma, image.shape).astype(np.float32)
    return np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def apply_chromatic_aberration(image: np.ndarray, shift_px: float) -> np.ndarray:
    shift = int(round(shift_px))
    if shift <= 0:
        return image
    h, w = image.shape[:2]
    result = image.copy()
    # Shift red channel right
    result[:, shift:, 2] = image[:, :w - shift, 2]
    result[:, :shift, 2] = image[:, 0:1, 2]
    # Shift blue channel left
    result[:, :w - shift, 0] = image[:, shift:, 0]
    result[:, w - shift:, 0] = image[:, w - 1:w, 0]
    return result


def apply_gamma(image: np.ndarray, gamma: float) -> np.ndarray:
    if abs(gamma - 1.0) < 0.01:
        return image
    inv_gamma = 1.0 / gamma
    table = np.array(
        [int(((i / 255.0) ** inv_gamma) * 255 + 0.5) for i in range(256)],
        dtype=np.uint8,
    )
    return cv2.LUT(image, table)


def apply_white_balance(
    image: np.ndarray, shift: float, rng: np.random.Generator
) -> np.ndarray:
    if shift <= 0:
        return image
    factors = (1.0 + rng.uniform(-shift, shift, size=3)).astype(np.float32)
    return np.clip(image.astype(np.float32) * factors, 0, 255).astype(np.uint8)


def apply_jpeg_artifacts(image: np.ndarray, quality: int) -> np.ndarray:
    quality = max(1, min(95, quality))
    ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        return image
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def apply_rotation(
    image: np.ndarray, angle_deg: float
) -> np.ndarray:
    if abs(angle_deg) < 0.1:
        return image
    h, w = image.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle_deg, 1.0)
    return cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REFLECT)
