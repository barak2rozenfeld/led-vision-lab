"""Classical baseline: an LED is ON when a compact region is much brighter than its
local surroundings in HSV space."""
from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

from .base import BoundingBox, LedDetector, Prediction


@dataclass
class HsvConfig:
    work_size: int = 128
    background_kernel_frac: float = 0.25
    v_min: int = 110
    excess_min: float = 35.0
    saturation_weight: float = 0.25
    min_area_frac: float = 0.0015
    halo_weight: float = 0.0
    on_threshold: float = 70.0
    confidence_scale: float = 18.0


def load_hsv_config(path: Path) -> HsvConfig:
    if not path.is_file():
        raise FileNotFoundError(f"HSV config not found: {path}")
    raw: Any = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"HSV config must be a mapping: {path}")
    known = {f.name for f in fields(HsvConfig)}
    unknown = set(raw) - known
    if unknown:
        raise ValueError(f"Unknown HSV config keys in {path}: {sorted(unknown)}")
    return HsvConfig(**raw)


class HSVDetector(LedDetector):
    name = "hsv"

    def __init__(self, config: HsvConfig | None = None) -> None:
        self._cfg = config or HsvConfig()

    @property
    def config(self) -> HsvConfig:
        return self._cfg

    def explain(self, image: np.ndarray) -> dict[str, np.ndarray]:
        """Intermediate maps at work_size: brightness, background estimate, score and mask."""
        return self._maps(image)

    def _maps(self, image: np.ndarray) -> dict[str, np.ndarray]:
        cfg = self._cfg
        small = cv2.resize(image, (cfg.work_size, cfg.work_size), interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        sat = hsv[:, :, 1].astype(np.float32)
        val = hsv[:, :, 2]

        k = max(3, int(cfg.work_size * cfg.background_kernel_frac) | 1)
        background = cv2.medianBlur(val, k).astype(np.float32)
        excess = val.astype(np.float32) - background
        # Coloured LEDs are bright *and* saturated; white ones rely on excess alone.
        score = excess + cfg.saturation_weight * sat * (val.astype(np.float32) / 255.0)

        mask = ((score >= cfg.excess_min) & (val >= cfg.v_min)).astype(np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        return {"value": val, "background": background, "excess": excess, "score": score, "mask": mask}

    def _predict(self, image: np.ndarray) -> Prediction:
        cfg = self._cfg
        maps = self._maps(image)
        excess, score, mask = maps["excess"], maps["score"], maps["mask"]

        n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        total = float(cfg.work_size * cfg.work_size)
        best_label, best_score, best_peak, best_halo, best_area = -1, 0.0, 0.0, 0.0, 0
        for label in range(1, n):
            area = int(stats[label, cv2.CC_STAT_AREA])
            if area / total < cfg.min_area_frac:
                continue
            blob = labels == label
            peak = float(np.percentile(score[blob], 95))
            halo = _halo_excess(blob, excess, area) if cfg.halo_weight else 0.0
            combined = peak + cfg.halo_weight * halo
            if combined > best_score:
                best_label, best_score, best_peak, best_halo, best_area = (
                    label, combined, peak, halo, area,
                )

        margin = best_score - cfg.on_threshold
        on = best_label > 0 and margin >= 0
        confidence = 1.0 / (1.0 + np.exp(-abs(margin) / cfg.confidence_scale))

        bbox = None
        if on:
            x, y, w, h = (int(v) for v in stats[best_label, :4])
            s = float(cfg.work_size)
            bbox = BoundingBox(x / s, y / s, w / s, h / s)

        return Prediction(
            state="ON" if on else "OFF",
            confidence=float(confidence),
            bbox=bbox,
            debug={
                "score": round(best_score, 2),
                "peak_score": round(best_peak, 2),
                "halo_excess": round(best_halo, 2),
                "threshold": cfg.on_threshold,
                "blob_area_frac": round(best_area / total, 5),
                "candidate_blobs": int(n - 1),
            },
        )


def _halo_excess(blob: np.ndarray, excess: np.ndarray, area: int) -> float:
    """Mean brightness above background in a ring around the blob.

    A lit LED illuminates its surroundings; a shiny pad or a grey lens does not.
    Light only adds brightness, so dark neighbours (an IC, the LED housing) count
    as zero rather than cancelling the glow.
    """
    radius = max(2, int(np.sqrt(area / np.pi)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    ring = cv2.dilate(blob.astype(np.uint8), kernel).astype(bool) & ~blob
    if not ring.any():
        return 0.0
    return float(np.clip(excess[ring], 0, None).mean())
