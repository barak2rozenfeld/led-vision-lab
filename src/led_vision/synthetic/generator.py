"""DatasetGenerator v2 — master→downsample, 30-field metadata, group quotas."""
from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from tqdm import tqdm

from .config import DatasetConfig, load_config
from .scene import SceneBuilder, SceneResult, _dataclass_weights, _weighted_choice


# Metadata CSV columns (order matters)
_CSV_COLUMNS = [
    "id",
    "group",
    "split",
    "size",
    "state",
    "difficulty",
    "false_positive",
    "false_positive_type",
    "led_count",
    # Primary LED fields (first LED, or FP object)
    "led_state",
    "led_color",
    "led_shape",
    "led_lens_type",
    "led_brightness",
    "led_size_pct",
    "led_pos_x",
    "led_pos_y",
    "led_glow_radius",
    "led_halo_intensity",
    "led_cx_master",
    "led_cy_master",
    "led_radius_master",
    # Scene parameters
    "background",
    "ambient_type",
    "ambient_factor",
    "view_angle",
    "glare_type",
    "occlusion_type",
    # Camera effects
    "blur_sigma",
    "noise_sigma",
    "motion_blur",
    "motion_blur_angle",
    "motion_blur_length",
    "chromatic_shift",
    "gamma",
    "white_balance_shift",
    "jpeg_quality",
    # Provenance
    "master_size",
    "seed",
    "filepath",
]


class DatasetGenerator:
    def __init__(self, config: DatasetConfig) -> None:
        self._cfg = config
        self._rng = np.random.default_rng(config.seed)

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    def generate(self) -> Path:
        """Generate the full dataset. Returns path to metadata CSV."""
        cfg = self._cfg
        out_dir = Path(cfg.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        # Build split assignments
        assignments = self._build_assignments()

        meta_path = out_dir / "metadata.csv"
        start = time.time()

        builder = SceneBuilder(cfg, self._rng)

        with meta_path.open("w", newline="", encoding="utf-8") as csv_f:
            writer = csv.DictWriter(csv_f, fieldnames=_CSV_COLUMNS)
            writer.writeheader()

            for idx, (split, state, difficulty, group) in enumerate(
                tqdm(assignments, desc="Generating scenes", unit="scene")
            ):
                scene_seed = int(self._rng.integers(0, 2**31))
                scene_rng = np.random.default_rng(scene_seed)
                local_builder = SceneBuilder(cfg, scene_rng)

                scene: SceneResult = local_builder.build(
                    target_state=state,
                    difficulty=difficulty,
                    group=group,
                    scene_seed=scene_seed,
                )

                scene_id = f"led_{idx:08d}"
                rows = self._save_scene(scene, scene_id, split, out_dir)
                for row in rows:
                    writer.writerow(row)

        elapsed = time.time() - start
        total_files = cfg.total * len(cfg.export_sizes)
        print(
            f"\nDone — {cfg.total} scenes × {len(cfg.export_sizes)} sizes "
            f"= {total_files} images in {elapsed:.1f}s"
        )
        print(f"Metadata: {meta_path}")

        self._save_config_json(out_dir)
        return meta_path

    # ------------------------------------------------------------------
    # Assignment builder
    # ------------------------------------------------------------------

    def _build_assignments(self) -> list[tuple[str, str, str, str]]:
        """Return list of (split, state, difficulty, group) for each scene."""
        cfg = self._cfg
        n = cfg.total

        # Split indices
        n_train = int(n * cfg.split.train)
        n_val = int(n * cfg.split.val)
        splits = (
            ["train"] * n_train
            + ["val"] * n_val
            + ["test"] * (n - n_train - n_val)
        )

        # Class balance
        on_ratio = self._on_ratio()
        n_on = int(n * on_ratio)
        states = ["on"] * n_on + ["off"] * (n - n_on)

        # Difficulty distribution
        diff_weights = _dataclass_weights(cfg.difficulty)
        difficulties = _sample_categorical(self._rng, diff_weights, n)

        # Group distribution
        group_weights = _dataclass_weights(cfg.groups)
        groups = _sample_categorical(self._rng, group_weights, n)

        self._rng.shuffle(splits)
        self._rng.shuffle(states)

        return list(zip(splits, states, difficulties, groups))

    def _on_ratio(self) -> float:
        balance = self._cfg.class_balance
        if balance == "balanced":
            return 0.50
        if balance == "imbalanced_80_20":
            return 0.80
        if balance == "imbalanced_95_5":
            return 0.95
        if balance == "rare_fault":
            return 0.05  # mostly OFF, ON is the rare "fault"
        return 0.50

    # ------------------------------------------------------------------
    # Scene save
    # ------------------------------------------------------------------

    def _save_scene(
        self,
        scene: SceneResult,
        scene_id: str,
        split: str,
        out_dir: Path,
    ) -> list[dict[str, Any]]:
        """Save master image at each export size and return one CSV row per size."""
        rows: list[dict[str, Any]] = []
        base_row = self._build_base_row(scene, scene_id, split)

        for size in self._cfg.export_sizes:
            img = self._resize(scene.image, size)
            rel_path = self._save_image(img, scene_id, split, scene.state, size, out_dir)
            row = {**base_row, "size": size, "filepath": rel_path}
            rows.append(row)

        return rows

    @staticmethod
    def _resize(image: np.ndarray, target: int) -> np.ndarray:
        h, w = image.shape[:2]
        if h == target and w == target:
            return image
        interp = cv2.INTER_AREA if target < h else cv2.INTER_LINEAR
        return cv2.resize(image, (target, target), interpolation=interp)

    def _save_image(
        self,
        image: np.ndarray,
        scene_id: str,
        split: str,
        state: str,
        size: int,
        out_dir: Path,
    ) -> str:
        folder = out_dir / str(size) / split / state
        folder.mkdir(parents=True, exist_ok=True)
        fname = f"{scene_id}.{self._cfg.image_format}"
        fpath = folder / fname
        ext = self._cfg.image_format.lower()
        if ext in ("jpg", "jpeg"):
            cv2.imwrite(str(fpath), image, [cv2.IMWRITE_JPEG_QUALITY, self._cfg.image_quality])
        else:
            cv2.imwrite(str(fpath), image)
        return str(fpath.relative_to(out_dir))

    # ------------------------------------------------------------------
    # Row builder
    # ------------------------------------------------------------------

    @staticmethod
    def _build_base_row(scene: SceneResult, scene_id: str, split: str) -> dict[str, Any]:
        primary = scene.leds[0] if scene.leds else {}
        return {
            "id": scene_id,
            "group": scene.group,
            "split": split,
            "size": None,          # filled per export size
            "state": scene.state,
            "difficulty": scene.difficulty,
            "false_positive": scene.is_false_positive,
            "false_positive_type": scene.fp_type,
            "led_count": scene.led_count,
            # Primary LED
            "led_state": primary.get("state", ""),
            "led_color": primary.get("color", ""),
            "led_shape": primary.get("shape", ""),
            "led_lens_type": primary.get("lens_type", ""),
            "led_brightness": primary.get("brightness", ""),
            "led_size_pct": primary.get("size_pct", ""),
            "led_pos_x": primary.get("pos_x", ""),
            "led_pos_y": primary.get("pos_y", ""),
            "led_glow_radius": primary.get("glow_radius", ""),
            "led_halo_intensity": primary.get("halo_intensity", ""),
            "led_cx_master": primary.get("cx_master", ""),
            "led_cy_master": primary.get("cy_master", ""),
            "led_radius_master": primary.get("radius_master", ""),
            # Scene
            "background": scene.background,
            "ambient_type": scene.ambient_type,
            "ambient_factor": scene.ambient_factor,
            "view_angle": scene.view_angle,
            "glare_type": scene.glare_type,
            "occlusion_type": scene.occlusion_type,
            # Camera
            "blur_sigma": scene.blur_sigma,
            "noise_sigma": scene.noise_sigma,
            "motion_blur": scene.motion_blur,
            "motion_blur_angle": scene.motion_blur_angle,
            "motion_blur_length": scene.motion_blur_length,
            "chromatic_shift": scene.chromatic_shift,
            "gamma": scene.gamma,
            "white_balance_shift": scene.white_balance_shift,
            "jpeg_quality": scene.jpeg_quality,
            # Provenance
            "master_size": scene.image.shape[0],
            "seed": scene.seed,
            "filepath": None,  # filled per export size
        }

    # ------------------------------------------------------------------
    # Config snapshot
    # ------------------------------------------------------------------

    def _save_config_json(self, out_dir: Path) -> None:
        cfg = self._cfg
        snapshot: dict[str, Any] = {
            "seed": cfg.seed,
            "total": cfg.total,
            "master_size": cfg.master_size,
            "export_sizes": cfg.export_sizes,
            "class_balance": cfg.class_balance,
            "image_format": cfg.image_format,
            "image_quality": cfg.image_quality,
            "split": {
                "train": cfg.split.train,
                "val": cfg.split.val,
                "test": cfg.split.test,
            },
            "backgrounds": _dataclass_weights(cfg.backgrounds),
            "ambient": _dataclass_weights(cfg.ambient),
            "glare": _dataclass_weights(cfg.glare),
            "occlusion": _dataclass_weights(cfg.occlusion),
            "difficulty": _dataclass_weights(cfg.difficulty),
            "groups": _dataclass_weights(cfg.groups),
            "led": {
                "colors": cfg.led.colors,
                "shapes": cfg.led.shapes,
                "lens_types": cfg.led.lens_types,
                "brightness_range": list(cfg.led.brightness_range),
                "size_range": list(cfg.led.size_range),
                "position_margin": cfg.led.position_margin,
                "glow_radius_range": list(cfg.led.glow_radius_range),
                "halo_intensity_range": list(cfg.led.halo_intensity_range),
            },
            "false_positive": {
                "enabled": cfg.false_positive.enabled,
                "rate": cfg.false_positive.rate,
                "types": cfg.false_positive.types,
            },
        }
        path = out_dir / "dataset_config.json"
        path.write_text(json.dumps(snapshot, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sample_categorical(
    rng: np.random.Generator,
    weights: dict[str, float],
    n: int,
) -> list[str]:
    names = list(weights.keys())
    probs = np.array(list(weights.values()), dtype=np.float64)
    probs /= probs.sum()
    indices = rng.choice(len(names), size=n, p=probs)
    return [names[i] for i in indices]
