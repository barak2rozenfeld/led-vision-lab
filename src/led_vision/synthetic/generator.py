"""DatasetGenerator v2 — master→downsample, 30-field metadata, parallel workers."""
from __future__ import annotations

import csv
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from tqdm import tqdm

from .config import DatasetConfig, load_config
from .scene import SceneBuilder, SceneResult, _dataclass_weights, _weighted_choice


# Metadata CSV columns (order matters)
_CSV_COLUMNS = [
    "id", "group", "split", "size", "state", "difficulty",
    "false_positive", "false_positive_type", "led_count",
    # Primary LED fields
    "led_state", "led_color", "led_shape", "led_lens_type",
    "led_brightness", "led_size_pct", "led_pos_x", "led_pos_y",
    "led_glow_radius", "led_halo_intensity",
    "led_cx_master", "led_cy_master", "led_radius_master",
    # Scene
    "background", "ambient_type", "ambient_factor", "view_angle",
    "glare_type", "occlusion_type",
    # Camera
    "blur_sigma", "noise_sigma", "motion_blur", "motion_blur_angle",
    "motion_blur_length", "chromatic_shift", "gamma", "white_balance_shift",
    "jpeg_quality",
    # Provenance
    "master_size", "seed", "filepath",
]


# ---------------------------------------------------------------------------
# Top-level worker function (must be at module scope for Windows pickling)
# ---------------------------------------------------------------------------

def _scene_worker(args: tuple) -> list[dict[str, Any]]:
    """Generate one scene and save all export sizes. Returns CSV rows."""
    import cv2 as _cv2
    # Prevent OpenCV's internal TBB/OpenMP thread pool from spawning extra
    # threads inside worker processes — avoids deadlocks on Windows (spawn).
    _cv2.setNumThreads(1)

    (
        cfg_path_or_dict,
        idx, split, state, difficulty, group, scene_seed,
        out_dir_str,
    ) = args

    # Reconstruct config inside the worker process
    if isinstance(cfg_path_or_dict, dict):
        cfg = _cfg_from_dict(cfg_path_or_dict)
    else:
        cfg = load_config(Path(cfg_path_or_dict))

    out_dir = Path(out_dir_str)
    rng = np.random.default_rng(scene_seed)
    builder = SceneBuilder(cfg, rng)
    scene: SceneResult = builder.build(
        target_state=state,
        difficulty=difficulty,
        group=group,
        scene_seed=scene_seed,
    )

    scene_id = f"led_{idx:08d}"
    rows: list[dict[str, Any]] = []
    base_row = _build_base_row(scene, scene_id, split)

    for size in cfg.export_sizes:
        img = _resize(scene.image, size)
        rel_path = _save_image(img, scene_id, split, scene.state, size, out_dir, cfg)
        rows.append({**base_row, "size": size, "filepath": rel_path})

    return rows


# ---------------------------------------------------------------------------
# Main generator class
# ---------------------------------------------------------------------------

class DatasetGenerator:
    def __init__(self, config: DatasetConfig) -> None:
        self._cfg = config
        self._rng = np.random.default_rng(config.seed)

    def generate(self) -> Path:
        cfg = self._cfg
        out_dir = Path(cfg.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        assignments = self._build_assignments()
        meta_path = out_dir / "metadata.csv"
        start = time.time()

        # Serialize config to dict for pickling across process boundaries
        cfg_dict = _cfg_to_dict(cfg)

        worker_args = [
            (
                cfg_dict,
                idx, split, state, difficulty, group,
                int(self._rng.integers(0, 2**31)),
                str(out_dir),
            )
            for idx, (split, state, difficulty, group) in enumerate(assignments)
        ]

        n_workers = max(1, cfg.n_workers)

        with meta_path.open("w", newline="", encoding="utf-8") as csv_f:
            writer = csv.DictWriter(csv_f, fieldnames=_CSV_COLUMNS)
            writer.writeheader()

            if n_workers == 1:
                # Single-threaded path (simpler, no pickling overhead)
                for args in tqdm(worker_args, desc="Generating scenes", unit="scene"):
                    for row in _scene_worker(args):
                        writer.writerow(row)
                    csv_f.flush()
            else:
                # Parallel path
                self._run_parallel(worker_args, writer, csv_f, n_workers)

        elapsed = time.time() - start
        total_files = cfg.total * len(cfg.export_sizes)
        print(
            f"\nDone — {cfg.total} scenes × {len(cfg.export_sizes)} sizes"
            f" = {total_files:,} images in {elapsed:.1f}s "
            f"({elapsed / max(1, cfg.total):.2f}s/scene)"
        )
        print(f"Metadata : {meta_path}")
        self._save_config_json(out_dir)
        return meta_path

    def _run_parallel(
        self,
        worker_args: list[tuple],
        writer: csv.DictWriter,
        csv_f: Any,
        n_workers: int,
    ) -> None:
        total = len(worker_args)
        # chunksize=1: each scene is submitted individually so the progress bar
        # updates after every scene. IPC overhead is negligible vs per-scene cost.
        chunksize = 1

        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            with tqdm(total=total, desc="Generating scenes", unit="scene") as bar:
                for rows in pool.map(_scene_worker, worker_args, chunksize=chunksize):
                    for row in rows:
                        writer.writerow(row)
                    csv_f.flush()
                    bar.update(1)

    # ------------------------------------------------------------------
    # Assignment builder
    # ------------------------------------------------------------------

    def _build_assignments(self) -> list[tuple[str, str, str, str]]:
        cfg = self._cfg
        n = cfg.total

        n_train = int(n * cfg.split.train)
        n_val = int(n * cfg.split.val)
        splits = (
            ["train"] * n_train
            + ["val"] * n_val
            + ["test"] * (n - n_train - n_val)
        )

        on_ratio = _on_ratio(cfg.class_balance)
        n_on = int(n * on_ratio)
        states = ["on"] * n_on + ["off"] * (n - n_on)

        diff_weights = _dataclass_weights(cfg.difficulty)
        difficulties = _sample_categorical(self._rng, diff_weights, n)

        group_weights = _dataclass_weights(cfg.groups)
        groups = _sample_categorical(self._rng, group_weights, n)

        self._rng.shuffle(splits)
        self._rng.shuffle(states)

        return list(zip(splits, states, difficulties, groups))

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
            "n_workers": cfg.n_workers,
            "image_format": cfg.image_format,
            "image_quality": cfg.image_quality,
            "split": {"train": cfg.split.train, "val": cfg.split.val, "test": cfg.split.test},
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
# Helpers shared between main process and workers
# ---------------------------------------------------------------------------

def _resize(image: np.ndarray, target: int) -> np.ndarray:
    h, w = image.shape[:2]
    if h == target and w == target:
        return image
    interp = cv2.INTER_AREA if target < h else cv2.INTER_LINEAR
    return cv2.resize(image, (target, target), interpolation=interp)


def _save_image(
    image: np.ndarray,
    scene_id: str,
    split: str,
    state: str,
    size: int,
    out_dir: Path,
    cfg: DatasetConfig,
) -> str:
    folder = out_dir / str(size) / split / state
    folder.mkdir(parents=True, exist_ok=True)
    fname = f"{scene_id}.{cfg.image_format}"
    fpath = folder / fname
    ext = cfg.image_format.lower()
    if ext in ("jpg", "jpeg"):
        cv2.imwrite(str(fpath), image, [cv2.IMWRITE_JPEG_QUALITY, cfg.image_quality])
    else:
        cv2.imwrite(str(fpath), image)
    return str(fpath.relative_to(out_dir))


def _build_base_row(scene: SceneResult, scene_id: str, split: str) -> dict[str, Any]:
    primary = scene.leds[0] if scene.leds else {}
    return {
        "id": scene_id,
        "group": scene.group,
        "split": split,
        "size": None,
        "state": scene.state,
        "difficulty": scene.difficulty,
        "false_positive": scene.is_false_positive,
        "false_positive_type": scene.fp_type,
        "led_count": scene.led_count,
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
        "background": scene.background,
        "ambient_type": scene.ambient_type,
        "ambient_factor": scene.ambient_factor,
        "view_angle": scene.view_angle,
        "glare_type": scene.glare_type,
        "occlusion_type": scene.occlusion_type,
        "blur_sigma": scene.blur_sigma,
        "noise_sigma": scene.noise_sigma,
        "motion_blur": scene.motion_blur,
        "motion_blur_angle": scene.motion_blur_angle,
        "motion_blur_length": scene.motion_blur_length,
        "chromatic_shift": scene.chromatic_shift,
        "gamma": scene.gamma,
        "white_balance_shift": scene.white_balance_shift,
        "jpeg_quality": scene.jpeg_quality,
        "master_size": scene.image.shape[0],
        "seed": scene.seed,
        "filepath": None,
    }


def _on_ratio(balance: str) -> float:
    return {"balanced": 0.50, "imbalanced_80_20": 0.80,
            "imbalanced_95_5": 0.95, "rare_fault": 0.05}.get(balance, 0.50)


def _sample_categorical(
    rng: np.random.Generator, weights: dict[str, float], n: int
) -> list[str]:
    names = list(weights.keys())
    probs = np.array(list(weights.values()), dtype=np.float64)
    probs /= probs.sum()
    indices = rng.choice(len(names), size=n, p=probs)
    return [names[i] for i in indices]


# ---------------------------------------------------------------------------
# Config serialisation helpers (needed for cross-process pickling)
# ---------------------------------------------------------------------------

def _cfg_to_dict(cfg: DatasetConfig) -> dict[str, Any]:
    """Flatten DatasetConfig to a plain dict safe for cross-process passing."""
    from dataclasses import fields as dc_fields

    def _val(v: Any) -> Any:
        if isinstance(v, Path):
            return str(v)
        if hasattr(v, "__dataclass_fields__"):
            return {f.name: _val(getattr(v, f.name)) for f in dc_fields(v)}
        if isinstance(v, (list, tuple)):
            return [_val(x) for x in v]
        return v

    return {f.name: _val(getattr(cfg, f.name)) for f in dc_fields(cfg)}


def _cfg_from_dict(d: dict[str, Any]) -> DatasetConfig:
    """Reconstruct DatasetConfig from the dict produced by _cfg_to_dict."""
    from .config import (
        AmbientWeights, BackgroundWeights, CameraConfig, DatasetConfig,
        DifficultyWeights, FalsePositiveConfig, GlareWeights, GroupWeights,
        LedConfig, MultiLedConfig, OcclusionWeights, SplitRatios,
    )

    def pair(v: Any) -> tuple[float, float]:
        return (float(v[0]), float(v[1]))

    def iipair(v: Any) -> tuple[int, int]:
        return (int(v[0]), int(v[1]))

    cfg = DatasetConfig()
    for key in ("seed", "total", "master_size", "image_format", "image_quality",
                "class_balance", "n_workers"):
        if key in d:
            setattr(cfg, key, d[key])

    if "output_dir" in d:
        cfg.output_dir = Path(d["output_dir"])
    if "export_sizes" in d:
        cfg.export_sizes = [int(s) for s in d["export_sizes"]]

    if "split" in d:
        v = d["split"]
        cfg.split = SplitRatios(float(v["train"]), float(v["val"]), float(v["test"]))

    if "led" in d:
        v = d["led"]
        cfg.led = LedConfig(
            colors=v["colors"], shapes=v["shapes"], lens_types=v["lens_types"],
            brightness_range=pair(v["brightness_range"]),
            size_range=pair(v["size_range"]),
            position_margin=float(v["position_margin"]),
            glow_radius_range=pair(v["glow_radius_range"]),
            halo_intensity_range=pair(v["halo_intensity_range"]),
        )

    if "multi_led" in d:
        v = d["multi_led"]
        cfg.multi_led = MultiLedConfig(
            single=float(v["single"]), two=float(v["two"]), four=float(v["four"]),
            row=float(v["row"]), grid=float(v["grid"]), mixed_states=bool(v["mixed_states"]),
        )

    def _load_weights(cls: type, key: str) -> Any:
        if key not in d:
            return cls()
        v = d[key]
        return cls(**{k: float(v[k]) for k in cls.__dataclass_fields__})

    cfg.backgrounds = _load_weights(BackgroundWeights, "backgrounds")
    cfg.ambient      = _load_weights(AmbientWeights,    "ambient")
    cfg.glare        = _load_weights(GlareWeights,       "glare")
    cfg.occlusion    = _load_weights(OcclusionWeights,   "occlusion")
    cfg.difficulty   = _load_weights(DifficultyWeights,  "difficulty")
    cfg.groups       = _load_weights(GroupWeights,        "groups")

    if "false_positive" in d:
        v = d["false_positive"]
        cfg.false_positive = FalsePositiveConfig(
            enabled=bool(v["enabled"]), rate=float(v["rate"]), types=v["types"],
        )

    if "camera" in d:
        v = d["camera"]
        cfg.camera = CameraConfig(
            noise_sigma_range=pair(v["noise_sigma_range"]),
            blur_sigma_range=pair(v["blur_sigma_range"]),
            motion_blur_prob=float(v["motion_blur_prob"]),
            motion_blur_length_range=iipair(v["motion_blur_length_range"]),
            chromatic_shift_range=pair(v["chromatic_shift_range"]),
            gamma_range=pair(v["gamma_range"]),
            white_balance_shift=float(v["white_balance_shift"]),
            jpeg_artifact_prob=float(v["jpeg_artifact_prob"]),
            jpeg_quality_range=iipair(v["jpeg_quality_range"]),
        )

    return cfg
