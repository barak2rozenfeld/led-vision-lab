"""Synthetic LED dataset generator."""
from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from led_vision.synthetic.backgrounds import BackgroundGenerator
from led_vision.synthetic.config import DatasetConfig
from led_vision.synthetic.led import LEDRenderer


class DatasetGenerator:
    """Generates a synthetic LED ON/OFF image dataset from a DatasetConfig."""

    def __init__(self, config: DatasetConfig) -> None:
        self.config = config
        self.rng = np.random.default_rng(config.seed)
        self.bg_gen = BackgroundGenerator(self.rng)
        self.led_renderer = LEDRenderer(self.rng)

    def generate(self) -> Path:
        """
        Run generation for all configured sizes.

        Returns the output directory path.
        """
        cfg = self.config
        cfg.output_dir.mkdir(parents=True, exist_ok=True)

        assignments = self._make_assignments()
        self._save_config_json(cfg.output_dir)

        all_rows: list[dict] = []

        for size in cfg.sizes:
            print(f"\n[{size}×{size}] Generating {cfg.total} images …")
            self._create_split_dirs(cfg.output_dir / str(size))

            for idx, (split, state) in enumerate(tqdm(assignments, desc=f"  {size}px")):
                row = self._generate_one(idx, size, split, state)
                all_rows.append(row)

        metadata_path = cfg.output_dir / "metadata.csv"
        _save_csv(all_rows, metadata_path)

        total = len(all_rows)
        print(f"\nDone — {total} image(s) saved to '{cfg.output_dir}'")
        print(f"Metadata : {metadata_path}")
        print(f"Config   : {cfg.output_dir / 'dataset_config.json'}")
        return cfg.output_dir

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _generate_one(self, idx: int, size: int, split: str, state: str) -> dict:
        cfg = self.config

        color_name = str(self.rng.choice(cfg.led.colors))
        shape = str(self.rng.choice(cfg.led.shapes))
        led_size_pct = float(self.rng.uniform(*cfg.led.size_range))

        margin = cfg.led.position_margin
        pos_x = float(self.rng.uniform(margin, 1.0 - margin))
        pos_y = float(self.rng.uniform(margin, 1.0 - margin))

        bg_style = self._sample_background_style()
        blur_px = float(self.rng.uniform(*cfg.variation.blur_range))
        noise_sigma = float(self.rng.uniform(*cfg.variation.noise_sigma_range))
        camera_angle = float(self.rng.uniform(*cfg.variation.camera_angle_range))
        ambient_factor = float(self.rng.uniform(*cfg.variation.ambient_factor_range))

        # Build image
        image = self.bg_gen.generate(bg_style, size)
        image = _apply_ambient(image, ambient_factor)
        image = self.led_renderer.render(image, state, color_name, shape, led_size_pct, pos_x, pos_y)
        if blur_px > 0.3:
            image = _apply_blur(image, blur_px)
        if noise_sigma > 0.3:
            image = _apply_noise(image, noise_sigma, self.rng)
        if abs(camera_angle) > 0.3:
            image = _apply_rotation(image, camera_angle)

        # Save
        image_id = f"{idx:08d}"
        filename = f"{image_id}_{state}.jpg"
        out_path = cfg.output_dir / str(size) / split / state / filename
        cv2.imwrite(str(out_path), image, [cv2.IMWRITE_JPEG_QUALITY, cfg.image_quality])

        return {
            "id": image_id,
            "split": split,
            "state": state,
            "size": size,
            "led_color": color_name,
            "led_shape": shape,
            "led_size_pct": round(led_size_pct, 4),
            "pos_x": round(pos_x, 4),
            "pos_y": round(pos_y, 4),
            "background_style": bg_style,
            "blur_px": round(blur_px, 3),
            "noise_sigma": round(noise_sigma, 3),
            "camera_angle_deg": round(camera_angle, 3),
            "ambient_factor": round(ambient_factor, 3),
            "seed": cfg.seed,
            "filepath": str((Path(str(size)) / split / state / filename)),
        }

    def _make_assignments(self) -> list[tuple[str, str]]:
        """Assign each sample a (split, state) pair, shuffled reproducibly."""
        cfg = self.config
        n = cfg.total

        n_on = int(n * cfg.on_ratio)
        n_off = n - n_on
        states = ["on"] * n_on + ["off"] * n_off
        self.rng.shuffle(states)

        n_train = int(n * cfg.split.train)
        n_val = int(n * cfg.split.val)
        n_test = n - n_train - n_val
        splits = ["train"] * n_train + ["val"] * n_val + ["test"] * n_test
        self.rng.shuffle(splits)

        return list(zip(splits, states))

    def _sample_background_style(self) -> str:
        bg = self.config.backgrounds
        styles = ["plain", "gradient", "pcb", "blue_pcb", "realistic_pcb",
                  "metal", "noisy", "white_studio", "aluminum_pcb"]
        weights = [bg.plain, bg.gradient, bg.pcb, bg.blue_pcb, bg.realistic_pcb,
                   bg.metal, bg.noisy, bg.white_studio, bg.aluminum_pcb]
        total = sum(weights)
        probs = [w / total for w in weights]
        return str(self.rng.choice(styles, p=probs))

    def _create_split_dirs(self, base: Path) -> None:
        for split in ("train", "val", "test"):
            for state in ("on", "off"):
                (base / split / state).mkdir(parents=True, exist_ok=True)

    def _save_config_json(self, output_dir: Path) -> None:
        cfg = self.config
        payload = {
            "seed": cfg.seed,
            "total": cfg.total,
            "sizes": cfg.sizes,
            "on_ratio": cfg.on_ratio,
            "split": {"train": cfg.split.train, "val": cfg.split.val, "test": cfg.split.test},
            "led_colors": cfg.led.colors,
            "led_shapes": cfg.led.shapes,
            "led_size_range": list(cfg.led.size_range),
            "position_margin": cfg.led.position_margin,
            "backgrounds": {
                "plain": cfg.backgrounds.plain,
                "gradient": cfg.backgrounds.gradient,
                "pcb": cfg.backgrounds.pcb,
                "blue_pcb": cfg.backgrounds.blue_pcb,
                "realistic_pcb": cfg.backgrounds.realistic_pcb,
                "metal": cfg.backgrounds.metal,
                "noisy": cfg.backgrounds.noisy,
                "white_studio": cfg.backgrounds.white_studio,
                "aluminum_pcb": cfg.backgrounds.aluminum_pcb,
            },
            "variation": {
                "blur_range": list(cfg.variation.blur_range),
                "noise_sigma_range": list(cfg.variation.noise_sigma_range),
                "camera_angle_range": list(cfg.variation.camera_angle_range),
                "ambient_factor_range": list(cfg.variation.ambient_factor_range),
                "brightness_range": list(cfg.variation.brightness_range),
            },
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        (output_dir / "dataset_config.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


# ------------------------------------------------------------------
# Image augmentation utilities
# ------------------------------------------------------------------

def _apply_ambient(image: np.ndarray, factor: float) -> np.ndarray:
    return np.clip(image.astype(np.float32) * factor, 0, 255).astype(np.uint8)


def _apply_blur(image: np.ndarray, sigma: float) -> np.ndarray:
    ksize = int(sigma * 4) | 1
    return cv2.GaussianBlur(image, (ksize, ksize), sigma)


def _apply_noise(image: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    noise = rng.normal(0.0, sigma, image.shape).astype(np.float32)
    return np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def _apply_rotation(image: np.ndarray, angle_deg: float) -> np.ndarray:
    h, w = image.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle_deg, 1.0)
    return cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def _save_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
