"""Unit tests for the synthetic dataset generator."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from led_vision.synthetic.backgrounds import BackgroundGenerator, BACKGROUND_STYLES
from led_vision.synthetic.config import DatasetConfig, SplitRatios, load_config
from led_vision.synthetic.generator import DatasetGenerator, _apply_blur, _apply_noise, _apply_rotation
from led_vision.synthetic.led import LEDRenderer, LED_COLORS_BGR, ALL_SHAPES


# ---------------------------------------------------------------------------
# BackgroundGenerator
# ---------------------------------------------------------------------------

class TestBackgroundGenerator:
    def setup_method(self) -> None:
        self.bg = BackgroundGenerator(np.random.default_rng(42))

    @pytest.mark.parametrize("style", BACKGROUND_STYLES)
    def test_output_shape_and_dtype(self, style: str) -> None:
        image = self.bg.generate(style, 128)
        assert image.shape == (128, 128, 3)
        assert image.dtype == np.uint8

    @pytest.mark.parametrize("size", [32, 64, 128, 256])
    def test_various_sizes(self, size: int) -> None:
        image = self.bg.generate("plain", size)
        assert image.shape == (size, size, 3)

    def test_unknown_style_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="Unknown background style"):
            self.bg.generate("laser", 64)

    def test_reproducible_with_same_seed(self) -> None:
        img1 = BackgroundGenerator(np.random.default_rng(7)).generate("pcb", 64)
        img2 = BackgroundGenerator(np.random.default_rng(7)).generate("pcb", 64)
        np.testing.assert_array_equal(img1, img2)

    def test_different_seeds_differ(self) -> None:
        img1 = BackgroundGenerator(np.random.default_rng(1)).generate("gradient", 64)
        img2 = BackgroundGenerator(np.random.default_rng(2)).generate("gradient", 64)
        assert not np.array_equal(img1, img2)


# ---------------------------------------------------------------------------
# LEDRenderer
# ---------------------------------------------------------------------------

class TestLEDRenderer:
    def setup_method(self) -> None:
        self.renderer = LEDRenderer(np.random.default_rng(42))

    def test_all_14_colors_defined(self) -> None:
        expected = {
            "red", "green", "yellow", "blue", "white", "orange",
            "purple", "cyan", "magenta", "amber", "warm_white",
            "cool_white", "lime_green", "uv_violet",
        }
        assert set(LED_COLORS_BGR.keys()) == expected

    def test_all_shapes_registered(self) -> None:
        assert set(ALL_SHAPES) == {"round", "square", "rectangular", "dome", "dome_side", "smd"}

    @pytest.mark.parametrize("shape", ALL_SHAPES)
    @pytest.mark.parametrize("state", ["on", "off"])
    def test_render_shape_and_dtype(self, shape: str, state: str) -> None:
        base = np.full((128, 128, 3), 40, dtype=np.uint8)
        result = self.renderer.render(base, state, "red", shape, 0.2, 0.5, 0.5)
        assert result.shape == (128, 128, 3)
        assert result.dtype == np.uint8

    @pytest.mark.parametrize("shape", ["round", "dome", "smd"])
    def test_on_brighter_than_off(self, shape: str) -> None:
        base = np.full((128, 128, 3), 20, dtype=np.uint8)
        on_img = LEDRenderer(np.random.default_rng(0)).render(base.copy(), "on", "red", shape, 0.25, 0.5, 0.5)
        off_img = LEDRenderer(np.random.default_rng(0)).render(base.copy(), "off", "red", shape, 0.25, 0.5, 0.5)
        assert float(on_img.mean()) > float(off_img.mean())

    def test_led_changes_image(self) -> None:
        base = np.zeros((128, 128, 3), dtype=np.uint8)
        result = self.renderer.render(base, "on", "green", "round", 0.2, 0.5, 0.5)
        assert result.sum() > 0

    def test_dome_side_changes_image(self) -> None:
        base = np.full((128, 128, 3), 245, dtype=np.uint8)
        result = self.renderer.render(base, "on", "red", "dome_side", 0.18, 0.5, 0.4)
        assert not np.array_equal(result, base)

    def test_smd_off_package_visible(self) -> None:
        base = np.zeros((128, 128, 3), dtype=np.uint8)
        result = self.renderer.render(base, "off", "white", "smd", 0.2, 0.5, 0.5)
        # SMD ceramic package should be visible even when OFF
        assert result.sum() > 0

    def test_unknown_color_raises(self) -> None:
        base = np.zeros((64, 64, 3), dtype=np.uint8)
        with pytest.raises(ValueError, match="Unknown LED color"):
            self.renderer.render(base, "on", "infrared", "round", 0.2, 0.5, 0.5)

    def test_unknown_shape_raises(self) -> None:
        base = np.zeros((64, 64, 3), dtype=np.uint8)
        with pytest.raises(ValueError, match="Unknown LED shape"):
            self.renderer.render(base, "on", "red", "triangle", 0.2, 0.5, 0.5)

    def test_unknown_state_raises(self) -> None:
        base = np.zeros((64, 64, 3), dtype=np.uint8)
        with pytest.raises(ValueError, match="Unknown LED state"):
            self.renderer.render(base, "blinking", "red", "round", 0.2, 0.5, 0.5)


# ---------------------------------------------------------------------------
# DatasetConfig
# ---------------------------------------------------------------------------

class TestDatasetConfig:
    def test_defaults_are_valid(self) -> None:
        cfg = DatasetConfig()
        assert cfg.seed >= 0
        assert cfg.total > 0
        assert len(cfg.sizes) > 0
        assert abs(cfg.split.train + cfg.split.val + cfg.split.test - 1.0) < 1e-5
        assert 0.0 < cfg.on_ratio < 1.0

    def test_split_invalid_sum_raises(self) -> None:
        with pytest.raises(ValueError, match="sum to 1.0"):
            SplitRatios(train=0.5, val=0.3, test=0.3)

    def test_load_config_from_yaml(self, tmp_path: Path) -> None:
        yaml_content = (
            "seed: 99\ntotal: 50\nsizes:\n  - 64\non_ratio: 0.6\n"
            "split:\n  train: 0.70\n  val: 0.15\n  test: 0.15\n"
        )
        config_file = tmp_path / "test.yaml"
        config_file.write_text(yaml_content, encoding="utf-8")
        cfg = load_config(config_file)
        assert cfg.seed == 99
        assert cfg.total == 50
        assert cfg.sizes == [64]
        assert abs(cfg.on_ratio - 0.6) < 1e-6


# ---------------------------------------------------------------------------
# DatasetGenerator
# ---------------------------------------------------------------------------

class TestDatasetGenerator:
    def test_assignments_correct_count(self) -> None:
        cfg = DatasetConfig(total=100, seed=42)
        gen = DatasetGenerator(cfg)
        assert len(gen._make_assignments()) == 100

    def test_assignments_on_off_balance(self) -> None:
        cfg = DatasetConfig(total=100, seed=42, on_ratio=0.5)
        gen = DatasetGenerator(cfg)
        assignments = gen._make_assignments()
        on_count = sum(1 for _, s in assignments if s == "on")
        assert on_count == 50

    def test_assignments_splits_exhaustive(self) -> None:
        cfg = DatasetConfig(total=100, seed=42)
        gen = DatasetGenerator(cfg)
        assignments = gen._make_assignments()
        splits = [s for s, _ in assignments]
        assert splits.count("train") + splits.count("val") + splits.count("test") == 100

    def test_generate_creates_files(self, tmp_path: Path) -> None:
        cfg = DatasetConfig(total=10, seed=0, sizes=[32], output_dir=tmp_path)
        DatasetGenerator(cfg).generate()

        # Check metadata and config exist
        assert (tmp_path / "metadata.csv").exists()
        assert (tmp_path / "dataset_config.json").exists()

        # Check image files were created
        images = list(tmp_path.rglob("*.jpg"))
        assert len(images) == 10

    def test_generate_reproducible(self, tmp_path: Path) -> None:
        out1 = tmp_path / "run1"
        out2 = tmp_path / "run2"

        DatasetGenerator(DatasetConfig(total=4, seed=7, sizes=[32], output_dir=out1)).generate()
        DatasetGenerator(DatasetConfig(total=4, seed=7, sizes=[32], output_dir=out2)).generate()

        imgs1 = sorted(out1.rglob("*.jpg"))
        imgs2 = sorted(out2.rglob("*.jpg"))
        assert len(imgs1) == len(imgs2)

        import cv2 as _cv2
        for p1, p2 in zip(imgs1, imgs2):
            i1 = _cv2.imread(str(p1))
            i2 = _cv2.imread(str(p2))
            np.testing.assert_array_equal(i1, i2)


# ---------------------------------------------------------------------------
# Augmentation utilities
# ---------------------------------------------------------------------------

class TestAugmentations:
    def test_blur_preserves_shape(self) -> None:
        img = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
        result = _apply_blur(img, sigma=2.0)
        assert result.shape == img.shape
        assert result.dtype == np.uint8

    def test_noise_preserves_shape(self) -> None:
        img = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
        rng = np.random.default_rng(0)
        result = _apply_noise(img, sigma=10.0, rng=rng)
        assert result.shape == img.shape
        assert result.dtype == np.uint8

    def test_rotation_preserves_shape(self) -> None:
        img = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
        result = _apply_rotation(img, angle_deg=15.0)
        assert result.shape == img.shape
        assert result.dtype == np.uint8
