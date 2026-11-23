"""Unit tests for the v2 synthetic dataset pipeline."""
from __future__ import annotations

from pathlib import Path
import json
import tempfile

import numpy as np
import pytest

from led_vision.synthetic.backgrounds import BackgroundGenerator, BACKGROUND_STYLES
from led_vision.synthetic.config import (
    DatasetConfig,
    LedConfig,
    SplitRatios,
    load_config,
)
from led_vision.synthetic.effects import (
    apply_ambient,
    apply_chromatic_aberration,
    apply_false_positive,
    apply_gamma,
    apply_gaussian_blur,
    apply_glare,
    apply_jpeg_artifacts,
    apply_motion_blur,
    apply_noise,
    apply_occlusion,
    apply_rotation,
    apply_white_balance,
)
from led_vision.synthetic.generator import DatasetGenerator
from led_vision.synthetic.led import ALL_SHAPES, LEDRenderer, LedSpec
from led_vision.synthetic.scene import SceneBuilder


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

RNG_SEED = 7


@pytest.fixture(scope="module")
def rng() -> np.random.Generator:
    return np.random.default_rng(RNG_SEED)


@pytest.fixture(scope="module")
def bg_gen(rng: np.random.Generator) -> BackgroundGenerator:
    return BackgroundGenerator(rng)


@pytest.fixture(scope="module")
def led_renderer(rng: np.random.Generator) -> LEDRenderer:
    return LEDRenderer(rng)


@pytest.fixture()
def default_config() -> DatasetConfig:
    return DatasetConfig(total=8, export_sizes=[64, 128], master_size=256)


# ---------------------------------------------------------------------------
# BackgroundGenerator tests
# ---------------------------------------------------------------------------

class TestBackgroundGenerator:
    @pytest.mark.parametrize("style", BACKGROUND_STYLES)
    def test_output_shape_and_dtype(self, bg_gen: BackgroundGenerator, style: str) -> None:
        image = bg_gen.generate(style, 128)
        assert image.shape == (128, 128, 3), f"Wrong shape for {style}"
        assert image.dtype == np.uint8, f"Wrong dtype for {style}"

    @pytest.mark.parametrize("size", [64, 128, 256])
    def test_various_sizes(self, bg_gen: BackgroundGenerator, size: int) -> None:
        image = bg_gen.generate("plain", size)
        assert image.shape == (size, size, 3)

    def test_unknown_style_raises(self, bg_gen: BackgroundGenerator) -> None:
        with pytest.raises(ValueError, match="Unknown background style"):
            bg_gen.generate("neon_rainbow", 64)

    def test_gradient_correct_shape(self, bg_gen: BackgroundGenerator) -> None:
        for _ in range(5):
            img = bg_gen.generate("gradient", 64)
            assert img.shape == (64, 64, 3)

    def test_pcb_family_are_different(self, rng: np.random.Generator) -> None:
        gen = BackgroundGenerator(rng)
        imgs = [gen.generate(s, 128) for s in ("pcb", "blue_pcb", "realistic_pcb", "aluminum_pcb")]
        # At least the first channels should differ
        assert not all(np.array_equal(imgs[0], img) for img in imgs[1:])

    @pytest.mark.parametrize("style", ["pcb", "blue_pcb", "realistic_pcb", "aluminum_pcb"])
    def test_drawn_details_are_visible(self, style: str) -> None:
        """Traces, pads and components must land on the image, not on a discarded copy."""
        for seed in range(5):
            img = BackgroundGenerator(np.random.default_rng(seed)).generate(style, 256)
            # Pure sensor noise is ~3-6 std; drawn structure pushes the 99th percentile
            # of the deviation from the median well beyond that.
            dev = np.abs(img.astype(np.float32) - np.median(img, axis=(0, 1)))
            assert np.percentile(dev, 99) > 25, f"{style} seed={seed} has no visible detail"

    @pytest.mark.parametrize("style", ["pcb", "realistic_pcb"])
    def test_green_pcb_not_near_black(self, style: str) -> None:
        for seed in range(5):
            img = BackgroundGenerator(np.random.default_rng(seed)).generate(style, 128)
            assert img[:, :, 1].mean() > 70, f"{style} seed={seed} solder mask too dark"


# ---------------------------------------------------------------------------
# LEDRenderer tests
# ---------------------------------------------------------------------------

class TestLEDRenderer:
    @pytest.mark.parametrize("shape", ALL_SHAPES)
    @pytest.mark.parametrize("state", ["on", "off"])
    def test_render_shape_dtype(
        self, led_renderer: LEDRenderer, shape: str, state: str
    ) -> None:
        base = np.full((128, 128, 3), 40, dtype=np.uint8)
        spec = LedSpec(
            state=state, color="red", shape=shape, lens_type="clear",
            brightness=0.8, size_pct=0.18, pos_x=0.5, pos_y=0.5,
            glow_radius=1.0, halo_intensity=0.7,
        )
        result, cx, cy, r = led_renderer.render(base, spec)
        assert result.shape == (128, 128, 3)
        assert result.dtype == np.uint8
        assert 0 <= cx < 128
        assert 0 <= cy < 128
        assert r > 0

    def test_on_brighter_than_off(self, led_renderer: LEDRenderer) -> None:
        base = np.full((128, 128, 3), 30, dtype=np.uint8)
        on_spec = LedSpec(
            state="on", color="green", shape="round", lens_type="clear",
            brightness=0.9, size_pct=0.25, pos_x=0.5, pos_y=0.5,
            glow_radius=1.5, halo_intensity=0.8,
        )
        off_spec = LedSpec(
            state="off", color="green", shape="round", lens_type="clear",
            brightness=0.05, size_pct=0.25, pos_x=0.5, pos_y=0.5,
            glow_radius=1.5, halo_intensity=0.8,
        )
        on_img, _, _, _ = led_renderer.render(base.copy(), on_spec)
        off_img, _, _, _ = led_renderer.render(base.copy(), off_spec)
        assert on_img.mean() > off_img.mean()

    @pytest.mark.parametrize("lens", ["clear", "diffused"])
    def test_lens_types(self, led_renderer: LEDRenderer, lens: str) -> None:
        base = np.zeros((128, 128, 3), dtype=np.uint8)
        spec = LedSpec(
            state="on", color="blue", shape="dome", lens_type=lens,
            brightness=0.75, size_pct=0.2, pos_x=0.5, pos_y=0.5,
            glow_radius=1.2, halo_intensity=0.6,
        )
        result, _, _, _ = led_renderer.render(base, spec)
        assert result.sum() > 0

    def test_render_multiple(self, led_renderer: LEDRenderer) -> None:
        base = np.zeros((256, 256, 3), dtype=np.uint8)
        specs = [
            LedSpec("on",  "red",   "round", "clear",  0.8, 0.1, 0.25, 0.5, 1.0, 0.7),
            LedSpec("off", "green", "smd",   "diffused",0.1, 0.1, 0.75, 0.5, 1.0, 0.5),
        ]
        result, positions = led_renderer.render_multiple(base, specs)
        assert result.shape == (256, 256, 3)
        assert len(positions) == 2

    def test_smd_package_visible_when_off(self, led_renderer: LEDRenderer) -> None:
        base = np.zeros((128, 128, 3), dtype=np.uint8)
        spec = LedSpec(
            state="off", color="white", shape="smd", lens_type="clear",
            brightness=0.05, size_pct=0.2, pos_x=0.5, pos_y=0.5,
            glow_radius=1.0, halo_intensity=0.5,
        )
        result, _, _, _ = led_renderer.render(base, spec)
        assert result.sum() > 0  # SMD package housing is always visible


# ---------------------------------------------------------------------------
# Effects tests
# ---------------------------------------------------------------------------

class TestEffects:
    @pytest.fixture(autouse=True)
    def _base_image(self) -> np.ndarray:
        self.img = np.full((128, 128, 3), 80, dtype=np.uint8)
        self.rng = np.random.default_rng(42)

    def test_gaussian_blur_no_change_for_zero_sigma(self) -> None:
        result = apply_gaussian_blur(self.img, 0.0)
        assert np.array_equal(result, self.img)

    def test_gaussian_blur_changes_image(self) -> None:
        result = apply_gaussian_blur(self.img.copy(), 2.0)
        # Uniform image won't change visually, but dtype/shape should hold
        assert result.shape == self.img.shape
        assert result.dtype == np.uint8

    def test_noise_adds_variation(self) -> None:
        base = np.full((128, 128, 3), 128, dtype=np.uint8)
        result = apply_noise(base.copy(), sigma=15.0, rng=self.rng)
        assert not np.array_equal(result, base)
        assert result.dtype == np.uint8

    def test_gamma_preserves_shape(self) -> None:
        for gamma in (0.5, 1.0, 1.8):
            result = apply_gamma(self.img.copy(), gamma)
            assert result.shape == (128, 128, 3)
            assert result.dtype == np.uint8

    def test_gamma_identity_unchanged(self) -> None:
        result = apply_gamma(self.img.copy(), 1.0)
        assert np.array_equal(result, self.img)

    def test_chromatic_aberration_shape(self) -> None:
        result = apply_chromatic_aberration(self.img.copy(), 3.0)
        assert result.shape == self.img.shape

    def test_chromatic_no_shift_unchanged(self) -> None:
        result = apply_chromatic_aberration(self.img.copy(), 0.0)
        assert np.array_equal(result, self.img)

    def test_white_balance_shape(self) -> None:
        result = apply_white_balance(self.img.copy(), 0.1, self.rng)
        assert result.shape == self.img.shape

    def test_motion_blur_shape(self) -> None:
        result = apply_motion_blur(self.img.copy(), 45.0, 7)
        assert result.shape == self.img.shape

    def test_rotation_shape(self) -> None:
        result = apply_rotation(self.img.copy(), 15.0)
        assert result.shape == self.img.shape

    def test_ambient_types(self) -> None:
        for atype in ("dark_room", "normal_indoor", "warm", "cool", "directional",
                      "low_light", "bright_office", "backlight"):
            result = apply_ambient(self.img.copy(), atype, 1.0, self.rng)
            assert result.shape == (128, 128, 3)
            assert result.dtype == np.uint8

    def test_dark_room_darkens(self) -> None:
        base = np.full((128, 128, 3), 200, dtype=np.uint8)
        result = apply_ambient(base.copy(), "dark_room", 1.0, self.rng)
        assert result.mean() < base.mean()

    def test_glare_types(self) -> None:
        positions = [(64, 64, 15)]
        for gtype in ("none", "small_highlight", "strong_glare",
                      "reflection_near", "reflection_overlap", "glossy"):
            result = apply_glare(self.img.copy(), gtype, positions, self.rng)
            assert result.shape == (128, 128, 3)

    def test_occlusion_types(self) -> None:
        positions = [(64, 64, 15)]
        for otype in ("none", "cable", "label", "shadow", "panel_edge"):
            result = apply_occlusion(self.img.copy(), otype, positions, self.rng)
            assert result.shape == (128, 128, 3)

    @pytest.mark.parametrize("fp_type", [
        "colored_screw", "reflection_spot", "printed_circle",
        "colored_button", "display_pixel", "specular_highlight",
    ])
    def test_false_positive_types(self, fp_type: str) -> None:
        result, cx, cy, r = apply_false_positive(self.img.copy(), fp_type, self.rng)
        assert result.shape == (128, 128, 3)
        assert 0 <= cx < 128
        assert 0 <= cy < 128
        assert r > 0

    def test_jpeg_artifacts_preserves_shape(self) -> None:
        result = apply_jpeg_artifacts(self.img.copy(), 75)
        assert result.shape == self.img.shape


# ---------------------------------------------------------------------------
# Config / YAML tests
# ---------------------------------------------------------------------------

class TestConfig:
    def test_default_config_valid(self) -> None:
        cfg = DatasetConfig()
        assert cfg.master_size == 512
        assert len(cfg.export_sizes) > 0

    def test_split_must_sum_to_one(self) -> None:
        with pytest.raises(ValueError):
            SplitRatios(train=0.5, val=0.5, test=0.5)

    def test_load_yaml(self) -> None:
        yaml_path = Path("configs/dataset/default.yaml")
        if not yaml_path.exists():
            pytest.skip("default.yaml not present")
        cfg = load_config(yaml_path)
        assert cfg.total > 0
        assert cfg.master_size > 0
        assert len(cfg.led.colors) > 0

    def test_led_config_has_all_shapes(self) -> None:
        cfg = DatasetConfig()
        for shape in ALL_SHAPES:
            assert shape in cfg.led.shapes


# ---------------------------------------------------------------------------
# SceneBuilder tests
# ---------------------------------------------------------------------------

class TestSceneBuilder:
    @pytest.fixture(autouse=True)
    def _setup(self) -> None:
        self.cfg = DatasetConfig(total=4, export_sizes=[64], master_size=128)
        self.rng = np.random.default_rng(42)
        self.builder = SceneBuilder(self.cfg, self.rng)

    @pytest.mark.parametrize("difficulty", ["easy", "medium", "hard", "extreme"])
    @pytest.mark.parametrize("state", ["on", "off"])
    def test_scene_builds(self, difficulty: str, state: str) -> None:
        scene = self.builder.build(
            target_state=state, difficulty=difficulty,
            group="clean_easy", scene_seed=123,
        )
        assert scene.image.shape == (128, 128, 3)
        assert scene.state in ("on", "off")
        assert scene.difficulty == difficulty

    def test_false_positive_off_state(self) -> None:
        """FP scenes must always be labelled OFF."""
        cfg = DatasetConfig(
            total=4, export_sizes=[64], master_size=128,
        )
        cfg.false_positive.rate = 1.0  # force FP
        builder = SceneBuilder(cfg, np.random.default_rng(99))
        scene = builder.build("off", "easy", "false_positives", 99)
        assert scene.state == "off"

    def test_multi_led_produces_multiple_positions(self) -> None:
        cfg = DatasetConfig(total=4, export_sizes=[64], master_size=256)
        cfg.multi_led.single = 0.0
        cfg.multi_led.two = 1.0
        cfg.multi_led.four = 0.0
        cfg.multi_led.row = 0.0
        cfg.multi_led.grid = 0.0
        builder = SceneBuilder(cfg, np.random.default_rng(7))
        scene = builder.build("on", "easy", "clean_easy", 7)
        assert len(scene.led_positions) == 2

    @pytest.mark.parametrize("layout", ["two", "four", "row", "grid"])
    def test_off_multi_led_scene_stays_off(self, layout: str) -> None:
        """Mixed states must not turn an OFF scene into an ON-labelled one."""
        cfg = DatasetConfig(total=4, export_sizes=[64], master_size=128)
        for name in ("single", "two", "four", "row", "grid"):
            setattr(cfg.multi_led, name, 1.0 if name == layout else 0.0)
        cfg.multi_led.mixed_states = True
        cfg.false_positive.enabled = False
        builder = SceneBuilder(cfg, np.random.default_rng(5))
        for seed in range(20):
            scene = builder.build("off", "easy", "clean_easy", seed)
            assert scene.state == "off"
            assert all(m["state"] == "off" for m in scene.leds)

    def test_on_scene_has_led_metadata(self) -> None:
        scene = self.builder.build("on", "easy", "clean_easy", 1)
        assert len(scene.leds) >= 1
        assert scene.leds[0]["state"] == "on"


# ---------------------------------------------------------------------------
# DatasetGenerator (integration)
# ---------------------------------------------------------------------------

class TestDatasetGenerator:
    def test_generates_expected_files(self) -> None:
        cfg = DatasetConfig(
            seed=0,
            total=4,
            master_size=128,
            export_sizes=[64],
            image_format="jpg",
        )
        with tempfile.TemporaryDirectory() as tmp:
            cfg.output_dir = Path(tmp)
            gen = DatasetGenerator(cfg)
            meta_path = gen.generate()

            assert meta_path.exists()
            # Expect 4 scenes × 1 size = 4 image files
            images = list(Path(tmp).rglob("*.jpg"))
            assert len(images) == 4

    def test_metadata_csv_has_correct_columns(self) -> None:
        cfg = DatasetConfig(
            seed=1, total=2, master_size=128, export_sizes=[64], image_format="jpg"
        )
        with tempfile.TemporaryDirectory() as tmp:
            cfg.output_dir = Path(tmp)
            meta_path = DatasetGenerator(cfg).generate()
            import csv
            with meta_path.open(encoding="utf-8") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            assert len(rows) == 2  # 2 scenes × 1 size
            assert "state" in rows[0]
            assert "difficulty" in rows[0]
            assert "background" in rows[0]

    def test_split_subdirectories_created(self) -> None:
        cfg = DatasetConfig(
            seed=2, total=6, master_size=128, export_sizes=[64], image_format="jpg"
        )
        cfg.split = SplitRatios(train=0.5, val=0.25, test=0.25)
        with tempfile.TemporaryDirectory() as tmp:
            cfg.output_dir = Path(tmp)
            DatasetGenerator(cfg).generate()
            subdirs = {p.name for p in Path(tmp).iterdir() if p.is_dir()}
            assert "64" in subdirs
            size_dir = Path(tmp) / "64"
            splits_found = {p.name for p in size_dir.iterdir() if p.is_dir()}
            assert splits_found.issubset({"train", "val", "test"})

    def test_reproducible_with_same_seed(self) -> None:
        cfg_a = DatasetConfig(seed=99, total=2, master_size=128,
                               export_sizes=[64], image_format="jpg")
        cfg_b = DatasetConfig(seed=99, total=2, master_size=128,
                               export_sizes=[64], image_format="jpg")
        with (
            tempfile.TemporaryDirectory() as tmp_a,
            tempfile.TemporaryDirectory() as tmp_b,
        ):
            cfg_a.output_dir = Path(tmp_a)
            cfg_b.output_dir = Path(tmp_b)
            DatasetGenerator(cfg_a).generate()
            DatasetGenerator(cfg_b).generate()
            files_a = sorted(
                (p.relative_to(tmp_a), p.stat().st_size)
                for p in Path(tmp_a).rglob("*.jpg")
            )
            files_b = sorted(
                (p.relative_to(tmp_b), p.stat().st_size)
                for p in Path(tmp_b).rglob("*.jpg")
            )
            assert files_a == files_b

    def test_config_json_saved(self) -> None:
        cfg = DatasetConfig(seed=3, total=2, master_size=128,
                             export_sizes=[64], image_format="jpg")
        with tempfile.TemporaryDirectory() as tmp:
            cfg.output_dir = Path(tmp)
            DatasetGenerator(cfg).generate()
            snap = json.loads((Path(tmp) / "dataset_config.json").read_text())
            assert snap["seed"] == 3
            assert snap["master_size"] == 128
