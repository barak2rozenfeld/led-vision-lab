#!/usr/bin/env python3
"""CLI — Generate a synthetic LED ON/OFF dataset.

Usage examples:
  uv run python scripts/generate_dataset.py
  uv run python scripts/generate_dataset.py --config configs/dataset/default.yaml
  uv run python scripts/generate_dataset.py --total 1000 --sizes 128 512
  uv run python scripts/generate_dataset.py --total 100000 --sizes 128 512 1024 2048 4096
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running from repo root without installing
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from led_vision.synthetic.config import DatasetConfig, load_config
from led_vision.synthetic.generator import DatasetGenerator


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="generate_dataset",
        description="Generate a synthetic LED ON/OFF image dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--config", type=Path, default=None, help="Path to YAML config file")
    parser.add_argument("--total", type=int, default=None, help="Total number of images per size")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--output", type=Path, default=None, help="Output directory")
    parser.add_argument("--sizes", nargs="+", type=int, default=None,
                        help="Image sizes to generate (e.g. --sizes 128 512 1024)")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    # Load config: CLI path > default YAML > built-in defaults
    default_yaml = Path("configs/dataset/default.yaml")
    if args.config and Path(args.config).exists():
        config = load_config(Path(args.config))
        print(f"Config : {args.config}")
    elif default_yaml.exists():
        config = load_config(default_yaml)
        print(f"Config : {default_yaml}")
    else:
        config = DatasetConfig()
        print("Config : built-in defaults")

    # Apply CLI overrides
    if args.total is not None:
        config.total = args.total
    if args.seed is not None:
        config.seed = args.seed
    if args.output is not None:
        config.output_dir = args.output
    if args.sizes is not None:
        config.sizes = args.sizes

    print(f"Total  : {config.total} images per size")
    print(f"Sizes  : {config.sizes}")
    print(f"Seed   : {config.seed}")
    print(f"Output : {config.output_dir}")
    print(f"ON/OFF : {config.on_ratio:.0%} / {1 - config.on_ratio:.0%}")

    DatasetGenerator(config).generate()


if __name__ == "__main__":
    main()
