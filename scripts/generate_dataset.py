#!/usr/bin/env python3
"""CLI — Generate a synthetic LED ON/OFF dataset.

Usage examples:
  uv run python scripts/generate_dataset.py
  uv run python scripts/generate_dataset.py --config configs/dataset/default.yaml
  uv run python scripts/generate_dataset.py --total 1000 --sizes 64 128 256
  uv run python scripts/generate_dataset.py --total 100000 --balance rare_fault
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

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
    parser.add_argument("--config", type=Path, default=None, help="YAML config file")
    parser.add_argument("--total", type=int, default=None, help="Number of scenes")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--output", type=Path, default=None, help="Output directory")
    parser.add_argument(
        "--sizes", nargs="+", type=int, default=None,
        help="Export sizes, e.g. --sizes 64 128 256",
    )
    parser.add_argument(
        "--balance", type=str, default=None,
        choices=["balanced", "imbalanced_80_20", "imbalanced_95_5", "rare_fault"],
        help="Class balance preset",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

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

    if args.total is not None:
        config.total = args.total
    if args.seed is not None:
        config.seed = args.seed
    if args.output is not None:
        config.output_dir = args.output
    if args.sizes is not None:
        config.export_sizes = args.sizes
    if args.balance is not None:
        config.class_balance = args.balance

    print(f"Scenes : {config.total}")
    print(f"Master : {config.master_size}×{config.master_size}")
    print(f"Sizes  : {config.export_sizes}")
    print(f"Balance: {config.class_balance}")
    print(f"Seed   : {config.seed}")
    print(f"Output : {config.output_dir}")

    DatasetGenerator(config).generate()


if __name__ == "__main__":
    main()
