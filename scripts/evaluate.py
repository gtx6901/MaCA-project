#!/usr/bin/env python
"""Unified evaluation entrypoint for the MaCA MAPPO lane."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List

import yaml

ROOT_DIR = Path(__file__).resolve().parent.parent
ENV_DIR = ROOT_DIR / "environment"

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(ENV_DIR) not in sys.path:
    sys.path.insert(0, str(ENV_DIR))

from scripts.eval_mappo_maca import main as eval_mappo_main

def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--experiment", type=str, default=None)
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--output_json", type=str, default=None)
    parser.add_argument("--maca_opponent", type=str, default=None)
    return parser.parse_args(argv)


def load_config(config_path: Path) -> Dict[str, object]:
    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError("Top-level config must be a mapping")
    return data


def build_eval_argv(config: Dict[str, object], args) -> List[str]:
    train_cfg = dict(config.get("train", {}))
    eval_cfg = dict(config.get("eval", {}))
    argv = [
        "--experiment",
        str(args.experiment or train_cfg["experiment"]),
        "--train_dir",
        str(train_cfg.get("train_dir", "train_dir/mappo")),
        "--device",
        str(train_cfg.get("device", "gpu")),
        "--episodes",
        str(int(args.episodes or eval_cfg.get("episodes", 20))),
        "--deterministic",
        str(bool(eval_cfg.get("deterministic", True))).lower(),
        "--progress",
        "true",
        "--seed",
        str(int(train_cfg.get("seed", 1))),
    ]
    if args.checkpoint:
        argv.extend(["--checkpoint", str(args.checkpoint)])
    if args.output_json:
        argv.extend(["--output_json", str(args.output_json)])
    if args.maca_opponent or eval_cfg.get("opponent"):
        argv.extend(["--maca_opponent", str(args.maca_opponent or eval_cfg.get("opponent"))])
    return argv


def main(argv=None):
    args = parse_args(argv)
    config = load_config(Path(args.config).resolve())
    eval_mappo_main(build_eval_argv(config, args))


if __name__ == "__main__":
    main()
