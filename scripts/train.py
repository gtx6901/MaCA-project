#!/usr/bin/env python
"""Unified training entrypoint for the MaCA MAPPO lane."""

from __future__ import annotations

import argparse
import subprocess
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

from scripts.train_mappo_maca import main as train_mappo_main


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--experiment", type=str, default=None)
    parser.add_argument("--train_dir", type=str, default=None)
    return parser.parse_args(argv)


def load_config(config_path: Path) -> Dict[str, object]:
    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError("Top-level config must be a mapping")
    return data


def _append_cli(cli_args: List[str], key: str, value) -> None:
    flag = "--%s" % key
    if isinstance(value, bool):
        cli_args.extend([flag, "true" if value else "false"])
    else:
        cli_args.extend([flag, str(value)])


def build_train_argv(config: Dict[str, object]) -> List[str]:
    cli_args: List[str] = []
    train_cfg = dict(config.get("train", {}))
    env_cfg = dict(config.get("env", {}))
    eval_cfg = dict(config.get("eval", {}))

    for key, value in train_cfg.items():
        if key == "resume":
            if value:
                cli_args.append("--resume")
            continue
        _append_cli(cli_args, key, value)

    env_key_map = {
        "map_path": "maca_map_path",
        "red_obs_ind": "maca_red_obs_ind",
        "opponent": "maca_opponent",
        "max_step": "maca_max_step",
        "render": "maca_render",
        "random_pos": "maca_random_pos",
        "adaptive_support_policy": "maca_adaptive_support_policy",
        "support_search_hold": "maca_support_search_hold",
        "delta_course_action": "maca_delta_course_action",
        "course_delta_deg": "maca_course_delta_deg",
        "max_visible_enemies": "maca_max_visible_enemies",
        "friendly_attrition_penalty": "maca_friendly_attrition_penalty",
        "enemy_attrition_reward": "maca_enemy_attrition_reward",
        "track_memory_steps": "maca_track_memory_steps",
        "contact_reward": "maca_contact_reward",
        "progress_reward_scale": "maca_progress_reward_scale",
        "progress_reward_cap": "maca_progress_reward_cap",
        "attack_window_reward": "maca_attack_window_reward",
        "agent_aux_reward_scale": "maca_agent_aux_reward_scale",
    }
    for key, value in env_cfg.items():
        mapped_key = env_key_map.get(key)
        if mapped_key is None:
            raise KeyError("Unsupported env config key: %s" % key)
        _append_cli(cli_args, mapped_key, value)

    eval_key_map = {
        "every_env_steps": "eval_every_env_steps",
        "episodes": "eval_episodes",
        "deterministic": "eval_deterministic",
        "opponent": "eval_opponent",
    }
    for key, value in eval_cfg.items():
        mapped_key = eval_key_map.get(key)
        if mapped_key is None:
            raise KeyError("Unsupported eval config key: %s" % key)
        _append_cli(cli_args, mapped_key, value)

    return cli_args


def maybe_collect_teacher_dataset(config: Dict[str, object], train_cfg: Dict[str, object]) -> Path:
    bc_cfg = dict(config.get("bc_warm_start", {}))
    dataset_path = bc_cfg.get("dataset_path")
    if dataset_path:
        return Path(dataset_path)

    output_path = ROOT_DIR / "exports" / ("%s_teacher_dataset.npz" % train_cfg["experiment"])
    cmd = [
        sys.executable,
        str(ROOT_DIR / "scripts" / "collect_teacher_maca.py"),
        "--teacher_agent",
        str(bc_cfg.get("teacher_agent", "fix_rule")),
        "--episodes",
        str(int(bc_cfg.get("teacher_episodes", 200))),
        "--max_attempt_episodes",
        str(int(bc_cfg.get("max_attempt_episodes", 0))),
        "--seed",
        str(int(train_cfg.get("seed", 1))),
        "--output_path",
        str(output_path),
        "--wins_only",
        "true" if bool(bc_cfg.get("wins_only", False)) else "false",
        "--maca_opponent",
        str(config.get("env", {}).get("opponent", "fix_rule")),
        "--maca_map_path",
        str(config.get("env", {}).get("map_path", "maps/1000_1000_fighter10v10.map")),
        "--maca_red_obs_ind",
        str(config.get("env", {}).get("red_obs_ind", "simple")),
        "--maca_max_step",
        str(int(config.get("env", {}).get("max_step", 1000))),
        "--maca_max_visible_enemies",
        str(int(config.get("env", {}).get("max_visible_enemies", 4))),
        "--maca_friendly_attrition_penalty",
        str(float(config.get("env", {}).get("friendly_attrition_penalty", 200.0))),
        "--maca_enemy_attrition_reward",
        str(float(config.get("env", {}).get("enemy_attrition_reward", 100.0))),
    ]
    if bc_cfg.get("min_destroy_balance") is not None:
        cmd.extend(["--min_destroy_balance", str(float(bc_cfg.get("min_destroy_balance")))])
    if bc_cfg.get("min_round_reward") is not None:
        cmd.extend(["--min_round_reward", str(float(bc_cfg.get("min_round_reward")))])
    subprocess.check_call(cmd, cwd=str(ROOT_DIR))
    return output_path


def maybe_run_bc_warm_start(config: Dict[str, object]) -> None:
    bc_cfg = dict(config.get("bc_warm_start", {}))
    if not bc_cfg.get("enabled", False):
        return

    train_cfg = dict(config.get("train", {}))
    dataset_path = maybe_collect_teacher_dataset(config, train_cfg)
    cmd = [
        sys.executable,
        str(ROOT_DIR / "scripts" / "pretrain_bc_maca.py"),
        "--dataset_path",
        str(dataset_path),
        "--experiment",
        str(train_cfg["experiment"]),
        "--train_dir",
        str(train_cfg.get("train_dir", "train_dir/mappo")),
        "--device",
        str(train_cfg.get("device", "gpu")),
        "--seed",
        str(int(train_cfg.get("seed", 1))),
        "--epochs",
        str(int(bc_cfg.get("epochs", 8))),
        "--batch_size",
        str(int(bc_cfg.get("batch_size", 8192))),
        "--learning_rate",
        str(float(bc_cfg.get("learning_rate", train_cfg.get("learning_rate", 3e-4)))),
        "--chunk_len",
        str(int(bc_cfg.get("chunk_len", train_cfg.get("chunk_len", 16)))),
        "--burn_in",
        str(int(bc_cfg.get("burn_in", train_cfg.get("burn_in", 8)))),
        "--hidden_size",
        str(int(train_cfg.get("hidden_size", 256))),
        "--role_embed_dim",
        str(int(train_cfg.get("role_embed_dim", 8))),
        "--course_embed_dim",
        str(int(train_cfg.get("course_embed_dim", 16))),
    ]
    subprocess.check_call(cmd, cwd=str(ROOT_DIR))


def main(argv=None):
    args = parse_args(argv)
    config_path = Path(args.config).resolve()
    config = load_config(config_path)
    train_cfg = dict(config.get("train", {}))
    if args.experiment:
        train_cfg["experiment"] = args.experiment
    if args.train_dir:
        train_cfg["train_dir"] = args.train_dir
    config["train"] = train_cfg

    if "experiment" not in train_cfg:
        raise KeyError("Config must define train.experiment")

    maybe_run_bc_warm_start(config)

    train_argv = build_train_argv(config)
    bc_cfg = dict(config.get("bc_warm_start", {}))
    if bc_cfg.get("enabled", False) and "--resume" not in train_argv:
        train_argv.append("--resume")
    train_mappo_main(train_argv)


if __name__ == "__main__":
    main()
