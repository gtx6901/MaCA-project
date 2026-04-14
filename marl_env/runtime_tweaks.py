"""Runtime-only training/evaluation tweaks for MaCA experiments.

These overrides are intentionally driven by environment variables so we can
adjust a specific recovery run without permanently mutating the project
defaults or invalidating old experiment configs.
"""

from __future__ import annotations

import json
import os
from collections import OrderedDict
from typing import Dict

from configuration.reward import GlobalVar as RewardGlobalVar


_REWARD_ENV_TO_ATTR = OrderedDict(
    [
        ("MACA_REWARD_RADAR_FIGHTER_DETECTOR", "reward_radar_fighter_detector"),
        ("MACA_REWARD_RADAR_FIGHTER_FIGHTER", "reward_radar_fighter_fighter"),
        ("MACA_REWARD_STRIKE_FIGHTER_SUCCESS", "reward_strike_fighter_success"),
        ("MACA_REWARD_STRIKE_FIGHTER_FAIL", "reward_strike_fighter_fail"),
        ("MACA_REWARD_STRIKE_ACT_VALID", "reward_strike_act_valid"),
        ("MACA_REWARD_KEEP_ALIVE_STEP", "reward_keep_alive_step"),
        ("MACA_REWARD_DRAW", "reward_draw"),
    ]
)

_CACHE = None


def _parse_float_env(name: str, default: float = 0.0) -> float:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return float(default)
    return float(raw)


def _parse_bool_env(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def load_runtime_tweaks() -> Dict[str, object]:
    global _CACHE
    if _CACHE is not None:
        return _CACHE

    applied_reward_overrides: Dict[str, float] = {}
    for env_name, attr_name in _REWARD_ENV_TO_ATTR.items():
        raw = os.getenv(env_name)
        if raw is None or raw == "":
            continue
        value = float(raw)
        setattr(RewardGlobalVar, attr_name, value)
        applied_reward_overrides[attr_name] = value

    _CACHE = {
        "reward_overrides": applied_reward_overrides,
        "missed_attack_penalty": _parse_float_env("MACA_MISSED_ATTACK_PENALTY", 0.0),
        "fire_logit_bias": _parse_float_env("MACA_FIRE_LOGIT_BIAS", 0.0),
        "fire_prob_floor": _parse_float_env("MACA_FIRE_PROB_FLOOR", 0.0),
        "eval_fire_prob_floor": _parse_float_env("MACA_EVAL_FIRE_PROB_FLOOR", 0.0),
        "contact_reward": _parse_float_env("MACA_CONTACT_REWARD", 0.0),
        "progress_reward_scale": _parse_float_env("MACA_PROGRESS_REWARD_SCALE", 0.0),
        "progress_reward_cap": _parse_float_env("MACA_PROGRESS_REWARD_CAP", 20.0),
        "attack_window_reward": _parse_float_env("MACA_ATTACK_WINDOW_REWARD", 0.0),
        "enable_buffer_squeeze_patch": _parse_bool_env("MACA_ENABLE_SF_BUFFER_SQUEEZE_PATCH", False),
    }
    return _CACHE


def get_missed_attack_penalty() -> float:
    return float(load_runtime_tweaks()["missed_attack_penalty"])


def get_fire_logit_bias() -> float:
    return float(load_runtime_tweaks()["fire_logit_bias"])


def get_fire_prob_floor() -> float:
    return float(load_runtime_tweaks()["fire_prob_floor"])


def get_eval_fire_prob_floor() -> float:
    return float(load_runtime_tweaks()["eval_fire_prob_floor"])


def get_contact_reward() -> float:
    return float(load_runtime_tweaks()["contact_reward"])


def get_progress_reward_scale() -> float:
    return float(load_runtime_tweaks()["progress_reward_scale"])


def get_progress_reward_cap() -> float:
    return float(load_runtime_tweaks()["progress_reward_cap"])


def get_attack_window_reward() -> float:
    return float(load_runtime_tweaks()["attack_window_reward"])


def buffer_squeeze_patch_enabled() -> bool:
    return bool(load_runtime_tweaks()["enable_buffer_squeeze_patch"])


def format_runtime_tweaks() -> str:
    tweaks = load_runtime_tweaks()
    return json.dumps(tweaks, ensure_ascii=False, sort_keys=True)
