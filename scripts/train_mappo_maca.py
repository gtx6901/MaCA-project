#!/usr/bin/env python
"""Train MaCA with recurrent MAPPO (chunked PPO + burn-in)."""

from __future__ import annotations

import argparse
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch

ROOT_DIR = Path(__file__).resolve().parent.parent
ENV_DIR = ROOT_DIR / "environment"
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(ENV_DIR) not in sys.path:
    sys.path.insert(0, str(ENV_DIR))

from marl_env.mappo_model import MAPPOModelConfig, TeamActorCritic
from marl_train.checkpoint import (
    ValueNormalizer,
    load_best_metric,
    load_checkpoint,
    save_best_checkpoint,
    save_checkpoint,
    save_config,
)
from marl_train.collector import CollectorPool, allocate_shared_actor_params, build_env, write_shared_actor_params
from marl_train.eval import run_evaluation
from marl_train.logging_utils import (
    build_summary_writer,
    log_summary,
    log_train_scalars,
    summarize_episode_stats,
)
from marl_train.ppo_update import action_distribution_stats, ppo_update
from marl_train.rollout import compute_aux_advantages, compute_gae, rollout


def str2bool(value) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=str, default="maca_mappo_recurrent")
    parser.add_argument("--train_dir", type=str, default="train_dir/mappo")
    parser.add_argument("--device", type=str, default="gpu")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--num_envs", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)

    parser.add_argument("--rollout", type=int, default=64)
    parser.add_argument("--chunk_len", type=int, default=16)
    parser.add_argument("--burn_in", type=int, default=8)
    parser.add_argument("--train_for_env_steps", type=int, default=20_000_000)

    parser.add_argument("--ppo_epochs", type=int, default=5)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--lr_schedule", type=str, default="linear")
    parser.add_argument("--lr_min_ratio", type=float, default=0.33)
    parser.add_argument("--lr_second_half_start_frac", type=float, default=0.5)
    parser.add_argument("--lr_second_half_ratio", type=float, default=0.33)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae_lambda", type=float, default=0.95)
    parser.add_argument("--clip_ratio", type=float, default=0.1)
    parser.add_argument("--value_loss_coeff", type=float, default=0.5)
    parser.add_argument("--aux_value_loss_coeff", type=float, default=0.25)
    parser.add_argument("--priority_aux_loss_coeff", type=float, default=0.05)
    parser.add_argument("--disable_aux_value_heads", type=str2bool, default=False)
    parser.add_argument("--entropy_coeff", type=float, default=0.01)
    parser.add_argument("--max_grad_norm", type=float, default=5.0)
    parser.add_argument("--max_actor_grad_norm", type=float, default=2.5)
    parser.add_argument("--max_critic_grad_norm", type=float, default=2.5)
    parser.add_argument("--num_mini_batches", type=int, default=4)
    parser.add_argument("--attack_rule_mode", type=str, default="none")
    parser.add_argument("--attack_policy_mode", type=str, default="full_discrete")
    parser.add_argument("--attack_rule_prefer_long", type=str2bool, default=True)
    parser.add_argument("--disable_high_level_mode", type=str2bool, default=False)

    parser.add_argument("--mode_interval", type=int, default=8)
    parser.add_argument("--high_level_loss_coeff", type=float, default=0.25)
    parser.add_argument("--high_level_entropy_coeff", type=float, default=0.001)
    parser.add_argument("--teacher_bc_checkpoint", type=str, default="")
    parser.add_argument("--imitation_coef", type=float, default=0.0)
    parser.add_argument("--imitation_warmup_updates", type=int, default=0)

    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--course_embed_dim", type=int, default=16)
    parser.add_argument("--screen_embed_dim", type=int, default=64)

    parser.add_argument("--save_every_sec", type=int, default=900)
    parser.add_argument("--log_every_sec", type=int, default=30)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--resume_checkpoint", type=str, default="")
    parser.add_argument("--resume_env_steps", type=int, default=-1)
    parser.add_argument("--tensorboard", type=str2bool, default=True)
    parser.add_argument("--eval_every_env_steps", type=int, default=1_000_000)
    parser.add_argument("--eval_episodes", type=int, default=20)
    parser.add_argument("--eval_deterministic", type=str2bool, default=True)
    parser.add_argument("--eval_maca_render", type=str2bool, default=True)
    parser.add_argument("--eval_opponent", type=str, default=None)
    parser.add_argument("--save_best_checkpoint", type=str2bool, default=True)
    parser.add_argument("--best_checkpoint_source", type=str, default="eval_then_train")
    parser.add_argument("--best_metric_name", type=str, default="total_win_rate")
    parser.add_argument("--debug_freeze_update_after_env_steps", type=int, default=-1)
    parser.add_argument("--freeze_value_normalizer_after_env_steps", type=int, default=-1)

    parser.add_argument("--team_adv_weight", type=float, default=1.0)
    parser.add_argument("--aux_adv_weight", type=float, default=1.0)

    parser.add_argument("--maca_map_path", type=str, default="maps/1000_1000_fighter10v10.map")
    parser.add_argument("--maca_red_obs_ind", type=str, default="simple")
    parser.add_argument("--maca_opponent", type=str, default="fix_rule")
    parser.add_argument("--maca_max_step", type=int, default=1000)
    parser.add_argument("--maca_render", type=str2bool, default=False)
    parser.add_argument("--maca_random_pos", type=str2bool, default=False)
    parser.add_argument("--maca_adaptive_support_policy", type=str2bool, default=True)
    parser.add_argument("--maca_support_search_hold", type=int, default=6)
    parser.add_argument("--maca_delta_course_action", type=str2bool, default=True)
    parser.add_argument("--maca_course_delta_deg", type=float, default=45.0)
    parser.add_argument("--maca_max_visible_enemies", type=int, default=4)
    parser.add_argument("--maca_friendly_attrition_penalty", type=float, default=200.0)
    parser.add_argument("--maca_enemy_attrition_reward", type=float, default=100.0)

    parser.add_argument("--maca_track_memory_steps", type=int, default=12)
    parser.add_argument("--maca_contact_reward", type=float, default=0.1)
    parser.add_argument("--maca_progress_reward_scale", type=float, default=0.002)
    parser.add_argument("--maca_progress_reward_cap", type=float, default=20.0)
    parser.add_argument("--maca_attack_window_reward", type=float, default=0.1)
    parser.add_argument("--maca_agent_aux_reward_scale", type=float, default=0.0)
    parser.add_argument("--maca_mode_reward_scale", type=float, default=0.5)
    parser.add_argument("--maca_exec_reward_scale", type=float, default=0.2)
    parser.add_argument("--maca_disengage_penalty", type=float, default=0.05)
    parser.add_argument("--maca_bearing_reward_scale", type=float, default=0.05)
    parser.add_argument("--maca_boundary_penalty_margin", type=float, default=120.0)
    parser.add_argument("--maca_boundary_penalty_scale", type=float, default=0.01)
    parser.add_argument("--maca_boundary_stuck_penalty_enabled", type=str2bool, default=True)
    parser.add_argument("--maca_boundary_stuck_trigger_steps", type=int, default=24)
    parser.add_argument("--maca_boundary_stuck_ramp_steps", type=int, default=20)
    parser.add_argument("--maca_search_reward_scale", type=float, default=0.015)
    parser.add_argument("--maca_reacquire_reward_scale", type=float, default=0.02)
    parser.add_argument("--maca_priority_grid_h", type=int, default=5)
    parser.add_argument("--maca_priority_grid_w", type=int, default=5)
    parser.add_argument("--maca_priority_top_k", type=int, default=3)
    parser.add_argument("--maca_priority_evidence_weight", type=float, default=1.0)
    parser.add_argument("--maca_priority_uncertainty_weight", type=float, default=0.9)
    parser.add_argument("--maca_priority_diffusion_weight", type=float, default=0.7)
    parser.add_argument("--maca_priority_crowding_penalty", type=float, default=0.6)
    parser.add_argument("--maca_priority_assignment_penalty", type=float, default=0.35)
    parser.add_argument("--maca_priority_distance_penalty", type=float, default=0.25)
    parser.add_argument("--maca_priority_unseen_cap_steps", type=int, default=40)
    parser.add_argument("--maca_priority_memory_decay", type=float, default=0.92)
    parser.add_argument("--maca_priority_diffusion_rate", type=float, default=0.25)
    parser.add_argument("--maca_priority_passive_recv_weight", type=float, default=0.25)
    parser.add_argument("--maca_priority_known_enemy_boost", type=float, default=0.8)
    parser.add_argument("--maca_priority_unseen_threshold", type=float, default=0.7)
    parser.add_argument("--maca_semantic_screen_downsample", type=int, default=4)
    parser.add_argument("--maca_terminal_ammo_fail_penalty", type=float, default=80.0)
    parser.add_argument("--maca_terminal_participation_penalty", type=float, default=40.0)

    parser.add_argument("--curriculum_enabled", type=str2bool, default=False)
    parser.add_argument("--curriculum_easy_frac", type=float, default=0.3)
    parser.add_argument("--curriculum_medium_frac", type=float, default=0.7)
    parser.add_argument("--curriculum_easy_opponent", type=str, default="fix_rule_no_att")
    parser.add_argument("--curriculum_medium_opponent", type=str, default="fix_rule")
    parser.add_argument("--curriculum_full_opponent", type=str, default="fix_rule")
    parser.add_argument("--curriculum_easy_max_step", type=int, default=600)
    parser.add_argument("--curriculum_medium_max_step", type=int, default=800)
    parser.add_argument("--curriculum_full_max_step", type=int, default=1000)
    parser.add_argument("--curriculum_easy_random_pos", type=str2bool, default=False)
    parser.add_argument("--curriculum_medium_random_pos", type=str2bool, default=False)
    parser.add_argument("--curriculum_full_random_pos", type=str2bool, default=True)

    args = parser.parse_args(argv)
    if args.chunk_len <= 0:
        raise ValueError("chunk_len must be > 0")
    if args.burn_in < 0:
        raise ValueError("burn_in must be >= 0")
    if args.rollout <= 0:
        raise ValueError("rollout must be > 0")
    if args.mode_interval <= 0:
        raise ValueError("mode_interval must be > 0")
    if not (0.0 <= args.curriculum_easy_frac <= 1.0):
        raise ValueError("curriculum_easy_frac must be in [0,1]")
    if not (0.0 <= args.curriculum_medium_frac <= 1.0):
        raise ValueError("curriculum_medium_frac must be in [0,1]")
    if args.curriculum_easy_frac > args.curriculum_medium_frac:
        raise ValueError("curriculum_easy_frac must be <= curriculum_medium_frac")
    if str(args.lr_schedule).lower() not in {"constant", "linear"}:
        raise ValueError("lr_schedule must be one of: constant, linear")
    if not (0.0 < float(args.lr_min_ratio) <= 1.0):
        raise ValueError("lr_min_ratio must be in (0,1]")
    if not (0.0 <= float(args.lr_second_half_start_frac) <= 1.0):
        raise ValueError("lr_second_half_start_frac must be in [0,1]")
    if not (0.0 < float(args.lr_second_half_ratio) <= 1.0):
        raise ValueError("lr_second_half_ratio must be in (0,1]")
    if str(args.best_checkpoint_source).lower() not in {"eval", "train", "eval_then_train"}:
        raise ValueError("best_checkpoint_source must be one of: eval, train, eval_then_train")
    if str(args.attack_rule_mode).lower() not in {"none", "nearest_target"}:
        raise ValueError("attack_rule_mode must be one of: none, nearest_target")
    if str(args.attack_policy_mode).lower() not in {"full_discrete", "fire_or_not"}:
        raise ValueError("attack_policy_mode must be one of: full_discrete, fire_or_not")
    if float(args.aux_value_loss_coeff) < 0.0:
        raise ValueError("aux_value_loss_coeff must be >= 0")
    if float(args.priority_aux_loss_coeff) < 0.0:
        raise ValueError("priority_aux_loss_coeff must be >= 0")
    if str(args.best_metric_name).lower() not in {"total_win_rate", "win_rate", "blue_alive_zero_rate"}:
        raise ValueError("best_metric_name must be one of: total_win_rate, win_rate, blue_alive_zero_rate")
    if float(args.maca_boundary_penalty_margin) <= 0.0:
        raise ValueError("maca_boundary_penalty_margin must be > 0")
    if float(args.maca_boundary_penalty_scale) < 0.0:
        raise ValueError("maca_boundary_penalty_scale must be >= 0")
    if int(args.maca_boundary_stuck_trigger_steps) <= 0:
        raise ValueError("maca_boundary_stuck_trigger_steps must be > 0")
    if int(args.maca_boundary_stuck_ramp_steps) <= 0:
        raise ValueError("maca_boundary_stuck_ramp_steps must be > 0")
    if int(args.maca_semantic_screen_downsample) <= 0:
        raise ValueError("maca_semantic_screen_downsample must be > 0")
    if float(args.maca_search_reward_scale) < 0.0:
        raise ValueError("maca_search_reward_scale must be >= 0")
    if float(args.maca_reacquire_reward_scale) < 0.0:
        raise ValueError("maca_reacquire_reward_scale must be >= 0")
    if int(args.maca_priority_grid_h) < 2 or int(args.maca_priority_grid_w) < 2:
        raise ValueError("maca_priority_grid_h and maca_priority_grid_w must be >= 2")
    if int(args.maca_priority_top_k) <= 0:
        raise ValueError("maca_priority_top_k must be > 0")
    if int(args.maca_priority_unseen_cap_steps) <= 0:
        raise ValueError("maca_priority_unseen_cap_steps must be > 0")
    if not (0.0 <= float(args.maca_priority_memory_decay) <= 1.0):
        raise ValueError("maca_priority_memory_decay must be in [0,1]")
    if not (0.0 <= float(args.maca_priority_diffusion_rate) <= 1.0):
        raise ValueError("maca_priority_diffusion_rate must be in [0,1]")
    if float(args.maca_priority_passive_recv_weight) < 0.0:
        raise ValueError("maca_priority_passive_recv_weight must be >= 0")
    if float(args.maca_priority_known_enemy_boost) < 0.0:
        raise ValueError("maca_priority_known_enemy_boost must be >= 0")
    if not (0.0 <= float(args.maca_priority_unseen_threshold) <= 1.0):
        raise ValueError("maca_priority_unseen_threshold must be in [0,1]")
    return args


def set_seed(seed: int):
    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def device_from_arg(device_arg: str):
    if device_arg == "gpu" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_teacher_policy(teacher_path: str, model, device):
    if not teacher_path:
        return None
    path = Path(teacher_path)
    if not path.exists():
        raise FileNotFoundError("Teacher checkpoint not found: %s" % teacher_path)
    payload = torch.load(path, map_location="cpu")
    teacher_model = TeamActorCritic(model.cfg).to(device)
    state = payload.get("model", payload)
    load_res = teacher_model.load_state_dict(state, strict=False)
    if load_res.missing_keys or load_res.unexpected_keys:
        print(
            "[teacher] non-strict load missing=%s unexpected=%s"
            % (load_res.missing_keys, load_res.unexpected_keys),
            flush=True,
        )
    teacher_model.eval()
    for p in teacher_model.parameters():
        p.requires_grad_(False)
    print("[teacher] loaded %s" % path, flush=True)
    return teacher_model


def curriculum_profile(args, env_steps: int):
    if not bool(args.curriculum_enabled):
        return {
            "name": "full",
            "id": 3,
            "opponent": args.maca_opponent,
            "max_step": int(args.maca_max_step),
            "random_pos": bool(args.maca_random_pos),
        }

    total = max(1, int(args.train_for_env_steps))
    frac = float(env_steps) / float(total)
    if frac < float(args.curriculum_easy_frac):
        return {
            "name": "easy",
            "id": 1,
            "opponent": args.curriculum_easy_opponent,
            "max_step": int(args.curriculum_easy_max_step),
            "random_pos": bool(args.curriculum_easy_random_pos),
        }
    if frac < float(args.curriculum_medium_frac):
        return {
            "name": "medium",
            "id": 2,
            "opponent": args.curriculum_medium_opponent,
            "max_step": int(args.curriculum_medium_max_step),
            "random_pos": bool(args.curriculum_medium_random_pos),
        }
    return {
        "name": "full",
        "id": 3,
        "opponent": args.curriculum_full_opponent,
        "max_step": int(args.curriculum_full_max_step),
        "random_pos": bool(args.curriculum_full_random_pos),
    }


def apply_curriculum_profile(args, profile):
    args.runtime_maca_opponent = str(profile["opponent"])
    args.runtime_maca_max_step = int(profile["max_step"])
    args.runtime_maca_random_pos = bool(profile["random_pos"])
    args.runtime_curriculum_name = str(profile["name"])
    args.runtime_curriculum_id = int(profile["id"])


def compute_scheduled_lr(args, env_steps: int) -> float:
    base_lr = float(args.learning_rate)
    if base_lr <= 0.0:
        return 0.0

    progress = min(1.0, max(0.0, float(env_steps) / float(max(1, int(args.train_for_env_steps)))))
    schedule = str(args.lr_schedule).lower()
    min_ratio = float(args.lr_min_ratio)

    if schedule == "constant":
        lr = base_lr
        start_lr = base_lr
    else:
        min_lr = base_lr * min_ratio
        lr = base_lr + (min_lr - base_lr) * progress
        start_frac = float(args.lr_second_half_start_frac)
        start_lr = base_lr + (min_lr - base_lr) * start_frac

    tail_start = float(args.lr_second_half_start_frac)
    tail_ratio = float(args.lr_second_half_ratio)
    if tail_ratio < 1.0 and progress >= tail_start and tail_start < 1.0:
        tail_progress = (progress - tail_start) / max(1e-6, 1.0 - tail_start)
        target_lr = base_lr * tail_ratio
        lr = start_lr + (target_lr - start_lr) * tail_progress

    return max(0.0, float(lr))


def set_optimizer_lr(optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = float(lr)


def resolve_metric(summary: dict, metric_name: str) -> float:
    key = str(metric_name).strip()
    aliases = {
        "win_rate": ["win_rate", "win_flag_mean"],
        "total_win_rate": ["total_win_rate", "total_win_flag_mean"],
        "blue_alive_zero_rate": ["blue_alive_zero_rate", "blue_alive_zero_flag_mean"],
    }
    for candidate in aliases.get(key, [key]):
        if candidate in summary:
            value = float(summary.get(candidate, float("nan")))
            if np.isfinite(value):
                return value
    return float("-inf")


def frozen_update_stats(returns: np.ndarray, values: np.ndarray, value_normalizer: ValueNormalizer):
    returns_np = np.asarray(returns, dtype=np.float32)
    values_np = np.asarray(values, dtype=np.float32)

    valid = np.isfinite(returns_np) & np.isfinite(values_np)
    explained_var = 0.0
    if np.any(valid):
        r = returns_np[valid]
        v = values_np[valid]
        var_r = float(np.var(r))
        if r.size > 1 and var_r > 1e-8:
            explained_var = float(1.0 - np.var(r - v) / var_r)

    norm_returns = value_normalizer.normalize(returns_np)
    return {
        "policy_loss": 0.0,
        "policy_loss_low": 0.0,
        "policy_loss_high": 0.0,
        "value_loss": 0.0,
        "entropy": 0.0,
        "entropy_low": 0.0,
        "entropy_high": 0.0,
        "imitation_loss": 0.0,
        "imitation_coef": 0.0,
        "active_samples": 0,
        "grad_steps": 0,
        "explained_variance": explained_var,
        "actor_grad_norm": 0.0,
        "critic_grad_norm": 0.0,
        "active_mask_ratio": 0.0,
        "hidden_state_init_abs_error": 0.0,
        "skipped_non_finite_batches": 0,
        "repaired_non_finite_params": 0,
        "value_target_mean": float(np.mean(returns_np)),
        "value_target_std": float(np.std(returns_np)),
        "value_target_norm_mean": float(np.mean(norm_returns)),
        "value_target_norm_std": float(np.std(norm_returns)),
    }


def main(argv=None):
    args = parse_args(argv)
    device = device_from_arg(args.device)
    set_seed(args.seed)

    stage_profile = curriculum_profile(args, env_steps=0)
    apply_curriculum_profile(args, stage_profile)

    writer = None
    probe_env = build_env(args, seed_offset=0)
    try:
        initial_obs = probe_env.reset(seed=args.seed)
        base_local_obs_dim = initial_obs["local_obs"].shape[1]
        local_screen_shape = tuple(initial_obs["local_screen"].shape[1:])
        global_state_dim = initial_obs["global_state"].shape[0]
        num_agents = initial_obs["local_obs"].shape[0]
        attack_dim = initial_obs["attack_masks"].shape[1]
        assigned_region_dim = int(initial_obs.get("assigned_region_obs", np.zeros((num_agents, 5), dtype=np.float32)).shape[1])
        priority_map_dim = initial_obs["priority_map_teacher"].shape[1]
        priority_grid_shape = tuple(probe_env.priority_grid_shape)
        priority_top_k = int(probe_env.priority_top_k)
    finally:
        probe_env.close()

    local_obs_dim = int(base_local_obs_dim)

    env_spec = {
        "local_obs_dim": int(local_obs_dim),
        "local_screen_shape": tuple(local_screen_shape),
        "global_state_dim": int(global_state_dim),
        "num_agents": int(num_agents),
        "attack_dim": int(attack_dim),
        "assigned_region_dim": int(assigned_region_dim),
        "priority_map_dim": int(priority_map_dim),
        "priority_grid_h": int(priority_grid_shape[0]),
        "priority_grid_w": int(priority_grid_shape[1]),
        "priority_top_k": int(priority_top_k),
        "actor_hidden_dim": int(args.hidden_size),
        "mode_interval": int(args.mode_interval),
    }

    model_cfg = MAPPOModelConfig(
        local_obs_dim=local_obs_dim,
        local_screen_shape=tuple(local_screen_shape),
        global_state_dim=global_state_dim,
        num_agents=num_agents,
        hidden_size=args.hidden_size,
        screen_embed_dim=args.screen_embed_dim,
        course_embed_dim=args.course_embed_dim,
        priority_grid_h=int(priority_grid_shape[0]),
        priority_grid_w=int(priority_grid_shape[1]),
        priority_top_k=int(priority_top_k),
    )
    model = TeamActorCritic(model_cfg).to(device)
    teacher_model = None
    if args.teacher_bc_checkpoint:
        teacher_model = load_teacher_policy(args.teacher_bc_checkpoint, model, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    value_normalizer = ValueNormalizer()

    env_steps = 0
    update_idx = 0
    if args.resume:
        env_steps, update_idx = load_checkpoint(args, model, optimizer, value_normalizer, root_dir=ROOT_DIR)

    set_optimizer_lr(optimizer, compute_scheduled_lr(args, env_steps))

    shared_actor = allocate_shared_actor_params(model)
    write_shared_actor_params(model, shared_actor)

    collectors = CollectorPool(args, env_spec, shared_actor)
    collectors.reset(seed=args.seed)

    save_config(args, local_obs_dim, global_state_dim, num_agents, collectors.worker_env_counts)
    print(
        "[collector] num_workers=%d num_envs=%d worker_env_counts=%s base_obs_dim=%d local_screen_shape=%s priority_grid=%s priority_top_k=%d curriculum_stage=%s attack_rule_mode=%s attack_policy_mode=%s disable_high_level_mode=%s"
        % (
            collectors.num_workers,
            collectors.num_envs,
            collectors.worker_env_counts,
            int(base_local_obs_dim),
            str(tuple(local_screen_shape)),
            str(tuple(priority_grid_shape)),
            int(priority_top_k),
            str(getattr(args, "runtime_curriculum_name", "full")),
            str(args.attack_rule_mode),
            str(args.attack_policy_mode),
            str(bool(args.disable_high_level_mode)),
        ),
        flush=True,
    )

    writer = build_summary_writer(args, purge_step=env_steps if args.resume else None)

    best_metric_name = str(args.best_metric_name)
    best_metric = load_best_metric(args, metric_name=best_metric_name) if bool(args.save_best_checkpoint) else float("-inf")
    if bool(args.save_best_checkpoint):
        print("[checkpoint] best metric init %s=%.4f" % (best_metric_name, float(best_metric)), flush=True)

    last_log_time = time.time()
    last_save_time = time.time()
    if args.eval_every_env_steps > 0:
        interval = int(args.eval_every_env_steps)
        next_eval_env_steps = ((env_steps // interval) + 1) * interval
    else:
        next_eval_env_steps = -1
    episode_stats = deque(maxlen=100)

    try:
        while env_steps < args.train_for_env_steps:
            next_profile = curriculum_profile(args, env_steps=env_steps)
            if str(next_profile["name"]) != str(getattr(args, "runtime_curriculum_name", "full")):
                apply_curriculum_profile(args, next_profile)
                collectors.close()
                collectors = CollectorPool(args, env_spec, shared_actor)
                collectors.reset(seed=args.seed + env_steps + update_idx * 17)
                print(
                    "[curriculum] switched stage=%s id=%d opponent=%s max_step=%d random_pos=%s"
                    % (
                        args.runtime_curriculum_name,
                        int(args.runtime_curriculum_id),
                        args.runtime_maca_opponent,
                        int(args.runtime_maca_max_step),
                        str(bool(args.runtime_maca_random_pos)),
                    ),
                    flush=True,
                )

            update_start = time.time()
            print(
                "[stage] update=%d env_steps=%d starting_rollout rollout=%d num_envs=%d workers=%d"
                % (update_idx + 1, env_steps, args.rollout, collectors.num_envs, collectors.num_workers),
                flush=True,
            )

            actor_sync_start = time.time()
            write_shared_actor_params(model, shared_actor)
            actor_sync_time = time.time() - actor_sync_start

            buffer, next_value, rollout_diag, rollout_timing = rollout(
                collectors, model, device, args, episode_stats, value_normalizer
            )
            rollout_wall_time = time.time() - update_start
            print(
                "[stage] update=%d env_steps=%d rollout_finished wall_time_sec=%.2f"
                % (update_idx + 1, env_steps, rollout_wall_time),
                flush=True,
            )

            gae_start = time.time()
            advantages_team, returns = compute_gae(buffer, next_value, args.gamma, args.gae_lambda)
            advantages_aux = compute_aux_advantages(buffer, args.gamma)
            gae_time = time.time() - gae_start

            print(
                "[stage] update=%d env_steps=%d starting_ppo_update"
                % (update_idx + 1, env_steps),
                flush=True,
            )

            current_lr = compute_scheduled_lr(args, env_steps)
            set_optimizer_lr(optimizer, current_lr)
            freeze_update = int(args.debug_freeze_update_after_env_steps) >= 0 and env_steps >= int(
                args.debug_freeze_update_after_env_steps
            )

            ppo_start = time.time()
            if freeze_update:
                stats = frozen_update_stats(returns, buffer["value"], value_normalizer)
                print(
                    "[debug] update frozen env_steps=%d threshold=%d"
                    % (env_steps, int(args.debug_freeze_update_after_env_steps)),
                    flush=True,
                )
            else:
                stats = ppo_update(
                    model,
                    optimizer,
                    buffer,
                    advantages_team,
                    advantages_aux,
                    returns,
                    device,
                    args,
                    value_normalizer,
                    teacher_model=teacher_model,
                    update_idx=update_idx,
                )

            freeze_vn_after = int(getattr(args, "freeze_value_normalizer_after_env_steps", -1))
            value_norm_updated = False
            if freeze_vn_after < 0 or env_steps < freeze_vn_after:
                value_normalizer.update(returns)
                value_norm_updated = True

            stats["value_norm_mean"] = float(value_normalizer.mean)
            stats["value_norm_std"] = float(np.sqrt(max(float(value_normalizer.var), 0.0) + float(value_normalizer.epsilon)))
            stats["value_norm_updated"] = 1.0 if value_norm_updated else 0.0
            ppo_update_time = time.time() - ppo_start
            total_update_time = time.time() - update_start

            timing_stats = {
                "actor_sync_time": float(actor_sync_time),
                "rollout_collect_time": float(rollout_timing.get("rollout_collect_time", 0.0)),
                "rollout_postprocess_time": float(rollout_timing.get("rollout_postprocess_time", 0.0)),
                "value_bootstrap_time": float(rollout_timing.get("value_bootstrap_time", 0.0)),
                "gae_time": float(gae_time),
                "ppo_update_time": float(ppo_update_time),
                "total_update_time": float(total_update_time),
            }

            action_stats = action_distribution_stats(
                course_actions=buffer["course_action"],
                attack_actions=buffer["attack_action"],
                alive_mask=buffer["alive_mask"],
                attack_masks=buffer["attack_masks"].astype(bool, copy=False),
                course_dim=model_cfg.course_dim,
                attack_dim=model_cfg.attack_dim,
                policy_attack_actions=buffer.get("policy_attack_action", None),
                attack_policy_mode=str(args.attack_policy_mode),
                attack_rule_mode=str(args.attack_rule_mode),
            )
            print(
                "[stage] update=%d env_steps=%d ppo_update_finished"
                % (update_idx + 1, env_steps),
                flush=True,
            )

            env_steps += args.rollout * collectors.num_envs
            update_idx += 1
            sample_fps = (args.rollout * collectors.num_envs) / max(total_update_time, 1e-6)

            print(
                "[timing] update=%d env_steps=%d actor_sync=%.4fs collect=%.4fs post=%.4fs bootstrap=%.4fs gae=%.4fs ppo=%.4fs total=%.4fs"
                % (
                    update_idx,
                    env_steps,
                    timing_stats["actor_sync_time"],
                    timing_stats["rollout_collect_time"],
                    timing_stats["rollout_postprocess_time"],
                    timing_stats["value_bootstrap_time"],
                    timing_stats["gae_time"],
                    timing_stats["ppo_update_time"],
                    timing_stats["total_update_time"],
                ),
                flush=True,
            )

            if time.time() - last_log_time >= args.log_every_sec:
                summary = summarize_episode_stats(list(episode_stats))
                reward_mean = summary.get("round_reward_mean", 0.0)
                win_rate = summary.get("win_rate", 0.0)
                stats["total_win_rate"] = float(summary.get("total_win_rate", 0.0))
                stats["blue_alive_zero_rate"] = float(summary.get("blue_alive_zero_rate", 0.0))
                stats["timeout_rate"] = float(summary.get("timeout_rate", 0.0))
                boundary_penalty_mean = float(summary.get("boundary_penalty_mean", 0.0))
                near_boundary_frac = float(summary.get("near_boundary_frac_mean", 0.0))
                damage_reward_mean = float(rollout_diag.get("damage_reward_mean", 0.0))
                log_summary(writer, "train_episode", summary, env_steps)
                log_train_scalars(
                    writer,
                    env_steps=env_steps,
                    stats=stats,
                    rollout_diag=rollout_diag,
                    action_stats=action_stats,
                    sample_fps=sample_fps,
                    timing_stats=timing_stats,
                    stage_id=int(getattr(args, "runtime_curriculum_id", 3)),
                    current_lr=current_lr,
                    reward_mean=reward_mean,
                    win_rate=win_rate,
                    attack_rule_mode=str(args.attack_rule_mode),
                    attack_policy_mode=str(args.attack_policy_mode),
                    disable_high_level_mode=bool(args.disable_high_level_mode),
                )
                print(
                    "[train] env_steps=%d update=%d stage=%s reward_mean=%.2f win_rate=%.3f total_win_rate=%.3f blue_alive_zero_rate=%.3f timeout_rate=%.3f boundary_penalty_mean=%.4f near_boundary_frac=%.3f fps=%.1f lr=%.6g attack_rule_mode=%s attack_policy_mode=%s disable_high_level_mode=%d policy=%.4f low=%.4f high=%.4f value_loss=%.4f v_team=%.4f v_aux=%.4f p_aux=%.4f entropy=%.4f imitation=%.4f coef=%.4f ev=%.4f actor_gn=%.4f critic_gn=%.4f active=%.3f skipped_nf=%d repaired_nf=%d hidden_err=%.6f rnn_mismatch=%d grad_steps=%d"
                    % (
                        env_steps,
                        update_idx,
                        str(getattr(args, "runtime_curriculum_name", "full")),
                        reward_mean,
                        win_rate,
                        stats.get("total_win_rate", 0.0),
                        stats.get("blue_alive_zero_rate", 0.0),
                        stats.get("timeout_rate", 0.0),
                        boundary_penalty_mean,
                        near_boundary_frac,
                        sample_fps,
                        current_lr,
                        str(args.attack_rule_mode),
                        str(args.attack_policy_mode),
                        int(bool(args.disable_high_level_mode)),
                        stats.get("policy_loss", 0.0),
                        stats.get("policy_loss_low", 0.0),
                        stats.get("policy_loss_high", 0.0),
                        stats.get("value_loss", 0.0),
                        stats.get("value_team_loss", 0.0),
                        stats.get("value_aux_loss", 0.0),
                        stats.get("priority_aux_loss", 0.0),
                        stats.get("entropy", 0.0),
                        stats.get("imitation_loss", 0.0),
                        stats.get("imitation_coef", 0.0),
                        stats.get("explained_variance", 0.0),
                        stats.get("actor_grad_norm", 0.0),
                        stats.get("critic_grad_norm", 0.0),
                        stats.get("active_mask_ratio", 0.0),
                        int(stats.get("skipped_non_finite_batches", 0)),
                        int(stats.get("repaired_non_finite_params", 0)),
                        stats.get("hidden_state_init_abs_error", 0.0),
                        int(rollout_diag.get("rnn_hidden_mismatch_count", 0)),
                        int(stats.get("grad_steps", 0)),
                    ),
                    flush=True,
                )
                print(
                    "[stability] env_steps=%d update=%d lr=%.6g attack_rule_mode=%s attack_policy_mode=%s disable_high_level_mode=%d actor_gn=%.4f critic_gn=%.4f value_loss=%.4f v_team=%.4f v_contact=%.4f v_opp=%.4f v_surv=%.4f aux_v_coef=%.3f aux_v_off=%d policy_loss=%.4f entropy=%.4f value_target_std=%.4f value_target_norm_std=%.4f value_norm_std=%.4f value_norm_updated=%d active_mask_ratio=%.4f reward_mean=%.2f win_rate=%.3f total_win_rate=%.3f blue_alive_zero_rate=%.3f timeout_rate=%.3f boundary_penalty_mean=%.4f near_boundary_frac=%.3f damage_reward_mean=%.4f attack_opportunity_frac=%.4f executed_fire_action_frac=%.4f no_fire_when_legal_frac=%.4f opportunity_to_fire_ratio=%.4f fire_decision_freq_00=%.4f fire_decision_freq_01=%.4f rule_selected_attack_nonzero_freq=%.4f freeze_update=%d"
                    % (
                        env_steps,
                        update_idx,
                        current_lr,
                        str(args.attack_rule_mode),
                        str(args.attack_policy_mode),
                        int(bool(args.disable_high_level_mode)),
                        stats.get("actor_grad_norm", 0.0),
                        stats.get("critic_grad_norm", 0.0),
                        stats.get("value_loss", 0.0),
                        stats.get("value_team_loss", 0.0),
                        stats.get("value_contact_loss", 0.0),
                        stats.get("value_opportunity_loss", 0.0),
                        stats.get("value_survival_loss", 0.0),
                        stats.get("aux_value_loss_coeff", 0.0),
                        int(stats.get("disable_aux_value_heads", 0.0) > 0.5),
                        stats.get("policy_loss", 0.0),
                        stats.get("entropy", 0.0),
                        stats.get("value_target_std", 0.0),
                        stats.get("value_target_norm_std", 0.0),
                        stats.get("value_norm_std", 0.0),
                        int(stats.get("value_norm_updated", 0.0) > 0.5),
                        stats.get("active_mask_ratio", 0.0),
                        reward_mean,
                        win_rate,
                        stats.get("total_win_rate", 0.0),
                        stats.get("blue_alive_zero_rate", 0.0),
                        stats.get("timeout_rate", 0.0),
                        boundary_penalty_mean,
                        near_boundary_frac,
                        damage_reward_mean,
                        action_stats.get("attack_opportunity_frac", 0.0),
                        action_stats.get("executed_fire_action_frac", 0.0),
                        action_stats.get("no_fire_when_legal_frac", 0.0),
                        action_stats.get("opportunity_to_fire_ratio", 0.0),
                        action_stats.get("fire_decision_freq_00", 0.0),
                        action_stats.get("fire_decision_freq_01", 0.0),
                        action_stats.get("rule_selected_attack_nonzero_freq", 0.0),
                        int(freeze_update),
                    ),
                    flush=True,
                )

                if bool(args.save_best_checkpoint) and str(args.best_checkpoint_source).lower() in {
                    "train",
                    "eval_then_train",
                }:
                    train_metric = resolve_metric(summary, best_metric_name)
                    if np.isfinite(train_metric) and train_metric > float(best_metric):
                        if save_best_checkpoint(
                            args,
                            model,
                            optimizer,
                            env_steps,
                            update_idx,
                            metric_value=train_metric,
                            metric_name=best_metric_name,
                            source="train",
                            value_normalizer=value_normalizer,
                        ):
                            best_metric = train_metric
                last_log_time = time.time()

            if time.time() - last_save_time >= args.save_every_sec:
                save_checkpoint(args, model, optimizer, env_steps, update_idx, value_normalizer)
                last_save_time = time.time()

            if args.eval_episodes > 0 and args.eval_every_env_steps > 0 and env_steps >= next_eval_env_steps:
                payload, output_path = run_evaluation(model, device, args, env_steps)
                eval_summary = payload.get("summary", {})
                log_summary(writer, "eval", eval_summary, env_steps)
                if writer is not None:
                    writer.add_scalar("eval/eval_wall_time_sec", float(payload["eval_wall_time_sec"]), env_steps)
                    writer.flush()
                print(
                    "[eval] env_steps=%d win_rate=%.3f total_win_rate=%.3f blue_alive_zero_rate=%.3f timeout_rate=%.3f destroy_balance=%.3f red_down=%.3f blue_down=%.3f saved=%s"
                    % (
                        env_steps,
                        float(eval_summary.get("win_rate", 0.0)),
                        float(eval_summary.get("total_win_rate", 0.0)),
                        float(eval_summary.get("blue_alive_zero_rate", 0.0)),
                        float(eval_summary.get("timeout_rate", 0.0)),
                        float(eval_summary.get("fighter_destroy_balance_end_mean", 0.0)),
                        float(eval_summary.get("red_fighter_destroyed_end_mean", 0.0)),
                        float(eval_summary.get("blue_fighter_destroyed_end_mean", 0.0)),
                        output_path,
                    ),
                    flush=True,
                )
                if bool(args.save_best_checkpoint) and str(args.best_checkpoint_source).lower() in {
                    "eval",
                    "eval_then_train",
                }:
                    eval_metric = resolve_metric(eval_summary, best_metric_name)
                    if np.isfinite(eval_metric) and eval_metric > float(best_metric):
                        if save_best_checkpoint(
                            args,
                            model,
                            optimizer,
                            env_steps,
                            update_idx,
                            metric_value=eval_metric,
                            metric_name=best_metric_name,
                            source="eval",
                            value_normalizer=value_normalizer,
                        ):
                            best_metric = eval_metric
                next_eval_env_steps += int(args.eval_every_env_steps)

        save_checkpoint(args, model, optimizer, env_steps, update_idx, value_normalizer)
    finally:
        collectors.close()
        if writer is not None:
            writer.close()


if __name__ == "__main__":
    main()
