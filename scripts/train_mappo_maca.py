#!/usr/bin/env python
"""Train MaCA with recurrent MAPPO (chunked PPO + burn-in)."""

from __future__ import annotations

import argparse
import ctypes
import json
import multiprocessing as mp
import random
import sys
import time
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter

ROOT_DIR = Path(__file__).resolve().parent.parent
ENV_DIR = ROOT_DIR / "environment"
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(ENV_DIR) not in sys.path:
    sys.path.insert(0, str(ENV_DIR))

from marl_env.mappo_env import MAPPOMaCAConfig, MAPPOMaCAEnv
from marl_env.mappo_model import MAPPOModelConfig, TeamActorCritic


ROLLOUT_BUFFER_DTYPES = {
    "local_obs": np.float32,
    "global_state": np.float32,
    "agent_ids": np.int64,
    "attack_masks": np.uint8,
    "alive_mask": np.float32,
    "actor_h": np.float32,
    "course_action": np.int64,
    "attack_action": np.int64,
    "mode_action": np.int64,
    "log_prob": np.float32,
    "mode_log_prob": np.float32,
    "mode_decision": np.float32,
    "mode_duration": np.int64,
    "reward": np.float32,
    "reward_env": np.float32,
    "reward_mode": np.float32,
    "reward_exec": np.float32,
    "damage_reward": np.float32,
    "kill_reward": np.float32,
    "survival_reward": np.float32,
    "win_indicator": np.float32,
    "done": np.float32,
    "agent_aux_reward": np.float32,
}


class ValueNormalizer:
    """Running mean/variance normalizer for value targets (PopArt-lite)."""

    def __init__(self, clip=10.0, epsilon=1e-6):
        self.mean = 0.0
        self.var = 1.0
        self.count = epsilon
        self.clip = clip
        self.epsilon = epsilon

    def update(self, values):
        values = np.asarray(values, dtype=np.float64).ravel()
        values = values[np.isfinite(values)]
        if values.size <= 0:
            return
        # Keep running stats bounded to avoid float overflow poisoning.
        values = np.clip(values, -1e9, 1e9)
        batch_mean = float(np.mean(values))
        batch_var = float(np.var(values))
        batch_count = float(values.size)
        delta = batch_mean - self.mean
        total_count = self.count + batch_count
        self.mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        self.var = (m_a + m_b + delta ** 2 * self.count * batch_count / total_count) / total_count
        self.count = total_count

    def normalize(self, values):
        std = np.sqrt(max(float(self.var), 0.0) + self.epsilon)
        base = np.asarray(values, dtype=np.float64)
        norm = (base - self.mean) / max(std, self.epsilon)
        norm = np.where(np.isfinite(norm), norm, 0.0)
        return np.clip(norm, -self.clip, self.clip)

    def denormalize(self, values):
        std = np.sqrt(max(float(self.var), 0.0) + self.epsilon)
        base = np.asarray(values, dtype=np.float64)
        denorm = base * max(std, self.epsilon) + self.mean
        return np.where(np.isfinite(denorm), denorm, 0.0)

    def state_dict(self):
        return {"mean": self.mean, "var": self.var, "count": self.count}

    def load_state_dict(self, state):
        self.mean = float(state.get("mean", 0.0))
        self.var = float(state.get("var", 1.0))
        self.count = float(state.get("count", self.epsilon))


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

    # P0: recurrent PPO with sequence chunks + burn-in
    parser.add_argument("--rollout", type=int, default=64)
    parser.add_argument("--chunk_len", type=int, default=16)
    parser.add_argument("--burn_in", type=int, default=8)
    parser.add_argument("--train_for_env_steps", type=int, default=20_000_000)

    parser.add_argument("--ppo_epochs", type=int, default=5)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae_lambda", type=float, default=0.95)
    parser.add_argument("--clip_ratio", type=float, default=0.1)
    parser.add_argument("--value_loss_coeff", type=float, default=0.5)
    parser.add_argument("--entropy_coeff", type=float, default=0.01)
    parser.add_argument("--max_grad_norm", type=float, default=5.0)
    parser.add_argument("--num_mini_batches", type=int, default=4)

    # P3: hierarchical control (high-level mode policy).
    parser.add_argument("--mode_interval", type=int, default=8)
    parser.add_argument("--high_level_loss_coeff", type=float, default=0.25)
    parser.add_argument("--high_level_entropy_coeff", type=float, default=0.001)
    parser.add_argument("--teacher_bc_checkpoint", type=str, default="")
    parser.add_argument("--imitation_coef", type=float, default=0.0)
    parser.add_argument("--imitation_warmup_updates", type=int, default=0)

    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--role_embed_dim", type=int, default=8)
    parser.add_argument("--course_embed_dim", type=int, default=16)

    parser.add_argument("--save_every_sec", type=int, default=900)
    parser.add_argument("--log_every_sec", type=int, default=30)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--resume_checkpoint", type=str, default="")
    parser.add_argument("--resume_env_steps", type=int, default=-1)
    parser.add_argument("--tensorboard", type=str2bool, default=True)
    parser.add_argument("--eval_every_env_steps", type=int, default=1_000_000)
    parser.add_argument("--eval_episodes", type=int, default=20)
    parser.add_argument("--eval_deterministic", type=str2bool, default=True)
    parser.add_argument("--eval_opponent", type=str, default=None)

    # P1: team + agent-aware credit blending
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

    # P1: dynamics features + agent aux reward knobs
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

    # Diagnostics / observation plumbing
    parser.add_argument("--concat_agent_id_onehot", type=str2bool, default=True)

    # P7: curriculum learning switches.
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
    return args


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def device_from_arg(device_arg: str):
    if device_arg == "gpu" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def experiment_dir(args) -> Path:
    return Path(args.train_dir) / args.experiment


def checkpoint_dir(args) -> Path:
    return experiment_dir(args) / "checkpoint"


def eval_dir(args) -> Path:
    return experiment_dir(args) / "eval"


def latest_checkpoint(args) -> Optional[Path]:
    ckpt_dir = checkpoint_dir(args)
    if not ckpt_dir.exists():
        return None
    checkpoints = sorted(ckpt_dir.glob("checkpoint_*.pt"))
    return checkpoints[-1] if checkpoints else None


def checkpoint_env_steps_from_name(path: Path) -> Optional[int]:
    stem = path.stem
    parts = stem.split("_")
    if len(parts) < 3:
        return None
    try:
        return int(parts[-1])
    except ValueError:
        return None


def resolve_resume_checkpoint(args) -> Optional[Path]:
    if args.resume_checkpoint:
        ckpt_path = Path(args.resume_checkpoint)
        if not ckpt_path.is_absolute():
            ckpt_path = ROOT_DIR / ckpt_path
        if not ckpt_path.exists():
            raise FileNotFoundError("resume checkpoint not found: %s" % ckpt_path)
        return ckpt_path

    if int(getattr(args, "resume_env_steps", -1)) >= 0:
        ckpt_dir = checkpoint_dir(args)
        if not ckpt_dir.exists():
            return None

        target = int(args.resume_env_steps)
        candidates = []
        for p in ckpt_dir.glob("checkpoint_*.pt"):
            env_s = checkpoint_env_steps_from_name(p)
            if env_s is None:
                continue
            candidates.append((env_s, p))

        if not candidates:
            return None

        candidates.sort(key=lambda x: x[0])
        exact = [item for item in candidates if item[0] == target]
        if exact:
            return exact[-1][1]

        le = [item for item in candidates if item[0] <= target]
        if le:
            chosen = le[-1]
            print(
                "[checkpoint] resume_env_steps=%d not exact, fallback to <= target checkpoint env_steps=%d path=%s"
                % (target, chosen[0], chosen[1]),
                flush=True,
            )
            return chosen[1]

        chosen = candidates[0]
        print(
            "[checkpoint] resume_env_steps=%d below all checkpoints, fallback to earliest env_steps=%d path=%s"
            % (target, chosen[0], chosen[1]),
            flush=True,
        )
        return chosen[1]

    return latest_checkpoint(args)


def save_config(args, local_obs_dim: int, global_state_dim: int, num_agents: int, worker_env_counts: List[int]):
    exp_dir = experiment_dir(args)
    exp_dir.mkdir(parents=True, exist_ok=True)
    cfg = vars(args).copy()
    cfg["local_obs_dim"] = int(local_obs_dim)
    cfg["global_state_dim"] = int(global_state_dim)
    cfg["num_agents"] = int(num_agents)
    cfg["worker_env_counts"] = [int(v) for v in worker_env_counts]
    cfg["saved_at_unix"] = float(time.time())
    (exp_dir / "cfg.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=2))


def save_checkpoint(args, model, optimizer, env_steps, update_idx, value_normalizer=None):
    ckpt_dir = checkpoint_dir(args)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    path = ckpt_dir / ("checkpoint_%09d_%d.pt" % (update_idx, env_steps))
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "env_steps": int(env_steps),
        "update_idx": int(update_idx),
    }
    if value_normalizer is not None:
        payload["value_normalizer"] = value_normalizer.state_dict()
    torch.save(payload, path)
    print("[checkpoint] saved %s" % path, flush=True)


def load_checkpoint(args, model, optimizer, value_normalizer=None):
    ckpt_path = resolve_resume_checkpoint(args)
    if ckpt_path is None:
        return 0, 0
    state = torch.load(ckpt_path, map_location="cpu")
    model_load = model.load_state_dict(state["model"], strict=False)
    if model_load.missing_keys or model_load.unexpected_keys:
        print(
            "[checkpoint] non-strict load missing=%s unexpected=%s"
            % (model_load.missing_keys, model_load.unexpected_keys),
            flush=True,
        )
    optimizer_state = state.get("optimizer", None)
    if optimizer_state is not None:
        try:
            optimizer.load_state_dict(optimizer_state)
        except Exception as exc:
            print("[checkpoint] optimizer state skipped: %s" % exc, flush=True)
    # Always enforce current run LR so resume can intentionally lower risk.
    target_lr = float(getattr(args, "learning_rate", 0.0) or 0.0)
    if target_lr > 0.0:
        for group in optimizer.param_groups:
            group["lr"] = target_lr
        print("[checkpoint] optimizer lr set to %.6g" % target_lr, flush=True)
    if value_normalizer is not None and "value_normalizer" in state:
        value_normalizer.load_state_dict(state["value_normalizer"])
        print("[checkpoint] value normalizer restored", flush=True)
    print("[checkpoint] resumed from %s" % ckpt_path, flush=True)
    return int(state.get("env_steps", 0)), int(state.get("update_idx", 0))


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


def build_summary_writer(args, purge_step: Optional[int] = None):
    if not args.tensorboard:
        return None
    log_dir = experiment_dir(args) / "tb"
    log_dir.mkdir(parents=True, exist_ok=True)
    if purge_step is not None and int(purge_step) >= 0:
        print("[tensorboard] log_dir=%s purge_step=%d" % (log_dir, int(purge_step)), flush=True)
        return SummaryWriter(log_dir=str(log_dir), purge_step=int(purge_step))
    return SummaryWriter(log_dir=str(log_dir))

def masked_categorical(logits: torch.Tensor, masks: torch.Tensor) -> Categorical:
    invalid_logit = torch.finfo(logits.dtype).min
    masked_logits = logits.masked_fill(~masks, invalid_logit)
    return Categorical(logits=masked_logits)


def append_agent_id_onehot(local_obs: np.ndarray, agent_ids: np.ndarray, num_agents: int) -> np.ndarray:
    if local_obs.ndim != 3:
        raise ValueError("Expected local_obs shape [env, agent, dim], got %s" % (local_obs.shape,))
    one_hot = np.eye(num_agents, dtype=np.float32)[agent_ids]
    return np.concatenate([local_obs.astype(np.float32, copy=False), one_hot], axis=-1)


def append_agent_id_onehot_torch(local_obs: torch.Tensor, agent_ids: torch.Tensor, num_agents: int) -> torch.Tensor:
    one_hot = torch.nn.functional.one_hot(agent_ids, num_classes=num_agents).to(dtype=local_obs.dtype)
    return torch.cat([local_obs, one_hot], dim=-1)


def tensor_explained_variance(returns_t: torch.Tensor, values_t: torch.Tensor) -> float:
    returns_flat = returns_t.reshape(-1)
    values_flat = values_t.reshape(-1)
    valid = torch.isfinite(returns_flat) & torch.isfinite(values_flat)
    if not torch.any(valid):
        return 0.0
    returns_flat = returns_flat[valid]
    values_flat = values_flat[valid]
    if returns_flat.numel() <= 1:
        return 0.0
    var_returns = torch.var(returns_flat, unbiased=False)
    if float(var_returns.item()) < 1e-8:
        return 0.0
    var_res = torch.var(returns_flat - values_flat, unbiased=False)
    return float((1.0 - var_res / var_returns).item())


def global_grad_norm(parameters) -> float:
    total = 0.0
    for p in parameters:
        if p.grad is None:
            continue
        grad_norm = p.grad.data.norm(2)
        total += float(grad_norm.item()) ** 2
    return float(total ** 0.5)


def sanitize_tensor(t: torch.Tensor, clamp_abs: float = 1e6) -> torch.Tensor:
    t = torch.nan_to_num(t, nan=0.0, posinf=clamp_abs, neginf=-clamp_abs)
    return torch.clamp(t, -clamp_abs, clamp_abs)


def sanitize_logits(logits: torch.Tensor, clamp_abs: float = 30.0) -> torch.Tensor:
    logits = torch.nan_to_num(logits, nan=0.0, posinf=clamp_abs, neginf=-clamp_abs)
    return torch.clamp(logits, -clamp_abs, clamp_abs)


def ensure_valid_action_mask(masks: torch.Tensor) -> torch.Tensor:
    # Categorical requires at least one valid action in each row.
    if masks.numel() <= 0:
        return masks
    has_any = torch.any(masks, dim=-1, keepdim=True)
    safe_masks = masks.clone()
    safe_masks[..., 0] = safe_masks[..., 0] | (~has_any.squeeze(-1))
    return safe_masks


def repair_non_finite_parameters(model: nn.Module, clamp_abs: float = 1e3) -> int:
    repaired = 0
    with torch.no_grad():
        for p in model.parameters():
            if p is None:
                continue
            bad = ~torch.isfinite(p)
            bad_count = int(bad.sum().item())
            if bad_count > 0:
                repaired += bad_count
                fixed = torch.nan_to_num(p, nan=0.0, posinf=clamp_abs, neginf=-clamp_abs)
                p.copy_(torch.clamp(fixed, -clamp_abs, clamp_abs))
    return repaired


def has_non_finite_gradients(model: nn.Module) -> bool:
    for p in model.parameters():
        if p.grad is None:
            continue
        if not torch.all(torch.isfinite(p.grad)):
            return True
    return False


def sample_actions(model, batch, device, deterministic: bool = False):
    local_obs = torch.as_tensor(batch["local_obs"], dtype=torch.float32, device=device)
    agent_ids = torch.as_tensor(batch["agent_ids"], dtype=torch.long, device=device)
    attack_masks = torch.as_tensor(batch["attack_masks"], dtype=torch.bool, device=device)
    actor_h = torch.as_tensor(batch["actor_h"], dtype=torch.float32, device=device)

    flat_local = local_obs.reshape(-1, local_obs.shape[-1])
    flat_ids = agent_ids.reshape(-1)
    flat_attack_masks = attack_masks.reshape(-1, attack_masks.shape[-1])
    flat_attack_masks = ensure_valid_action_mask(flat_attack_masks)
    flat_actor_h = actor_h.reshape(-1, actor_h.shape[-1])
    mode_actions = None
    if "mode_action" in batch:
        mode_actions = torch.as_tensor(batch["mode_action"], dtype=torch.long, device=device)
        mode_actions = mode_actions.reshape(-1)

    # Sequential two-head sampling: sample course first, then condition attack.
    course_logits, _attack_logits_unused, next_actor_h = model.actor_step(
        flat_local,
        flat_ids,
        flat_actor_h,
        mode_actions=mode_actions,
    )
    course_logits = sanitize_logits(course_logits)
    course_dist = Categorical(logits=course_logits)
    if deterministic:
        course_action = torch.argmax(course_logits, dim=-1)
    else:
        course_action = course_dist.sample()

    attack_logits = model.attack_logits(next_actor_h, course_action, mode_actions=mode_actions)
    attack_logits = sanitize_logits(attack_logits)
    attack_dist = masked_categorical(attack_logits, flat_attack_masks)
    if deterministic:
        attack_action = torch.argmax(attack_dist.logits, dim=-1)
    else:
        attack_action = attack_dist.sample()

    log_prob = course_dist.log_prob(course_action) + attack_dist.log_prob(attack_action)
    entropy = course_dist.entropy() + attack_dist.entropy()

    course_action = course_action.reshape(local_obs.shape[0], local_obs.shape[1])
    attack_action = attack_action.reshape(local_obs.shape[0], local_obs.shape[1])
    log_prob = log_prob.reshape(local_obs.shape[0], local_obs.shape[1])
    entropy = entropy.reshape(local_obs.shape[0], local_obs.shape[1])
    next_actor_h = next_actor_h.reshape(local_obs.shape[0], local_obs.shape[1], -1)

    return {
        "course_action": course_action,
        "attack_action": attack_action,
        "log_prob": log_prob,
        "entropy": entropy,
        "next_actor_h": next_actor_h,
    }


def export_actor_state_cpu(model) -> dict:
    actor_state = {}
    for key, value in model.state_dict().items():
        if key.startswith("critic"):
            continue
        actor_state[key] = value.detach().cpu().numpy()
    return actor_state


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


def build_env(args, seed_offset: int) -> MAPPOMaCAEnv:
    runtime_opponent = str(getattr(args, "runtime_maca_opponent", args.maca_opponent))
    runtime_max_step = int(getattr(args, "runtime_maca_max_step", args.maca_max_step))
    runtime_random_pos = bool(getattr(args, "runtime_maca_random_pos", args.maca_random_pos))
    return MAPPOMaCAEnv(
        MAPPOMaCAConfig(
            map_path=args.maca_map_path,
            red_obs_ind=args.maca_red_obs_ind,
            opponent=runtime_opponent,
            max_step=runtime_max_step,
            render=args.maca_render,
            random_pos=runtime_random_pos,
            random_seed=args.seed + seed_offset,
            adaptive_support_policy=args.maca_adaptive_support_policy,
            support_search_hold=args.maca_support_search_hold,
            delta_course_action=args.maca_delta_course_action,
            course_delta_deg=args.maca_course_delta_deg,
            max_visible_enemies=args.maca_max_visible_enemies,
            friendly_attrition_penalty=args.maca_friendly_attrition_penalty,
            enemy_attrition_reward=args.maca_enemy_attrition_reward,
            track_memory_steps=args.maca_track_memory_steps,
            contact_reward=args.maca_contact_reward,
            progress_reward_scale=args.maca_progress_reward_scale,
            progress_reward_cap=args.maca_progress_reward_cap,
            attack_window_reward=args.maca_attack_window_reward,
            agent_aux_reward_scale=args.maca_agent_aux_reward_scale,
            mode_reward_scale=args.maca_mode_reward_scale,
            exec_reward_scale=args.maca_exec_reward_scale,
            disengage_penalty=args.maca_disengage_penalty,
            bearing_reward_scale=args.maca_bearing_reward_scale,
        )
    )


def make_shared_ndarray(shape, dtype):
    dtype = np.dtype(dtype)
    c_type = np.ctypeslib.as_ctypes_type(dtype)
    size = int(np.prod(shape))
    raw = mp.RawArray(c_type, size)
    array = np.frombuffer(raw, dtype=dtype, count=size).reshape(shape)
    return raw, array


def shared_ndarray_view(raw, shape, dtype):
    dtype = np.dtype(dtype)
    size = int(np.prod(shape))
    return np.frombuffer(raw, dtype=dtype, count=size).reshape(shape)


def build_worker_buffer_shapes(rollout_steps: int, env_count: int, env_spec: dict):
    num_agents = int(env_spec["num_agents"])
    local_obs_dim = int(env_spec["local_obs_dim"])
    global_state_dim = int(env_spec["global_state_dim"])
    attack_dim = int(env_spec["attack_dim"])
    actor_hidden_dim = int(env_spec["actor_hidden_dim"])
    return {
        "local_obs": (rollout_steps, env_count, num_agents, local_obs_dim),
        "global_state": (rollout_steps, env_count, global_state_dim),
        "agent_ids": (rollout_steps, env_count, num_agents),
        "attack_masks": (rollout_steps, env_count, num_agents, attack_dim),
        "alive_mask": (rollout_steps, env_count, num_agents),
        "actor_h": (rollout_steps, env_count, num_agents, actor_hidden_dim),
        "course_action": (rollout_steps, env_count, num_agents),
        "attack_action": (rollout_steps, env_count, num_agents),
        "mode_action": (rollout_steps, env_count, num_agents),
        "log_prob": (rollout_steps, env_count, num_agents),
        "mode_log_prob": (rollout_steps, env_count, num_agents),
        "mode_decision": (rollout_steps, env_count, num_agents),
        "mode_duration": (rollout_steps, env_count, num_agents),
        "reward": (rollout_steps, env_count),
        "reward_env": (rollout_steps, env_count),
        "reward_mode": (rollout_steps, env_count),
        "reward_exec": (rollout_steps, env_count),
        "damage_reward": (rollout_steps, env_count),
        "kill_reward": (rollout_steps, env_count),
        "survival_reward": (rollout_steps, env_count),
        "win_indicator": (rollout_steps, env_count),
        "done": (rollout_steps, env_count),
        "agent_aux_reward": (rollout_steps, env_count, num_agents),
        "final_global_state": (env_count, global_state_dim),
    }


def allocate_worker_shared_buffers(rollout_steps: int, env_count: int, env_spec: dict):
    shapes = build_worker_buffer_shapes(rollout_steps, env_count, env_spec)
    shared = {}
    for key, shape in shapes.items():
        dtype = np.float32 if key == "final_global_state" else ROLLOUT_BUFFER_DTYPES[key]
        raw, _view = make_shared_ndarray(shape, dtype)
        shared[key] = {
            "raw": raw,
            "shape": shape,
            "dtype": np.dtype(dtype).str,
        }
    return shared


def attach_worker_shared_buffers(shared_buffers):
    attached = {}
    for key, meta in shared_buffers.items():
        attached[key] = shared_ndarray_view(meta["raw"], meta["shape"], np.dtype(meta["dtype"]))
    return attached


def _collector_process_main(args, worker_idx: int, env_count: int, conn, env_spec: dict, shared_buffers):
    torch.set_num_threads(1)
    set_seed(args.seed + worker_idx * 1009)
    envs = [build_env(args, seed_offset=worker_idx * 100000 + env_idx * 9973) for env_idx in range(env_count)]
    worker_model = None
    obs_batch = None
    shared_views = attach_worker_shared_buffers(shared_buffers)

    try:
        while True:
            command, payload = conn.recv()
            if command == "reset":
                base_seed = payload
                obs_batch = []
                for env_idx, env in enumerate(envs):
                    seed = None if base_seed is None else int(base_seed + worker_idx * 1000 + env_idx)
                    obs_batch.append(env.reset(seed=seed))
                conn.send(obs_batch)
                continue

            if command == "collect":
                actor_state, rollout_steps = payload
                if obs_batch is None:
                    obs_batch = [env.reset() for env in envs]

                if worker_model is None:
                    local_obs_dim = int(env_spec["local_obs_dim"])
                    global_state_dim = obs_batch[0]["global_state"].shape[0]
                    num_agents = obs_batch[0]["local_obs"].shape[0]
                    worker_model = TeamActorCritic(
                        MAPPOModelConfig(
                            local_obs_dim=local_obs_dim,
                            global_state_dim=global_state_dim,
                            num_agents=num_agents,
                            hidden_size=args.hidden_size,
                            role_embed_dim=args.role_embed_dim,
                            course_embed_dim=args.course_embed_dim,
                        )
                    )
                    worker_model.eval()

                actor_state_tensors = {key: torch.from_numpy(value) for key, value in actor_state.items()}
                worker_model.load_state_dict(actor_state_tensors, strict=False)

                actor_hidden_dim = worker_model.actor_hidden_dim
                num_agents = int(env_spec["num_agents"])
                use_obs_id_onehot = bool(env_spec.get("use_obs_id_onehot", False))
                mode_interval = int(env_spec.get("mode_interval", 8))

                actor_h_batch = np.zeros((env_count, num_agents, actor_hidden_dim), dtype=np.float32)
                mode_action_batch = np.zeros((env_count, num_agents), dtype=np.int64)
                mode_duration_batch = np.zeros((env_count, num_agents), dtype=np.int64)
                mode_steps_to_refresh = np.zeros((env_count, num_agents), dtype=np.int64)
                episodes = []
                rnn_hidden_mismatch_count = 0
                rnn_hidden_max_abs_diff = 0.0
                prev_expected_actor_h = None

                for step_idx in range(rollout_steps):
                    raw_local_obs = np.stack([obs["local_obs"] for obs in obs_batch], axis=0)
                    agent_ids_batch = np.stack([obs["agent_ids"] for obs in obs_batch], axis=0)
                    if use_obs_id_onehot:
                        local_obs_batch = append_agent_id_onehot(raw_local_obs, agent_ids_batch, num_agents)
                    else:
                        local_obs_batch = raw_local_obs

                    if prev_expected_actor_h is not None:
                        diff = np.abs(actor_h_batch - prev_expected_actor_h)
                        max_diff = float(np.max(diff))
                        if max_diff > 1e-5:
                            rnn_hidden_mismatch_count += int(np.count_nonzero(diff > 1e-5))
                            rnn_hidden_max_abs_diff = max(rnn_hidden_max_abs_diff, max_diff)

                    # High-level mode selection runs at a slower cadence.
                    mode_decision_mask = mode_steps_to_refresh <= 0
                    with torch.no_grad():
                        flat_actor_h_t = torch.as_tensor(
                            actor_h_batch.reshape(-1, actor_hidden_dim),
                            dtype=torch.float32,
                            device=torch.device("cpu"),
                        )
                        mode_logits = worker_model.mode_head(flat_actor_h_t)
                        mode_dist = Categorical(logits=mode_logits)
                        sampled_mode_t = mode_dist.sample()
                        sampled_mode = sampled_mode_t.reshape(env_count, num_agents).cpu().numpy()
                        sampled_mode_log_prob = (
                            mode_dist.log_prob(sampled_mode_t).reshape(env_count, num_agents).cpu().numpy()
                        )
                        prev_mode_t = torch.as_tensor(
                            mode_action_batch.reshape(-1),
                            dtype=torch.long,
                            device=torch.device("cpu"),
                        )
                        prev_mode_log_prob = (
                            mode_dist.log_prob(prev_mode_t).reshape(env_count, num_agents).cpu().numpy()
                        )

                    mode_action_batch = np.where(mode_decision_mask, sampled_mode, mode_action_batch)
                    mode_log_prob_batch = np.where(
                        mode_decision_mask,
                        sampled_mode_log_prob,
                        prev_mode_log_prob,
                    ).astype(np.float32, copy=False)
                    mode_duration_batch = np.where(mode_decision_mask, 1, mode_duration_batch + 1).astype(np.int64, copy=False)
                    mode_steps_to_refresh = np.where(
                        mode_decision_mask,
                        max(mode_interval - 1, 0),
                        mode_steps_to_refresh - 1,
                    ).astype(np.int64, copy=False)

                    stacked = {
                        "local_obs": local_obs_batch,
                        "global_state": np.stack([obs["global_state"] for obs in obs_batch], axis=0),
                        "attack_masks": np.stack([obs["attack_masks"] for obs in obs_batch], axis=0),
                        "alive_mask": np.stack([obs["alive_mask"] for obs in obs_batch], axis=0),
                        "agent_ids": agent_ids_batch,
                        "actor_h": actor_h_batch.copy(),
                        "mode_action": mode_action_batch.copy(),
                    }

                    with torch.no_grad():
                        actions = sample_actions(worker_model, stacked, torch.device("cpu"), deterministic=False)

                    shared_views["local_obs"][step_idx] = stacked["local_obs"]
                    shared_views["global_state"][step_idx] = stacked["global_state"]
                    shared_views["attack_masks"][step_idx] = stacked["attack_masks"].astype(np.uint8, copy=False)
                    shared_views["alive_mask"][step_idx] = stacked["alive_mask"]
                    shared_views["agent_ids"][step_idx] = stacked["agent_ids"]
                    shared_views["actor_h"][step_idx] = stacked["actor_h"]
                    shared_views["course_action"][step_idx] = actions["course_action"].cpu().numpy()
                    shared_views["attack_action"][step_idx] = actions["attack_action"].cpu().numpy()
                    shared_views["mode_action"][step_idx] = mode_action_batch
                    shared_views["log_prob"][step_idx] = actions["log_prob"].cpu().numpy()
                    shared_views["mode_log_prob"][step_idx] = mode_log_prob_batch
                    shared_views["mode_decision"][step_idx] = mode_decision_mask.astype(np.float32, copy=False)
                    shared_views["mode_duration"][step_idx] = mode_duration_batch

                    next_obs_batch = []
                    rewards = np.zeros((env_count,), dtype=np.float32)
                    rewards_env = np.zeros((env_count,), dtype=np.float32)
                    rewards_mode = np.zeros((env_count,), dtype=np.float32)
                    rewards_exec = np.zeros((env_count,), dtype=np.float32)
                    damage_rewards = np.zeros((env_count,), dtype=np.float32)
                    kill_rewards = np.zeros((env_count,), dtype=np.float32)
                    survival_rewards = np.zeros((env_count,), dtype=np.float32)
                    win_indicators = np.zeros((env_count,), dtype=np.float32)
                    dones = np.zeros((env_count,), dtype=np.float32)
                    aux_rewards = np.zeros((env_count, num_agents), dtype=np.float32)

                    for env_idx, env in enumerate(envs):
                        env_actions = np.stack(
                            [
                                actions["course_action"][env_idx].cpu().numpy(),
                                actions["attack_action"][env_idx].cpu().numpy(),
                            ],
                            axis=-1,
                        )
                        next_obs, reward, done, info = env.step(env_actions)
                        rewards[env_idx] = float(reward)
                        rewards_env[env_idx] = float(info.get("reward_env", 0.0))
                        rewards_mode[env_idx] = float(info.get("reward_mode", 0.0))
                        rewards_exec[env_idx] = float(info.get("reward_exec", 0.0))
                        damage_rewards[env_idx] = float(info.get("damage_reward", 0.0))
                        kill_rewards[env_idx] = float(info.get("kill_reward", 0.0))
                        survival_rewards[env_idx] = float(info.get("survival_reward", 0.0))
                        win_indicators[env_idx] = float(info.get("win_indicator", 0.0))
                        dones[env_idx] = 1.0 if done else 0.0

                        info_aux = info.get("agent_aux_reward", None)
                        if info_aux is not None:
                            info_aux = np.asarray(info_aux, dtype=np.float32)
                            if info_aux.shape == (num_agents,):
                                aux_rewards[env_idx] = info_aux

                        if done:
                            episodes.append(info["episode_extra_stats"])
                            next_obs = env.reset()
                        next_obs_batch.append(next_obs)

                    shared_views["reward"][step_idx] = rewards
                    shared_views["reward_env"][step_idx] = rewards_env
                    shared_views["reward_mode"][step_idx] = rewards_mode
                    shared_views["reward_exec"][step_idx] = rewards_exec
                    shared_views["damage_reward"][step_idx] = damage_rewards
                    shared_views["kill_reward"][step_idx] = kill_rewards
                    shared_views["survival_reward"][step_idx] = survival_rewards
                    shared_views["win_indicator"][step_idx] = win_indicators
                    shared_views["done"][step_idx] = dones
                    shared_views["agent_aux_reward"][step_idx] = aux_rewards

                    actor_h_batch = actions["next_actor_h"].cpu().numpy()
                    prev_expected_actor_h = actor_h_batch.copy()
                    if np.any(dones > 0.5):
                        done_rows = np.where(dones > 0.5)[0]
                        actor_h_batch[done_rows, :, :] = 0.0
                        prev_expected_actor_h[done_rows, :, :] = 0.0
                        mode_action_batch[done_rows, :] = 0
                        mode_duration_batch[done_rows, :] = 0
                        mode_steps_to_refresh[done_rows, :] = 0
                    obs_batch = next_obs_batch

                final_global_state = np.stack([obs["global_state"] for obs in obs_batch], axis=0)
                shared_views["final_global_state"][:] = final_global_state

                conn.send(
                    {
                        "episodes": episodes,
                        "rnn_hidden_mismatch_count": int(rnn_hidden_mismatch_count),
                        "rnn_hidden_max_abs_diff": float(rnn_hidden_max_abs_diff),
                    }
                )
                continue

            if command == "close":
                break

            raise ValueError("Unknown collector command: %s" % command)
    finally:
        for env in envs:
            env.close()
        conn.close()


class CollectorPool:
    def __init__(self, args, env_spec: dict):
        total_envs = max(1, int(args.num_envs))
        requested_workers = max(1, int(args.num_workers))
        actual_workers = min(requested_workers, total_envs)

        base_envs = total_envs // actual_workers
        remainder = total_envs % actual_workers
        self.worker_env_counts = [
            base_envs + (1 if worker_idx < remainder else 0) for worker_idx in range(actual_workers)
        ]

        ctx = mp.get_context("spawn")
        self.parent_conns = []
        self.processes = []
        self.num_workers = actual_workers
        self.num_envs = total_envs
        self.rollout_steps = int(args.rollout)
        self.env_spec = env_spec
        self.worker_shared_meta = []
        self.worker_shared_views = []

        for worker_idx, env_count in enumerate(self.worker_env_counts):
            shared_buffers = allocate_worker_shared_buffers(self.rollout_steps, env_count, env_spec)
            parent_conn, child_conn = ctx.Pipe()
            process = ctx.Process(
                target=_collector_process_main,
                args=(args, worker_idx, env_count, child_conn, env_spec, shared_buffers),
                daemon=True,
            )
            process.start()
            child_conn.close()
            self.parent_conns.append(parent_conn)
            self.processes.append(process)
            self.worker_shared_meta.append(shared_buffers)
            self.worker_shared_views.append(attach_worker_shared_buffers(shared_buffers))

    def reset(self, seed=None):
        for worker_idx, conn in enumerate(self.parent_conns):
            worker_seed = None if seed is None else int(seed + worker_idx * 1000)
            conn.send(("reset", worker_seed))

        obs_batch = []
        for conn in self.parent_conns:
            obs_batch.extend(conn.recv())
        return obs_batch

    def collect(self, actor_state: dict, rollout_steps: int):
        for conn in self.parent_conns:
            conn.send(("collect", (actor_state, rollout_steps)))

        results = []
        for worker_idx, conn in enumerate(self.parent_conns):
            worker_result = conn.recv()
            worker_result["buffer"] = self.worker_shared_views[worker_idx]
            worker_result["final_global_state"] = self.worker_shared_views[worker_idx]["final_global_state"]
            results.append(worker_result)
        return results

    def close(self):
        for conn in self.parent_conns:
            try:
                conn.send(("close", None))
            except (BrokenPipeError, EOFError):
                pass
        for process in self.processes:
            process.join(timeout=5)
            if process.is_alive():
                process.terminate()


def summarize_episode_stats(episodes):
    if not episodes:
        return {}
    keys = sorted(episodes[0].keys())
    summary = {"episodes": len(episodes)}
    for key in keys:
        values = [float(ep[key]) for ep in episodes if key in ep]
        if not values:
            continue
        summary_key = key if key.endswith("_mean") else "%s_mean" % key
        summary[summary_key] = float(np.mean(values))
    if "win_flag_mean" in summary:
        summary["win_rate"] = summary["win_flag_mean"]
    return summary


def log_summary(writer: Optional[SummaryWriter], prefix: str, summary: Dict[str, float], step: int) -> None:
    if writer is None:
        return
    for key, value in summary.items():
        if key == "episodes":
            continue
        try:
            writer.add_scalar("%s/%s" % (prefix, key), float(value), step)
        except Exception:
            continue


def select_eval_actions(model, obs, actor_h, device, deterministic: bool, concat_agent_id_onehot: bool, num_agents: int):
    local_obs_np = obs["local_obs"]
    if concat_agent_id_onehot:
        local_obs_np = append_agent_id_onehot(local_obs_np[None, ...], obs["agent_ids"][None, ...], num_agents)[0]
    local_obs = torch.as_tensor(local_obs_np, dtype=torch.float32, device=device)
    agent_ids = torch.as_tensor(obs["agent_ids"], dtype=torch.long, device=device)
    attack_masks = torch.as_tensor(obs["attack_masks"], dtype=torch.bool, device=device)
    attack_masks = ensure_valid_action_mask(attack_masks)
    actor_h_t = torch.as_tensor(actor_h, dtype=torch.float32, device=device)

    with torch.no_grad():
        course_logits, _attack_logits_unused, next_actor_h = model.actor_step(local_obs, agent_ids, actor_h_t)
        course_logits = sanitize_logits(course_logits)
        if deterministic:
            course_action = torch.argmax(course_logits, dim=-1)
        else:
            course_action = torch.distributions.Categorical(logits=course_logits).sample()

        attack_logits = model.attack_logits(next_actor_h, course_action)
        attack_logits = sanitize_logits(attack_logits)
        invalid_logit = torch.finfo(attack_logits.dtype).min
        attack_logits = attack_logits.masked_fill(~attack_masks, invalid_logit)
        if deterministic:
            attack_action = torch.argmax(attack_logits, dim=-1)
        else:
            attack_action = torch.distributions.Categorical(logits=attack_logits).sample()

    actions = np.stack([course_action.cpu().numpy(), attack_action.cpu().numpy()], axis=-1)
    return actions, next_actor_h.cpu().numpy()


def run_evaluation(model, device, args, env_steps: int):
    runtime_opponent = str(getattr(args, "runtime_maca_opponent", args.maca_opponent))
    runtime_max_step = int(getattr(args, "runtime_maca_max_step", args.maca_max_step))
    runtime_random_pos = bool(getattr(args, "runtime_maca_random_pos", args.maca_random_pos))
    eval_opponent = args.eval_opponent or runtime_opponent
    eval_config = MAPPOMaCAConfig(
        map_path=args.maca_map_path,
        red_obs_ind=args.maca_red_obs_ind,
        opponent=eval_opponent,
        max_step=runtime_max_step,
        render=False,
        random_pos=runtime_random_pos,
        random_seed=args.seed + 900000 + int(env_steps),
        adaptive_support_policy=args.maca_adaptive_support_policy,
        support_search_hold=args.maca_support_search_hold,
        delta_course_action=args.maca_delta_course_action,
        course_delta_deg=args.maca_course_delta_deg,
        max_visible_enemies=args.maca_max_visible_enemies,
        friendly_attrition_penalty=args.maca_friendly_attrition_penalty,
        enemy_attrition_reward=args.maca_enemy_attrition_reward,
        track_memory_steps=args.maca_track_memory_steps,
        contact_reward=args.maca_contact_reward,
        progress_reward_scale=args.maca_progress_reward_scale,
        progress_reward_cap=args.maca_progress_reward_cap,
        attack_window_reward=args.maca_attack_window_reward,
        agent_aux_reward_scale=args.maca_agent_aux_reward_scale,
        mode_reward_scale=args.maca_mode_reward_scale,
        exec_reward_scale=args.maca_exec_reward_scale,
        disengage_penalty=args.maca_disengage_penalty,
        bearing_reward_scale=args.maca_bearing_reward_scale,
    )
    env = MAPPOMaCAEnv(eval_config)
    actor_h = np.zeros((env.num_agents, model.actor_hidden_dim), dtype=np.float32)
    episode_results = []
    start_time = time.time()

    try:
        obs = env.reset(seed=eval_config.random_seed)
        while len(episode_results) < args.eval_episodes:
            actions, actor_h = select_eval_actions(
                model,
                obs,
                actor_h,
                device,
                args.eval_deterministic,
                concat_agent_id_onehot=bool(args.concat_agent_id_onehot),
                num_agents=env.num_agents,
            )
            obs, _reward, done, info = env.step(actions)
            if not done:
                continue
            episode_results.append(info["episode_extra_stats"])
            actor_h.fill(0.0)
            obs = env.reset(seed=eval_config.random_seed + len(episode_results))
    finally:
        env.close()

    summary = summarize_episode_stats(episode_results)
    payload = {
        "env_steps": int(env_steps),
        "episodes": int(args.eval_episodes),
        "maca_opponent": eval_opponent,
        "deterministic": bool(args.eval_deterministic),
        "eval_wall_time_sec": float(time.time() - start_time),
        "summary": summary,
        "episodes_detail": episode_results,
    }
    eval_output_dir = eval_dir(args)
    eval_output_dir.mkdir(parents=True, exist_ok=True)
    output_path = eval_output_dir / ("eval_%09d.json" % int(env_steps))
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    return payload, output_path


def rollout(collectors: CollectorPool, actor_state: dict, model, device, args, episode_stats, value_normalizer=None):
    worker_results = collectors.collect(actor_state, args.rollout)

    buffer_keys = [
        "local_obs",
        "global_state",
        "agent_ids",
        "attack_masks",
        "alive_mask",
        "actor_h",
        "course_action",
        "attack_action",
        "mode_action",
        "log_prob",
        "mode_log_prob",
        "mode_decision",
        "mode_duration",
        "reward",
        "reward_env",
        "reward_mode",
        "reward_exec",
        "damage_reward",
        "kill_reward",
        "survival_reward",
        "win_indicator",
        "done",
        "agent_aux_reward",
    ]
    buffer = {
        key: np.concatenate([result["buffer"][key] for result in worker_results], axis=1) for key in buffer_keys
    }
    final_global_state = np.concatenate([result["final_global_state"] for result in worker_results], axis=0)

    for result in worker_results:
        for episode in result["episodes"]:
            episode_stats.append(episode)

    hidden_mismatch_count = int(sum(int(result.get("rnn_hidden_mismatch_count", 0)) for result in worker_results))
    hidden_max_abs_diff = float(max(float(result.get("rnn_hidden_max_abs_diff", 0.0)) for result in worker_results))

    global_state_t = torch.as_tensor(buffer["global_state"], dtype=torch.float32, device=device)
    mode_action_t = torch.as_tensor(buffer["mode_action"], dtype=torch.long, device=device)
    with torch.no_grad():
        value_heads = model.value_heads(global_state_t, mode_actions=mode_action_t)
        raw_values = value_heads["team"].cpu().numpy()
        final_mode_action = mode_action_t[-1]
        raw_next_value = model.value(
            torch.as_tensor(final_global_state, dtype=torch.float32, device=device),
            mode_actions=final_mode_action,
        ).cpu().numpy()
        if value_normalizer is not None:
            buffer["value"] = value_normalizer.denormalize(raw_values)
            next_value = value_normalizer.denormalize(raw_next_value)
        else:
            buffer["value"] = raw_values
            next_value = raw_next_value

    rollout_diag = {
        "rnn_hidden_mismatch_count": hidden_mismatch_count,
        "rnn_hidden_max_abs_diff": hidden_max_abs_diff,
        "damage_reward_mean": float(np.mean(buffer["damage_reward"])),
        "reward_env_mean": float(np.mean(buffer["reward_env"])),
        "reward_mode_mean": float(np.mean(buffer["reward_mode"])),
        "reward_exec_mean": float(np.mean(buffer["reward_exec"])),
        "kill_reward_mean": float(np.mean(buffer["kill_reward"])),
        "survival_reward_mean": float(np.mean(buffer["survival_reward"])),
        "win_indicator_mean": float(np.mean(buffer["win_indicator"])),
        "value_contact_mean": float(value_heads["contact"].mean().item()),
        "value_opportunity_mean": float(value_heads["opportunity"].mean().item()),
        "value_survival_mean": float(value_heads["survival"].mean().item()),
    }
    return buffer, next_value, rollout_diag


def compute_gae(buffer, next_value, gamma, gae_lambda):
    rewards = buffer["reward"]
    dones = buffer["done"]
    values = buffer["value"]
    advantages = np.zeros_like(rewards, dtype=np.float32)
    last_gae = np.zeros_like(next_value, dtype=np.float32)

    for step in reversed(range(rewards.shape[0])):
        next_nonterminal = 1.0 - dones[step]
        next_values = next_value if step == rewards.shape[0] - 1 else values[step + 1]
        delta = rewards[step] + gamma * next_values * next_nonterminal - values[step]
        last_gae = delta + gamma * gae_lambda * next_nonterminal * last_gae
        advantages[step] = last_gae

    returns = advantages + values
    return advantages, returns


def compute_aux_advantages(buffer, gamma):
    aux_rewards = buffer["agent_aux_reward"]
    dones = buffer["done"]

    returns = np.zeros_like(aux_rewards, dtype=np.float32)
    running = np.zeros_like(aux_rewards[0], dtype=np.float32)
    for step in reversed(range(aux_rewards.shape[0])):
        nonterminal = (1.0 - dones[step])[:, None]
        running = aux_rewards[step] + gamma * nonterminal * running
        returns[step] = running

    # Center using alive-agent mean to form proper pseudo-advantages
    alive = buffer["alive_mask"]
    active_mask = alive > 0.5
    active_returns = returns[active_mask]
    if active_returns.size > 0:
        returns = returns - float(np.mean(active_returns))
    return returns


def compute_mode_advantages(buffer, team_advantages):
    mode_decision = np.asarray(buffer["mode_decision"], dtype=np.float32)
    mode_duration = np.asarray(buffer["mode_duration"], dtype=np.float32)
    alive_mask = np.asarray(buffer["alive_mask"], dtype=np.float32)

    # High-level advantage uses team signal, weighted by option duration and
    # evaluated only when a new mode is selected.
    high_adv = team_advantages[:, :, None] * np.maximum(mode_duration, 1.0)
    high_adv = high_adv * mode_decision * alive_mask

    active = high_adv[mode_decision > 0.5]
    if active.size > 0:
        mean = float(np.mean(active))
        std = float(np.std(active))
        if std > 1e-6:
            high_adv = (high_adv - mean) / std
        else:
            high_adv = high_adv - mean
        high_adv = high_adv * mode_decision * alive_mask
    return high_adv.astype(np.float32, copy=False)


def build_recurrent_chunks(rollout_steps: int, num_envs: int, num_agents: int, chunk_len: int):
    chunks = []
    for env_idx in range(num_envs):
        for agent_idx in range(num_agents):
            start = 0
            while start < rollout_steps:
                end = min(rollout_steps, start + chunk_len)
                chunks.append((env_idx, agent_idx, start, end))
                start = end
    return chunks


def pack_recurrent_minibatch(
    chunk_indices,
    chunks,
    local_obs,
    global_state,
    agent_ids,
    attack_masks,
    alive_mask,
    actor_h,
    old_log_prob,
    mode_action,
    mode_log_prob,
    mode_decision,
    mode_duration,
    course_action,
    attack_action,
    damage_reward,
    survival_reward,
    aux_reward_mean,
    returns_t,
    combined_adv,
    mode_adv,
    burn_in: int,
):
    batch_size = len(chunk_indices)
    hidden_dim = actor_h.shape[-1]
    obs_dim = local_obs.shape[-1]
    global_dim = global_state.shape[-1]
    attack_dim = attack_masks.shape[-1]

    chunk_meta = []
    max_total_len = 0
    for chunk_idx in chunk_indices:
        env_idx, agent_idx, start, end = chunks[int(chunk_idx)]
        burn_start = max(0, start - burn_in)
        total_len = end - burn_start
        train_len = end - start
        train_offset = start - burn_start
        chunk_meta.append((env_idx, agent_idx, start, end, burn_start, total_len, train_len, train_offset))
        max_total_len = max(max_total_len, total_len)

    local_batch = local_obs.new_zeros((max_total_len, batch_size, obs_dim))
    global_batch = global_state.new_zeros((max_total_len, batch_size, global_dim))
    agent_id_batch = agent_ids.new_zeros((max_total_len, batch_size))
    attack_mask_batch = attack_masks.new_zeros((max_total_len, batch_size, attack_dim))
    actor_h_batch = actor_h.new_zeros((max_total_len, batch_size, hidden_dim))
    course_batch = course_action.new_zeros((max_total_len, batch_size))
    attack_batch = attack_action.new_zeros((max_total_len, batch_size))
    mode_batch = mode_action.new_zeros((max_total_len, batch_size))
    old_log_prob_batch = old_log_prob.new_zeros((max_total_len, batch_size))
    old_mode_log_prob_batch = mode_log_prob.new_zeros((max_total_len, batch_size))
    mode_decision_batch = mode_decision.new_zeros((max_total_len, batch_size))
    mode_duration_batch = mode_duration.new_zeros((max_total_len, batch_size))
    damage_reward_batch = damage_reward.new_zeros((max_total_len, batch_size))
    survival_reward_batch = survival_reward.new_zeros((max_total_len, batch_size))
    aux_reward_mean_batch = aux_reward_mean.new_zeros((max_total_len, batch_size))
    adv_batch = combined_adv.new_zeros((max_total_len, batch_size))
    mode_adv_batch = mode_adv.new_zeros((max_total_len, batch_size))
    return_batch = returns_t.new_zeros((max_total_len, batch_size))
    train_mask = torch.zeros((max_total_len, batch_size), dtype=torch.bool, device=local_obs.device)
    active_policy_mask = torch.zeros((max_total_len, batch_size), dtype=torch.bool, device=local_obs.device)
    seq_valid_mask = torch.zeros((max_total_len, batch_size), dtype=torch.bool, device=local_obs.device)
    init_h = actor_h.new_zeros((batch_size, hidden_dim))
    hidden_state_init_abs_error = actor_h.new_zeros((batch_size,))

    for batch_col, meta in enumerate(chunk_meta):
        env_idx, agent_idx, start, end, burn_start, total_len, train_len, train_offset = meta
        # Suspicious zone for recurrent bugs: init_h MUST be the hidden state recorded
        # at chunk start (or burn-in start). Any mismatch here breaks temporal credit.
        init_h[batch_col] = actor_h[burn_start, env_idx, agent_idx]
        hidden_state_init_abs_error[batch_col] = torch.max(
            torch.abs(init_h[batch_col] - actor_h[burn_start, env_idx, agent_idx])
        )
        seq_valid_mask[:total_len, batch_col] = True

        local_batch[:total_len, batch_col] = local_obs[burn_start:end, env_idx, agent_idx]
        global_batch[:total_len, batch_col] = global_state[burn_start:end, env_idx]
        agent_id_batch[:total_len, batch_col] = agent_ids[burn_start:end, env_idx, agent_idx]
        attack_mask_batch[:total_len, batch_col] = attack_masks[burn_start:end, env_idx, agent_idx]
        actor_h_batch[:total_len, batch_col] = actor_h[burn_start:end, env_idx, agent_idx]
        course_batch[:total_len, batch_col] = course_action[burn_start:end, env_idx, agent_idx]
        attack_batch[:total_len, batch_col] = attack_action[burn_start:end, env_idx, agent_idx]
        mode_batch[:total_len, batch_col] = mode_action[burn_start:end, env_idx, agent_idx]
        mode_decision_batch[:total_len, batch_col] = mode_decision[burn_start:end, env_idx, agent_idx]
        mode_duration_batch[:total_len, batch_col] = mode_duration[burn_start:end, env_idx, agent_idx]
        damage_reward_batch[:total_len, batch_col] = damage_reward[burn_start:end, env_idx]
        survival_reward_batch[:total_len, batch_col] = survival_reward[burn_start:end, env_idx]
        aux_reward_mean_batch[:total_len, batch_col] = aux_reward_mean[burn_start:end, env_idx]

        train_slice = slice(train_offset, train_offset + train_len)
        train_mask[train_slice, batch_col] = True
        old_log_prob_batch[train_slice, batch_col] = old_log_prob[start:end, env_idx, agent_idx]
        old_mode_log_prob_batch[train_slice, batch_col] = mode_log_prob[start:end, env_idx, agent_idx]
        adv_batch[train_slice, batch_col] = combined_adv[start:end, env_idx, agent_idx]
        mode_adv_batch[train_slice, batch_col] = mode_adv[start:end, env_idx, agent_idx]
        return_batch[train_slice, batch_col] = returns_t[start:end, env_idx]
        active_policy_mask[train_slice, batch_col] = alive_mask[start:end, env_idx, agent_idx] > 0.5

    return {
        "init_h": init_h.detach(),
        "local_obs": local_batch,
        "global_state": global_batch,
        "agent_ids": agent_id_batch,
        "attack_masks": attack_mask_batch,
        "actor_h": actor_h_batch,
        "course_action": course_batch,
        "attack_action": attack_batch,
        "mode_action": mode_batch,
        "mode_decision": mode_decision_batch,
        "mode_duration": mode_duration_batch,
        "damage_reward": damage_reward_batch,
        "survival_reward": survival_reward_batch,
        "aux_reward_mean": aux_reward_mean_batch,
        "old_log_prob": old_log_prob_batch,
        "old_mode_log_prob": old_mode_log_prob_batch,
        "advantages": adv_batch,
        "mode_advantages": mode_adv_batch,
        "returns": return_batch,
        "train_mask": train_mask,
        "active_policy_mask": active_policy_mask,
        "seq_valid_mask": seq_valid_mask,
        "hidden_state_init_abs_error": hidden_state_init_abs_error,
    }


def ppo_update(
    model,
    optimizer,
    buffer,
    advantages_team,
    advantages_aux,
    returns,
    device,
    args,
    value_normalizer=None,
    teacher_model=None,
    update_idx: int = 0,
):
    local_obs = torch.as_tensor(buffer["local_obs"], dtype=torch.float32, device=device)
    global_state = torch.as_tensor(buffer["global_state"], dtype=torch.float32, device=device)
    agent_ids = torch.as_tensor(buffer["agent_ids"], dtype=torch.long, device=device)
    attack_masks = torch.as_tensor(buffer["attack_masks"], dtype=torch.bool, device=device)
    alive_mask = torch.as_tensor(buffer["alive_mask"], dtype=torch.float32, device=device)
    actor_h = torch.as_tensor(buffer["actor_h"], dtype=torch.float32, device=device)
    old_log_prob = torch.as_tensor(buffer["log_prob"], dtype=torch.float32, device=device)
    old_mode_log_prob = torch.as_tensor(buffer["mode_log_prob"], dtype=torch.float32, device=device)
    course_action = torch.as_tensor(buffer["course_action"], dtype=torch.long, device=device)
    attack_action = torch.as_tensor(buffer["attack_action"], dtype=torch.long, device=device)
    mode_action = torch.as_tensor(buffer["mode_action"], dtype=torch.long, device=device)
    mode_decision = torch.as_tensor(buffer["mode_decision"], dtype=torch.float32, device=device)
    mode_duration = torch.as_tensor(buffer["mode_duration"], dtype=torch.float32, device=device)
    damage_reward_t = torch.as_tensor(buffer["damage_reward"], dtype=torch.float32, device=device)
    survival_reward_t = torch.as_tensor(buffer["survival_reward"], dtype=torch.float32, device=device)
    aux_reward_mean_t = torch.as_tensor(
        np.mean(buffer["agent_aux_reward"], axis=-1),
        dtype=torch.float32,
        device=device,
    )

    adv_team_t = torch.as_tensor(advantages_team, dtype=torch.float32, device=device)
    adv_aux_t = torch.as_tensor(advantages_aux, dtype=torch.float32, device=device)

    # Value normalization: normalize returns for critic training
    if value_normalizer is not None:
        norm_returns = value_normalizer.normalize(returns)
        returns_t = torch.as_tensor(norm_returns, dtype=torch.float32, device=device)
    else:
        returns_t = torch.as_tensor(returns, dtype=torch.float32, device=device)

    returns_denorm_t = torch.as_tensor(returns, dtype=torch.float32, device=device)

    combined_adv = args.team_adv_weight * adv_team_t.unsqueeze(-1) + args.aux_adv_weight * adv_aux_t
    combined_adv = sanitize_tensor(combined_adv)
    active_mask_bool = alive_mask > 0.5
    active_adv = combined_adv[active_mask_bool]
    if active_adv.numel() > 0:
        adv_mean = active_adv.mean()
        adv_std = active_adv.std(unbiased=False)
        combined_adv = (combined_adv - adv_mean) / torch.clamp(adv_std, min=1e-6)
    # Explicitly zero dead-agent advantages to avoid stale gradients from padded/dead slots.
    combined_adv = combined_adv * alive_mask

    mode_adv = adv_team_t.unsqueeze(-1) * torch.clamp(mode_duration, min=1.0)
    mode_adv = mode_adv * mode_decision * alive_mask
    mode_adv = sanitize_tensor(mode_adv)
    active_mode_adv = mode_adv[mode_decision > 0.5]
    if active_mode_adv.numel() > 0:
        mode_adv_mean = active_mode_adv.mean()
        mode_adv_std = active_mode_adv.std(unbiased=False)
        mode_adv = (mode_adv - mode_adv_mean) / torch.clamp(mode_adv_std, min=1e-6)
        mode_adv = mode_adv * mode_decision * alive_mask

    imitation_coef = float(max(args.imitation_coef, 0.0))
    if imitation_coef > 0.0 and int(args.imitation_warmup_updates) > 0:
        warmup = float(max(1, int(args.imitation_warmup_updates)))
        decay = max(0.0, 1.0 - float(max(update_idx, 0)) / warmup)
        imitation_coef = imitation_coef * decay

    with torch.no_grad():
        explained_var = tensor_explained_variance(
            returns_denorm_t,
            torch.as_tensor(buffer["value"], dtype=torch.float32, device=device),
        )

    rollout_steps, num_envs, num_agents = local_obs.shape[0], local_obs.shape[1], local_obs.shape[2]
    chunks = build_recurrent_chunks(rollout_steps, num_envs, num_agents, args.chunk_len)
    num_chunks = len(chunks)

    total_policy_low = 0.0
    total_policy_high = 0.0
    total_value = 0.0
    total_entropy_low = 0.0
    total_entropy_high = 0.0
    total_imitation = 0.0
    total_minibatches = 0
    total_active = 0
    total_grad_steps = 0
    total_actor_grad_norm = 0.0
    total_critic_grad_norm = 0.0
    total_train_mask_count = 0
    total_active_mask_count = 0
    total_hidden_init_err = 0.0
    total_hidden_init_count = 0
    total_skipped_non_finite = 0
    total_repaired_params = 0

    actor_params = [
        p for name, p in model.named_parameters() if not name.startswith("critic")
    ]
    critic_params = [
        p for name, p in model.named_parameters() if name.startswith("critic")
    ]

    for _ in range(args.ppo_epochs):
        order = np.random.permutation(num_chunks)
        mini_batches = np.array_split(order, max(1, args.num_mini_batches))

        for mini_batch_indices in mini_batches:
            if len(mini_batch_indices) <= 0:
                continue

            repaired = repair_non_finite_parameters(model)
            total_repaired_params += int(repaired)

            packed = pack_recurrent_minibatch(
                mini_batch_indices,
                chunks,
                local_obs,
                global_state,
                agent_ids,
                attack_masks,
                alive_mask,
                actor_h,
                old_log_prob,
                mode_action,
                old_mode_log_prob,
                mode_decision,
                mode_duration,
                course_action,
                attack_action,
                damage_reward_t,
                survival_reward_t,
                aux_reward_mean_t,
                returns_t,
                combined_adv,
                mode_adv,
                args.burn_in,
            )

            train_mask = packed["train_mask"]
            active_train_mask = train_mask & packed["active_policy_mask"]
            train_global_state = packed["global_state"][active_train_mask]
            train_returns = packed["returns"][active_train_mask]
            train_mode_action = packed["mode_action"][active_train_mask]
            if train_global_state.numel() <= 0:
                continue

            train_global_state = sanitize_tensor(train_global_state)
            train_returns = sanitize_tensor(train_returns)

            total_train_mask_count += int(train_mask.sum().item())
            total_active_mask_count += int(active_train_mask.sum().item())
            init_err = packed["hidden_state_init_abs_error"]
            total_hidden_init_err += float(init_err.sum().item())
            total_hidden_init_count += int(init_err.numel())

            value_heads_t = model.value_heads(train_global_state, mode_actions=train_mode_action)
            value_team_pred = sanitize_tensor(value_heads_t["team"])
            value_contact_pred = sanitize_tensor(value_heads_t["contact"])
            value_opportunity_pred = sanitize_tensor(value_heads_t["opportunity"])
            value_survival_pred = sanitize_tensor(value_heads_t["survival"])

            contact_target = sanitize_tensor(packed["aux_reward_mean"][active_train_mask])
            opportunity_target = sanitize_tensor(packed["damage_reward"][active_train_mask])
            survival_target = sanitize_tensor(packed["survival_reward"][active_train_mask])

            def _norm_target(x: torch.Tensor) -> torch.Tensor:
                if x.numel() <= 1:
                    return x
                mean = x.mean()
                std = x.std(unbiased=False)
                return (x - mean) / torch.clamp(std, min=1e-6)

            value_team_loss = 0.5 * (value_team_pred - train_returns).pow(2).mean()
            value_contact_loss = 0.5 * (value_contact_pred - _norm_target(contact_target)).pow(2).mean()
            value_opportunity_loss = 0.5 * (value_opportunity_pred - _norm_target(opportunity_target)).pow(2).mean()
            value_survival_loss = 0.5 * (value_survival_pred - _norm_target(survival_target)).pow(2).mean()
            value_loss = (
                value_team_loss
                + 0.25 * (value_contact_loss + value_opportunity_loss + value_survival_loss)
            ) * args.value_loss_coeff

            h = packed["init_h"]
            new_log_probs = []
            old_log_probs = []
            entropies = []
            advantages = []
            high_new_log_probs = []
            high_old_log_probs = []
            high_entropies = []
            high_advantages = []
            imitation_losses = []

            max_total_len = packed["local_obs"].shape[0]
            for seq_idx in range(max_total_len):
                seq_valid = packed["seq_valid_mask"][seq_idx]
                if not torch.any(seq_valid):
                    continue

                seq_local = packed["local_obs"][seq_idx, seq_valid]
                seq_ids = packed["agent_ids"][seq_idx, seq_valid]
                seq_h = h[seq_valid]
                seq_course = packed["course_action"][seq_idx, seq_valid]
                seq_mode = packed["mode_action"][seq_idx, seq_valid]

                course_logits, attack_logits, next_h_valid = model.actor_step(
                    seq_local,
                    seq_ids,
                    seq_h,
                    course_actions=seq_course,
                    mode_actions=seq_mode,
                )
                course_logits = sanitize_logits(course_logits)
                attack_logits = sanitize_logits(attack_logits)
                next_h_valid = sanitize_tensor(next_h_valid)
                h = h.clone()
                h[seq_valid] = next_h_valid

                seq_train = packed["train_mask"][seq_idx, seq_valid]
                if not torch.any(seq_train):
                    continue

                course_logits = course_logits[seq_train]
                attack_logits = attack_logits[seq_train]
                seq_course = seq_course[seq_train]
                seq_attack = packed["attack_action"][seq_idx, seq_valid][seq_train]
                seq_attack_mask = packed["attack_masks"][seq_idx, seq_valid][seq_train]
                seq_attack_mask = ensure_valid_action_mask(seq_attack_mask)
                seq_old_log_prob = packed["old_log_prob"][seq_idx, seq_valid][seq_train]
                seq_adv = packed["advantages"][seq_idx, seq_valid][seq_train]
                seq_active = packed["active_policy_mask"][seq_idx, seq_valid][seq_train]
                seq_mode = packed["mode_action"][seq_idx, seq_valid][seq_train]
                seq_mode_old_log_prob = packed["old_mode_log_prob"][seq_idx, seq_valid][seq_train]
                seq_mode_decision = packed["mode_decision"][seq_idx, seq_valid][seq_train]
                seq_mode_adv = packed["mode_advantages"][seq_idx, seq_valid][seq_train]
                seq_actor_h = packed["actor_h"][seq_idx, seq_valid][seq_train]
                seq_old_log_prob = sanitize_tensor(seq_old_log_prob)
                seq_adv = sanitize_tensor(seq_adv)
                seq_mode_old_log_prob = sanitize_tensor(seq_mode_old_log_prob)
                seq_mode_adv = sanitize_tensor(seq_mode_adv)
                seq_actor_h = sanitize_tensor(seq_actor_h)

                if course_logits.shape[0] <= 0:
                    continue

                course_dist = Categorical(logits=course_logits)
                attack_dist = masked_categorical(attack_logits, seq_attack_mask)
                seq_log_prob = course_dist.log_prob(seq_course) + attack_dist.log_prob(seq_attack)
                seq_entropy = course_dist.entropy() + attack_dist.entropy()

                if torch.any(seq_active):
                    new_log_probs.append(seq_log_prob[seq_active])
                    old_log_probs.append(seq_old_log_prob[seq_active])
                    entropies.append(seq_entropy[seq_active])
                    advantages.append(seq_adv[seq_active])

                    if teacher_model is not None and imitation_coef > 0.0:
                        with torch.no_grad():
                            teacher_course_logits, teacher_attack_logits, _teacher_next_h = teacher_model.actor_step(
                                seq_local,
                                seq_ids,
                                seq_h.detach(),
                                course_actions=packed["course_action"][seq_idx, seq_valid],
                                mode_actions=packed["mode_action"][seq_idx, seq_valid],
                            )
                            teacher_course_logits = teacher_course_logits[seq_train]
                            teacher_attack_logits = teacher_attack_logits[seq_train]
                            teacher_course_logits = sanitize_logits(teacher_course_logits)
                            teacher_attack_logits = sanitize_logits(teacher_attack_logits)
                            teacher_attack_logits = teacher_attack_logits.masked_fill(
                                ~seq_attack_mask,
                                torch.finfo(teacher_attack_logits.dtype).min,
                            )

                        student_attack_logits = sanitize_logits(attack_logits).masked_fill(
                            ~seq_attack_mask,
                            torch.finfo(attack_logits.dtype).min,
                        )
                        student_course = Categorical(logits=sanitize_logits(course_logits))
                        teacher_course = Categorical(logits=teacher_course_logits)
                        student_attack = Categorical(logits=student_attack_logits)
                        teacher_attack = Categorical(logits=teacher_attack_logits)
                        kl_course = torch.distributions.kl_divergence(student_course, teacher_course)
                        kl_attack = torch.distributions.kl_divergence(student_attack, teacher_attack)
                        imitation_losses.append((kl_course + kl_attack)[seq_active])

                    mode_logits = sanitize_logits(model.mode_head(seq_actor_h))
                    mode_dist = Categorical(logits=mode_logits)
                    seq_mode_log_prob = mode_dist.log_prob(seq_mode)
                    seq_mode_entropy = mode_dist.entropy()
                    mode_active = seq_active & (seq_mode_decision > 0.5)
                    if torch.any(mode_active):
                        high_new_log_probs.append(seq_mode_log_prob[mode_active])
                        high_old_log_probs.append(seq_mode_old_log_prob[mode_active])
                        high_entropies.append(seq_mode_entropy[mode_active])
                        high_advantages.append(seq_mode_adv[mode_active])

            if new_log_probs:
                new_log_prob_t = torch.cat(new_log_probs, dim=0)
                old_log_prob_t = torch.cat(old_log_probs, dim=0)
                adv_t = torch.cat(advantages, dim=0)
                entropy_t = torch.cat(entropies, dim=0)
                new_log_prob_t = sanitize_tensor(new_log_prob_t)
                old_log_prob_t = sanitize_tensor(old_log_prob_t)
                adv_t = sanitize_tensor(adv_t)
                entropy_t = sanitize_tensor(entropy_t)

                ratio = torch.exp(new_log_prob_t - old_log_prob_t)
                clipped_ratio = torch.clamp(ratio, 1.0 - args.clip_ratio, 1.0 + args.clip_ratio)
                policy_loss_low = -torch.min(ratio * adv_t, clipped_ratio * adv_t).mean()
                entropy_loss_low = -args.entropy_coeff * entropy_t.mean()
                entropy_mean_low = float(entropy_t.mean().item())
                total_active += int(adv_t.numel())
            else:
                policy_loss_low = value_loss * 0.0
                entropy_loss_low = value_loss * 0.0
                entropy_mean_low = 0.0

            if high_new_log_probs:
                high_new_log_prob_t = torch.cat(high_new_log_probs, dim=0)
                high_old_log_prob_t = torch.cat(high_old_log_probs, dim=0)
                high_adv_t = torch.cat(high_advantages, dim=0)
                high_entropy_t = torch.cat(high_entropies, dim=0)
                high_new_log_prob_t = sanitize_tensor(high_new_log_prob_t)
                high_old_log_prob_t = sanitize_tensor(high_old_log_prob_t)
                high_adv_t = sanitize_tensor(high_adv_t)
                high_entropy_t = sanitize_tensor(high_entropy_t)

                high_ratio = torch.exp(high_new_log_prob_t - high_old_log_prob_t)
                high_clipped_ratio = torch.clamp(high_ratio, 1.0 - args.clip_ratio, 1.0 + args.clip_ratio)
                policy_loss_high = -torch.min(high_ratio * high_adv_t, high_clipped_ratio * high_adv_t).mean()
                entropy_loss_high = -args.high_level_entropy_coeff * high_entropy_t.mean()
                entropy_mean_high = float(high_entropy_t.mean().item())
            else:
                policy_loss_high = value_loss * 0.0
                entropy_loss_high = value_loss * 0.0
                entropy_mean_high = 0.0

            policy_loss = policy_loss_low + args.high_level_loss_coeff * policy_loss_high
            entropy_loss = entropy_loss_low + entropy_loss_high
            entropy_mean = entropy_mean_low + entropy_mean_high

            if imitation_losses:
                imitation_loss = sanitize_tensor(torch.cat(imitation_losses, dim=0)).mean()
            else:
                imitation_loss = value_loss * 0.0

            loss = policy_loss + value_loss + entropy_loss + imitation_coef * imitation_loss
            if not torch.isfinite(loss):
                total_skipped_non_finite += 1
                optimizer.zero_grad()
                continue

            optimizer.zero_grad()
            loss.backward()
            if has_non_finite_gradients(model):
                total_skipped_non_finite += 1
                optimizer.zero_grad()
                repaired = repair_non_finite_parameters(model)
                total_repaired_params += int(repaired)
                continue
            actor_grad_norm = global_grad_norm(actor_params)
            critic_grad_norm = global_grad_norm(critic_params)
            nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()

            repaired = repair_non_finite_parameters(model)
            total_repaired_params += int(repaired)

            total_policy_low += float(policy_loss_low.item())
            total_policy_high += float(policy_loss_high.item())
            total_value += float(value_loss.item())
            total_entropy_low += entropy_mean_low
            total_entropy_high += entropy_mean_high
            total_imitation += float(imitation_loss.item())
            total_minibatches += 1
            total_grad_steps += 1
            total_actor_grad_norm += actor_grad_norm
            total_critic_grad_norm += critic_grad_norm

    if total_minibatches <= 0:
        return {
            "policy_loss": 0.0,
            "policy_loss_low": 0.0,
            "policy_loss_high": 0.0,
            "value_loss": 0.0,
            "entropy": 0.0,
            "entropy_low": 0.0,
            "entropy_high": 0.0,
            "imitation_loss": 0.0,
            "imitation_coef": float(imitation_coef),
            "active_samples": 0,
            "grad_steps": 0,
            "explained_variance": explained_var,
            "actor_grad_norm": 0.0,
            "critic_grad_norm": 0.0,
            "active_mask_ratio": 0.0,
            "hidden_state_init_abs_error": 0.0,
            "skipped_non_finite_batches": int(total_skipped_non_finite),
            "repaired_non_finite_params": int(total_repaired_params),
            "value_target_mean": float(np.mean(returns)),
            "value_target_std": float(np.std(returns)),
            "value_target_norm_mean": float(np.mean(norm_returns)) if value_normalizer is not None else float(np.mean(returns)),
            "value_target_norm_std": float(np.std(norm_returns)) if value_normalizer is not None else float(np.std(returns)),
        }

    active_mask_ratio = 0.0
    if total_train_mask_count > 0:
        active_mask_ratio = float(total_active_mask_count) / float(total_train_mask_count)

    hidden_state_init_abs_error = 0.0
    if total_hidden_init_count > 0:
        hidden_state_init_abs_error = total_hidden_init_err / float(total_hidden_init_count)

    return {
        "policy_loss": (total_policy_low + args.high_level_loss_coeff * total_policy_high) / float(total_minibatches),
        "policy_loss_low": total_policy_low / float(total_minibatches),
        "policy_loss_high": total_policy_high / float(total_minibatches),
        "value_loss": total_value / float(total_minibatches),
        "entropy": (total_entropy_low + total_entropy_high) / float(total_minibatches),
        "entropy_low": total_entropy_low / float(total_minibatches),
        "entropy_high": total_entropy_high / float(total_minibatches),
        "imitation_loss": total_imitation / float(total_minibatches),
        "imitation_coef": float(imitation_coef),
        "active_samples": total_active,
        "grad_steps": total_grad_steps,
        "explained_variance": explained_var,
        "actor_grad_norm": total_actor_grad_norm / float(total_minibatches),
        "critic_grad_norm": total_critic_grad_norm / float(total_minibatches),
        "active_mask_ratio": active_mask_ratio,
        "hidden_state_init_abs_error": hidden_state_init_abs_error,
        "skipped_non_finite_batches": int(total_skipped_non_finite),
        "repaired_non_finite_params": int(total_repaired_params),
        "value_target_mean": float(np.mean(returns)),
        "value_target_std": float(np.std(returns)),
        "value_target_norm_mean": float(np.mean(norm_returns)) if value_normalizer is not None else float(np.mean(returns)),
        "value_target_norm_std": float(np.std(norm_returns)) if value_normalizer is not None else float(np.std(returns)),
    }


def action_distribution_stats(
    course_actions: np.ndarray,
    attack_actions: np.ndarray,
    alive_mask: np.ndarray,
    attack_masks: np.ndarray,
    course_dim: int,
    attack_dim: int,
) -> Dict[str, float]:
    alive = alive_mask > 0.5
    stats: Dict[str, float] = {}

    if np.any(alive):
        selected_course = course_actions[alive]
        selected_attack = attack_actions[alive]
        total = float(selected_course.size)
        for action in range(course_dim):
            stats["course_action_freq_%02d" % action] = float(np.mean(selected_course == action))
        for action in range(attack_dim):
            stats["attack_action_freq_%02d" % action] = float(np.mean(selected_attack == action))

        # Collapse indicator: any legal non-zero attack that is almost never chosen.
        legal_non_zero = attack_masks[..., 1:] & alive[..., None]
        if np.any(legal_non_zero):
            legal_count = np.sum(legal_non_zero, axis=(0, 1, 2)).astype(np.float64)
            pick_count = np.array(
                [np.sum((attack_actions == (a + 1)) & alive) for a in range(attack_dim - 1)],
                dtype=np.float64,
            )
            valid_idx = legal_count > 0
            if np.any(valid_idx):
                usage = np.zeros_like(legal_count)
                usage[valid_idx] = pick_count[valid_idx] / legal_count[valid_idx]
                stats["attack_legal_usage_min_nonzero"] = float(np.min(usage[valid_idx]))
                stats["attack_legal_usage_mean_nonzero"] = float(np.mean(usage[valid_idx]))
            else:
                stats["attack_legal_usage_min_nonzero"] = 0.0
                stats["attack_legal_usage_mean_nonzero"] = 0.0
        else:
            stats["attack_legal_usage_min_nonzero"] = 0.0
            stats["attack_legal_usage_mean_nonzero"] = 0.0

        stats["alive_sample_count"] = total
    else:
        for action in range(course_dim):
            stats["course_action_freq_%02d" % action] = 0.0
        for action in range(attack_dim):
            stats["attack_action_freq_%02d" % action] = 0.0
        stats["attack_legal_usage_min_nonzero"] = 0.0
        stats["attack_legal_usage_mean_nonzero"] = 0.0
        stats["alive_sample_count"] = 0.0

    return stats


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
        global_state_dim = initial_obs["global_state"].shape[0]
        num_agents = initial_obs["local_obs"].shape[0]
        attack_dim = initial_obs["attack_masks"].shape[1]
    finally:
        probe_env.close()

    # Agent identity disambiguation for shared-parameter policy.
    local_obs_dim = int(base_local_obs_dim + (num_agents if args.concat_agent_id_onehot else 0))

    env_spec = {
        "local_obs_dim": int(local_obs_dim),
        "global_state_dim": int(global_state_dim),
        "num_agents": int(num_agents),
        "attack_dim": int(attack_dim),
        "actor_hidden_dim": int(args.hidden_size),
        "use_obs_id_onehot": bool(args.concat_agent_id_onehot),
        "mode_interval": int(args.mode_interval),
    }

    collectors = CollectorPool(args, env_spec)
    collectors.reset(seed=args.seed)

    save_config(args, local_obs_dim, global_state_dim, num_agents, collectors.worker_env_counts)
    print(
        "[collector] num_workers=%d num_envs=%d worker_env_counts=%s obs_id_onehot=%s base_obs_dim=%d final_obs_dim=%d curriculum_stage=%s"
        % (
            collectors.num_workers,
            collectors.num_envs,
            collectors.worker_env_counts,
            str(bool(args.concat_agent_id_onehot)),
            int(base_local_obs_dim),
            int(local_obs_dim),
            str(getattr(args, "runtime_curriculum_name", "full")),
        ),
        flush=True,
    )

    model_cfg = MAPPOModelConfig(
        local_obs_dim=local_obs_dim,
        global_state_dim=global_state_dim,
        num_agents=num_agents,
        hidden_size=args.hidden_size,
        role_embed_dim=args.role_embed_dim,
        course_embed_dim=args.course_embed_dim,
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
        env_steps, update_idx = load_checkpoint(args, model, optimizer, value_normalizer)

    # If resuming to an earlier step in the same tb directory, purge old future steps.
    writer = build_summary_writer(args, purge_step=env_steps if args.resume else None)

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
                collectors = CollectorPool(args, env_spec)
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

            actor_state = export_actor_state_cpu(model)
            buffer, next_value, rollout_diag = rollout(
                collectors, actor_state, model, device, args, episode_stats, value_normalizer
            )
            rollout_wall_time = time.time() - update_start
            print(
                "[stage] update=%d env_steps=%d rollout_finished wall_time_sec=%.2f"
                % (update_idx + 1, env_steps, rollout_wall_time),
                flush=True,
            )
            advantages_team, returns = compute_gae(buffer, next_value, args.gamma, args.gae_lambda)
            advantages_aux = compute_aux_advantages(buffer, args.gamma)

            # Update value normalizer BEFORE ppo_update (so critic trains on fresh stats)
            value_normalizer.update(returns)
            print(
                "[stage] update=%d env_steps=%d starting_ppo_update"
                % (update_idx + 1, env_steps),
                flush=True,
            )

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

            action_stats = action_distribution_stats(
                course_actions=buffer["course_action"],
                attack_actions=buffer["attack_action"],
                alive_mask=buffer["alive_mask"],
                attack_masks=buffer["attack_masks"].astype(bool, copy=False),
                course_dim=model_cfg.course_dim,
                attack_dim=model_cfg.attack_dim,
            )
            print(
                "[stage] update=%d env_steps=%d ppo_update_finished"
                % (update_idx + 1, env_steps),
                flush=True,
            )

            env_steps += args.rollout * collectors.num_envs
            update_idx += 1
            sample_fps = (args.rollout * collectors.num_envs) / max(time.time() - update_start, 1e-6)

            if time.time() - last_log_time >= args.log_every_sec:
                summary = summarize_episode_stats(list(episode_stats))
                reward_mean = summary.get("round_reward_mean", 0.0)
                win_rate = summary.get("win_rate", 0.0)
                log_summary(writer, "train_episode", summary, env_steps)
                if writer is not None:
                    writer.add_scalar("train/policy_loss", float(stats.get("policy_loss", 0.0)), env_steps)
                    writer.add_scalar("train/policy_loss_low", float(stats.get("policy_loss_low", 0.0)), env_steps)
                    writer.add_scalar("train/policy_loss_high", float(stats.get("policy_loss_high", 0.0)), env_steps)
                    writer.add_scalar("train/value_loss", float(stats.get("value_loss", 0.0)), env_steps)
                    writer.add_scalar("train/entropy", float(stats.get("entropy", 0.0)), env_steps)
                    writer.add_scalar("train/entropy_low", float(stats.get("entropy_low", 0.0)), env_steps)
                    writer.add_scalar("train/entropy_high", float(stats.get("entropy_high", 0.0)), env_steps)
                    writer.add_scalar("train/imitation_loss", float(stats.get("imitation_loss", 0.0)), env_steps)
                    writer.add_scalar("train/imitation_coef", float(stats.get("imitation_coef", 0.0)), env_steps)
                    writer.add_scalar("train/explained_variance", float(stats.get("explained_variance", 0.0)), env_steps)
                    writer.add_scalar("train/actor_grad_norm", float(stats.get("actor_grad_norm", 0.0)), env_steps)
                    writer.add_scalar("train/critic_grad_norm", float(stats.get("critic_grad_norm", 0.0)), env_steps)
                    writer.add_scalar("train/active_mask_ratio", float(stats.get("active_mask_ratio", 0.0)), env_steps)
                    writer.add_scalar(
                        "train/skipped_non_finite_batches",
                        float(stats.get("skipped_non_finite_batches", 0.0)),
                        env_steps,
                    )
                    writer.add_scalar(
                        "train/repaired_non_finite_params",
                        float(stats.get("repaired_non_finite_params", 0.0)),
                        env_steps,
                    )
                    writer.add_scalar(
                        "train/hidden_state_init_abs_error",
                        float(stats.get("hidden_state_init_abs_error", 0.0)),
                        env_steps,
                    )
                    writer.add_scalar("train/value_target_mean", float(stats.get("value_target_mean", 0.0)), env_steps)
                    writer.add_scalar("train/value_target_std", float(stats.get("value_target_std", 0.0)), env_steps)
                    writer.add_scalar(
                        "train/value_target_norm_mean",
                        float(stats.get("value_target_norm_mean", 0.0)),
                        env_steps,
                    )
                    writer.add_scalar(
                        "train/value_target_norm_std",
                        float(stats.get("value_target_norm_std", 0.0)),
                        env_steps,
                    )
                    writer.add_scalar("train/damage_reward_mean", float(rollout_diag.get("damage_reward_mean", 0.0)), env_steps)
                    writer.add_scalar("train/reward_env_mean", float(rollout_diag.get("reward_env_mean", 0.0)), env_steps)
                    writer.add_scalar("train/reward_mode_mean", float(rollout_diag.get("reward_mode_mean", 0.0)), env_steps)
                    writer.add_scalar("train/reward_exec_mean", float(rollout_diag.get("reward_exec_mean", 0.0)), env_steps)
                    writer.add_scalar("train/kill_reward_mean", float(rollout_diag.get("kill_reward_mean", 0.0)), env_steps)
                    writer.add_scalar(
                        "train/survival_reward_mean",
                        float(rollout_diag.get("survival_reward_mean", 0.0)),
                        env_steps,
                    )
                    writer.add_scalar(
                        "train/win_indicator_mean",
                        float(rollout_diag.get("win_indicator_mean", 0.0)),
                        env_steps,
                    )
                    writer.add_scalar(
                        "train/value_contact_mean",
                        float(rollout_diag.get("value_contact_mean", 0.0)),
                        env_steps,
                    )
                    writer.add_scalar(
                        "train/value_opportunity_mean",
                        float(rollout_diag.get("value_opportunity_mean", 0.0)),
                        env_steps,
                    )
                    writer.add_scalar(
                        "train/value_survival_mean",
                        float(rollout_diag.get("value_survival_mean", 0.0)),
                        env_steps,
                    )
                    writer.add_scalar(
                        "train/rnn_hidden_mismatch_count",
                        float(rollout_diag.get("rnn_hidden_mismatch_count", 0)),
                        env_steps,
                    )
                    writer.add_scalar(
                        "train/rnn_hidden_max_abs_diff",
                        float(rollout_diag.get("rnn_hidden_max_abs_diff", 0.0)),
                        env_steps,
                    )
                    writer.add_scalar(
                        "train/obs_agent_id_concat_enabled",
                        1.0 if args.concat_agent_id_onehot else 0.0,
                        env_steps,
                    )
                    for key, value in action_stats.items():
                        writer.add_scalar("train_action/%s" % key, float(value), env_steps)
                    writer.add_scalar("train/sample_fps", float(sample_fps), env_steps)
                    writer.add_scalar("train/active_samples", float(stats.get("active_samples", 0)), env_steps)
                    writer.add_scalar("train/grad_steps", float(stats.get("grad_steps", 0)), env_steps)
                    writer.add_scalar(
                        "curriculum/stage_id",
                        float(getattr(args, "runtime_curriculum_id", 3)),
                        env_steps,
                    )
                    writer.flush()
                print(
                    "[train] env_steps=%d update=%d stage=%s reward_mean=%.2f win_rate=%.3f fps=%.1f policy=%.4f low=%.4f high=%.4f value_loss=%.4f entropy=%.4f imitation=%.4f coef=%.4f ev=%.4f actor_gn=%.4f critic_gn=%.4f active=%.3f skipped_nf=%d repaired_nf=%d hidden_err=%.6f rnn_mismatch=%d grad_steps=%d"
                    % (
                        env_steps,
                        update_idx,
                        str(getattr(args, "runtime_curriculum_name", "full")),
                        reward_mean,
                        win_rate,
                        sample_fps,
                        stats.get("policy_loss", 0.0),
                        stats.get("policy_loss_low", 0.0),
                        stats.get("policy_loss_high", 0.0),
                        stats.get("value_loss", 0.0),
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
                    "[eval] env_steps=%d win_rate=%.3f destroy_balance=%.3f red_down=%.3f blue_down=%.3f saved=%s"
                    % (
                        env_steps,
                        float(eval_summary.get("win_rate", 0.0)),
                        float(eval_summary.get("fighter_destroy_balance_end_mean", 0.0)),
                        float(eval_summary.get("red_fighter_destroyed_end_mean", 0.0)),
                        float(eval_summary.get("blue_fighter_destroyed_end_mean", 0.0)),
                        output_path,
                    ),
                    flush=True,
                )
                next_eval_env_steps += int(args.eval_every_env_steps)

        save_checkpoint(args, model, optimizer, env_steps, update_idx, value_normalizer)
    finally:
        collectors.close()
        if writer is not None:
            writer.close()


if __name__ == "__main__":
    main()
