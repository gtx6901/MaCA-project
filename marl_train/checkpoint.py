from __future__ import annotations

import json
import time
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch


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


def resolve_resume_checkpoint(args, root_dir: Path) -> Optional[Path]:
    if args.resume_checkpoint:
        ckpt_path = Path(args.resume_checkpoint)
        if not ckpt_path.is_absolute():
            ckpt_path = root_dir / ckpt_path
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


def load_checkpoint(args, model, optimizer, value_normalizer=None, root_dir: Optional[Path] = None):
    base_dir = Path(".") if root_dir is None else Path(root_dir)
    ckpt_path = resolve_resume_checkpoint(args, base_dir)
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
