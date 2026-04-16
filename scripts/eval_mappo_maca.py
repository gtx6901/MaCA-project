#!/usr/bin/env python
"""Evaluate the lightweight MaCA MAPPO lane."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from marl_env.mappo_env import MAPPOMaCAConfig, MAPPOMaCAEnv
from marl_env.mappo_model import MAPPOModelConfig, TeamActorCritic


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=str, required=True)
    parser.add_argument("--train_dir", type=str, default="train_dir/mappo")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--device", type=str, default="gpu")
    parser.add_argument("--output_json", type=str, default=None)
    parser.add_argument("--progress", type=str, default="True")
    parser.add_argument("--deterministic", type=str, default="True")
    parser.add_argument("--maca_opponent", type=str, default=None)
    parser.add_argument("--maca_render", type=str, default="False")
    parser.add_argument("--seed", type=int, default=1)
    return parser.parse_args()


def str2bool(value: str) -> bool:
    return str(value).strip().lower() not in {"0", "false", "no", "off"}


def device_from_arg(device_arg: str):
    if device_arg == "gpu" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def latest_checkpoint(exp_dir: Path) -> Optional[Path]:
    ckpt_dir = exp_dir / "checkpoint"
    if not ckpt_dir.exists():
        return None
    checkpoints = sorted(ckpt_dir.glob("checkpoint_*.pt"))
    return checkpoints[-1] if checkpoints else None


def summarize_episode_stats(episodes):
    if not episodes:
        return {}
    keys = sorted(episodes[0].keys())
    summary = {"episodes": len(episodes)}
    for key in keys:
        values = [float(ep[key]) for ep in episodes if key in ep]
        if values:
            summary_key = key if key.endswith("_mean") else f"{key}_mean"
            summary[summary_key] = float(np.mean(values))
    if "win_flag_mean" in summary:
        summary["win_rate"] = summary["win_flag_mean"]
    return summary


def load_train_cfg(exp_dir: Path) -> dict:
    cfg_path = exp_dir / "cfg.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing training config: {cfg_path}")
    return json.loads(cfg_path.read_text())


def build_env(train_cfg: dict, args) -> MAPPOMaCAEnv:
    return MAPPOMaCAEnv(
        MAPPOMaCAConfig(
            map_path=train_cfg.get("maca_map_path", "maps/1000_1000_fighter10v10.map"),
            red_obs_ind=train_cfg.get("maca_red_obs_ind", "simple"),
            opponent=args.maca_opponent or train_cfg.get("maca_opponent", "fix_rule"),
            max_step=int(train_cfg.get("maca_max_step", 1000)),
            render=str2bool(args.maca_render),
            random_pos=bool(train_cfg.get("maca_random_pos", False)),
            random_seed=int(args.seed),
            adaptive_support_policy=bool(train_cfg.get("maca_adaptive_support_policy", True)),
            support_search_hold=int(train_cfg.get("maca_support_search_hold", 6)),
            delta_course_action=bool(train_cfg.get("maca_delta_course_action", True)),
            course_delta_deg=float(train_cfg.get("maca_course_delta_deg", 45.0)),
            max_visible_enemies=int(train_cfg.get("maca_max_visible_enemies", 4)),
            friendly_attrition_penalty=float(train_cfg.get("maca_friendly_attrition_penalty", 200.0)),
            enemy_attrition_reward=float(train_cfg.get("maca_enemy_attrition_reward", 100.0)),
            track_memory_steps=int(train_cfg.get("maca_track_memory_steps", 12)),
            contact_reward=float(train_cfg.get("maca_contact_reward", 0.1)),
            progress_reward_scale=float(train_cfg.get("maca_progress_reward_scale", 0.002)),
            progress_reward_cap=float(train_cfg.get("maca_progress_reward_cap", 20.0)),
            attack_window_reward=float(train_cfg.get("maca_attack_window_reward", 0.1)),
            agent_aux_reward_scale=float(train_cfg.get("maca_agent_aux_reward_scale", 0.0)),
        )
    )


def select_actions(model, obs, actor_h, device, deterministic: bool):
    local_obs = torch.as_tensor(obs["local_obs"], dtype=torch.float32, device=device)
    agent_ids = torch.as_tensor(obs["agent_ids"], dtype=torch.long, device=device)
    attack_masks = torch.as_tensor(obs["attack_masks"], dtype=torch.bool, device=device)
    actor_h_t = torch.as_tensor(actor_h, dtype=torch.float32, device=device)

    with torch.no_grad():
        course_logits, _attack_logits_unused, next_actor_h = model.actor_step(local_obs, agent_ids, actor_h_t)
        if deterministic:
            course_action = torch.argmax(course_logits, dim=-1)
        else:
            course_action = torch.distributions.Categorical(logits=course_logits).sample()

        attack_logits = model.attack_logits(next_actor_h, course_action)
        invalid_logit = torch.finfo(attack_logits.dtype).min
        attack_logits = attack_logits.masked_fill(~attack_masks, invalid_logit)
        if deterministic:
            attack_action = torch.argmax(attack_logits, dim=-1)
        else:
            attack_action = torch.distributions.Categorical(logits=attack_logits).sample()

    return np.stack([course_action.cpu().numpy(), attack_action.cpu().numpy()], axis=-1), next_actor_h.cpu().numpy()


def main():
    args = parse_args()
    device = device_from_arg(args.device)
    deterministic = str2bool(args.deterministic)
    progress = str2bool(args.progress)

    exp_dir = Path(args.train_dir) / args.experiment
    train_cfg = load_train_cfg(exp_dir)
    checkpoint = Path(args.checkpoint) if args.checkpoint else latest_checkpoint(exp_dir)
    if checkpoint is None or not checkpoint.exists():
        raise FileNotFoundError(f"No checkpoint found under {exp_dir / 'checkpoint'}")

    env = build_env(train_cfg, args)
    obs = env.reset(seed=args.seed)
    model_cfg = MAPPOModelConfig(
        local_obs_dim=env.local_obs_dim,
        global_state_dim=env.global_state_dim,
        num_agents=env.num_agents,
        hidden_size=int(train_cfg.get("hidden_size", 256)),
        role_embed_dim=int(train_cfg.get("role_embed_dim", 8)),
        course_embed_dim=int(train_cfg.get("course_embed_dim", 16)),
    )
    model = TeamActorCritic(model_cfg).to(device)
    checkpoint_state = torch.load(checkpoint, map_location="cpu")
    model_load = model.load_state_dict(checkpoint_state["model"], strict=False)
    if model_load.missing_keys or model_load.unexpected_keys:
        print(
            "[eval] non-strict load missing=%s unexpected=%s"
            % (model_load.missing_keys, model_load.unexpected_keys),
            flush=True,
        )
    model.eval()
    actor_h = np.zeros((env.num_agents, model.actor_hidden_dim), dtype=np.float32)

    episode_results = []
    start_time = time.time()
    try:
        while len(episode_results) < args.episodes:
            actions, next_actor_h = select_actions(model, obs, actor_h, device, deterministic)
            obs, _reward, done, info = env.step(actions)
            actor_h = next_actor_h
            if not done:
                continue

            episode_results.append(info["episode_extra_stats"])
            if progress:
                ep = episode_results[-1]
                print(
                    "[eval] episode=%d/%d win=%.0f red_down=%.1f blue_down=%.1f contact=%.4f dist=%.2f"
                    % (
                        len(episode_results),
                        args.episodes,
                        ep.get("win_flag", 0.0),
                        ep.get("red_fighter_destroyed_end", 0.0),
                        ep.get("blue_fighter_destroyed_end", 0.0),
                        ep.get("contact_frac", 0.0),
                        ep.get("nearest_enemy_distance_mean", 0.0),
                    ),
                    flush=True,
                )
            actor_h.fill(0.0)
            obs = env.reset(seed=args.seed + len(episode_results))
    finally:
        env.close()

    summary = summarize_episode_stats(episode_results)
    payload = {
        "experiment": args.experiment,
        "checkpoint": str(checkpoint),
        "episodes": args.episodes,
        "deterministic": deterministic,
        "maca_opponent": args.maca_opponent or train_cfg.get("maca_opponent", "fix_rule"),
        "eval_wall_time_sec": float(time.time() - start_time),
        "summary": summary,
        "episodes_detail": episode_results,
    }

    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
