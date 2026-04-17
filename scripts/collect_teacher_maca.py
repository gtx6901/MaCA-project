#!/usr/bin/env python
"""Collect teacher trajectories for MaCA MAPPO behavior cloning warm start."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path
from typing import Mapping

import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
ENV_DIR = ROOT_DIR / "environment"
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(ENV_DIR) not in sys.path:
    sys.path.insert(0, str(ENV_DIR))

from fighter_action_utils import ATTACK_IND_NUM, COURSE_NUM
from marl_env.mappo_env import MAPPOMaCAConfig, MAPPOMaCAEnv


def str2bool(value) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher_agent", type=str, default="fix_rule")
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--max_attempt_episodes", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--output_path", type=str, default="exports/teacher_maca_dataset.npz")
    parser.add_argument("--wins_only", type=str2bool, default=False)
    parser.add_argument("--min_destroy_balance", type=float, default=None)
    parser.add_argument("--min_round_reward", type=float, default=None)

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
    return parser.parse_args()


def build_env(args) -> MAPPOMaCAEnv:
    return MAPPOMaCAEnv(
        MAPPOMaCAConfig(
            map_path=args.maca_map_path,
            red_obs_ind=args.maca_red_obs_ind,
            opponent=args.maca_opponent,
            max_step=args.maca_max_step,
            render=args.maca_render,
            random_pos=args.maca_random_pos,
            random_seed=args.seed,
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
        )
    )


def load_teacher(agent_name: str):
    module = importlib.import_module("agent.%s.agent" % agent_name)
    return module.Agent()


def resolve_teacher_obs_mode(teacher) -> str:
    if not hasattr(teacher, "get_obs_ind"):
        return "simple"
    try:
        mode = str(teacher.get_obs_ind()).strip().lower()
    except Exception:
        mode = "simple"
    return mode or "simple"


def build_teacher_obs(base, obs_mode: str):
    if obs_mode == "simple":
        teacher_obs = base._last_red_obs
        if teacher_obs is None:
            raise RuntimeError("Missing simple teacher observation snapshot from base env")
        return teacher_obs

    red_raw_obs, _blue_raw_obs = base.get_raw_snapshot()
    if red_raw_obs is None:
        raise RuntimeError("Missing raw teacher observation snapshot from base env")
    return red_raw_obs


def wrap_angle_deg(angle_deg: float) -> float:
    return ((angle_deg + 180.0) % 360.0) - 180.0


def absolute_course_to_action_bin(course_deg: float) -> int:
    unit = 360.0 / float(max(COURSE_NUM, 1))
    return int(np.clip(int(round((course_deg % 360.0) / unit)) % COURSE_NUM, 0, COURSE_NUM - 1))


def delta_course_to_action_bin(target_course_deg: float, current_course_deg: float, max_delta_deg: float) -> int:
    delta = wrap_angle_deg(target_course_deg - current_course_deg)
    max_delta = max(1e-6, float(max_delta_deg))
    normalized = np.clip((delta / max_delta + 1.0) * 0.5, 0.0, 1.0)
    return int(np.clip(int(round(normalized * float(max(COURSE_NUM - 1, 1)))), 0, COURSE_NUM - 1))


def _to_int(value, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _parse_attack_from_dict(action: Mapping[str, object]) -> int:
    direct_keys = (
        "attack",
        "attack_action",
        "action4",
        "target_action",
        "fire_action",
    )
    for key in direct_keys:
        if key in action:
            return _to_int(action.get(key, 0), 0)

    target = action.get("hit_target", action.get("target_id", action.get("target", 0)))
    missile_type = action.get("missile_type", action.get("missle_type", action.get("weapon_type", 0)))
    target_id = _to_int(target, 0)
    if target_id <= 0:
        return 0

    # MaCA attack space is [0] + long-range block + short-range block.
    half = max(1, (ATTACK_IND_NUM - 1) // 2)
    missile_type_i = _to_int(missile_type, 0)
    if missile_type_i <= 1:
        return target_id
    return target_id + half


def _parse_fighter_row(raw_action) -> np.ndarray:
    if isinstance(raw_action, Mapping):
        course = raw_action.get("course", raw_action.get("heading", raw_action.get("move_action", 0)))
        attack = _parse_attack_from_dict(raw_action)
        return np.asarray([_to_int(course, 0), 0, 0, _to_int(attack, 0)], dtype=np.int32)

    arr = np.asarray(raw_action)
    flat = arr.reshape(-1)
    if flat.size >= 4:
        return np.asarray([_to_int(flat[0], 0), _to_int(flat[1], 0), _to_int(flat[2], 0), _to_int(flat[3], 0)], dtype=np.int32)
    if flat.size == 2:
        return np.asarray([_to_int(flat[0], 0), 0, 0, _to_int(flat[1], 0)], dtype=np.int32)
    if flat.size == 1:
        return np.asarray([_to_int(flat[0], 0), 0, 0, 0], dtype=np.int32)
    return np.asarray([0, 0, 0, 0], dtype=np.int32)


def normalize_teacher_fighter_actions(fighter_actions, num_agents: int) -> np.ndarray:
    rows = [None] * int(num_agents)

    if isinstance(fighter_actions, Mapping):
        if "fighter_action" in fighter_actions:
            source_rows = list(fighter_actions["fighter_action"])
            for idx in range(min(num_agents, len(source_rows))):
                rows[idx] = source_rows[idx]
        else:
            indexed = []
            for key, value in fighter_actions.items():
                try:
                    idx = int(key)
                except Exception:
                    continue
                if 0 <= idx < num_agents:
                    indexed.append((idx, value))
            if indexed:
                for idx, value in indexed:
                    rows[idx] = value
            else:
                source_rows = list(fighter_actions.values())
                for idx in range(min(num_agents, len(source_rows))):
                    rows[idx] = source_rows[idx]
    else:
        source_rows = list(fighter_actions)
        for idx in range(min(num_agents, len(source_rows))):
            rows[idx] = source_rows[idx]

    normalized = np.zeros((num_agents, 4), dtype=np.int32)
    for idx in range(num_agents):
        if rows[idx] is None:
            continue
        normalized[idx] = _parse_fighter_row(rows[idx])
    return normalized


def teacher_actions_to_mappo(env: MAPPOMaCAEnv, fighter_actions) -> np.ndarray:
    fighter_actions = normalize_teacher_fighter_actions(fighter_actions, env.num_agents)
    actions = np.zeros((env.num_agents, 2), dtype=np.int64)
    red_raw_obs, _ = env.base_env.get_raw_snapshot()
    fighter_raw_obs_list = red_raw_obs["fighter_obs_list"] if red_raw_obs is not None else None

    for idx in range(env.num_agents):
        raw = fighter_actions[idx]
        target_course_deg = float(raw[0])
        attack_action = int(np.clip(int(raw[3]), 0, ATTACK_IND_NUM - 1))

        if 0.0 <= target_course_deg <= float(COURSE_NUM - 1) and abs(target_course_deg - round(target_course_deg)) < 1e-6:
            course_action = int(np.clip(int(round(target_course_deg)), 0, COURSE_NUM - 1))
        elif env.base_env.config.delta_course_action and fighter_raw_obs_list is not None:
            current_course = float(fighter_raw_obs_list[idx]["course"])
            course_action = delta_course_to_action_bin(
                target_course_deg,
                current_course,
                max_delta_deg=env.base_env.config.course_delta_deg,
            )
        else:
            course_action = absolute_course_to_action_bin(target_course_deg)

        actions[idx, 0] = int(course_action)
        actions[idx, 1] = int(attack_action)
    return actions

def infer_mode_labels(obs: dict, mappo_actions: np.ndarray, max_visible_enemies: int = 4) -> np.ndarray:
    """Auto-generate hierarchical mode labels from observation/action snapshots.

    Labels:
      0: search (no contact)
      1: approach (contact but no attack window)
      2: attack_ready (attack window available, no fire)
      3: fire_commit (attack action > 0)
    """

    local_obs = np.asarray(obs["local_obs"], dtype=np.float32)
    attack_masks = np.asarray(obs["attack_masks"], dtype=np.bool_)
    actions = np.asarray(mappo_actions, dtype=np.int64)

    mode_labels = np.zeros((local_obs.shape[0],), dtype=np.int64)
    for idx in range(local_obs.shape[0]):
        attack_action = int(actions[idx, 1])
        if attack_action > 0:
            mode_labels[idx] = 3
            continue

        has_opportunity = bool(attack_masks[idx, 1:].any())
        if has_opportunity:
            mode_labels[idx] = 2
            continue

        # Local obs packs visible targets in slots, each slot starts with valid flag.
        has_contact = False
        base = 16
        for slot in range(max(1, int(max_visible_enemies))):
            pos = base + slot * 8
            if pos < local_obs.shape[1] and local_obs[idx, pos] > 0.5:
                has_contact = True
                break
        mode_labels[idx] = 1 if has_contact else 0
    return mode_labels


def main():
    args = parse_args()

    env = build_env(args)
    teacher = load_teacher(args.teacher_agent)
    teacher_obs_mode = resolve_teacher_obs_mode(teacher)

    base = env.base_env
    teacher.set_map_info(
        base._size_x,
        base._size_y,
        base.red_detector_num,
        base.red_fighter_num,
    )

    accepted_steps = []
    episode_returns = []
    accepted_episode_returns = []
    accepted_episode_lengths = []
    accepted_episode_wins = []
    accepted_episode_destroy_balance = []
    accepted_episode_round_reward = []
    obs = env.reset(seed=args.seed)
    ep_return = 0.0
    accepted_episode_idx = 0
    attempt_episode_idx = 0
    step_cnt = 0
    max_attempts = int(args.max_attempt_episodes) if int(args.max_attempt_episodes) > 0 else int(args.episodes)

    episode_local_obs = []
    episode_agent_ids = []
    episode_attack_masks = []
    episode_alive_mask = []
    episode_course_action = []
    episode_attack_action = []
    episode_mode_label = []

    def flush_episode_if_accepted(episode_stats, episode_return: float):
        nonlocal accepted_episode_idx
        win_flag = float(episode_stats.get("win_flag", 0.0))
        destroy_balance = float(episode_stats.get("fighter_destroy_balance_end", 0.0))
        round_reward = float(episode_stats.get("round_reward", 0.0))

        accepted = True
        if args.wins_only and win_flag < 0.5:
            accepted = False
        if args.min_destroy_balance is not None and destroy_balance < float(args.min_destroy_balance):
            accepted = False
        if args.min_round_reward is not None and round_reward < float(args.min_round_reward):
            accepted = False

        episode_len = len(episode_local_obs)
        if accepted and episode_len > 0:
            accepted_steps.extend(
                [
                    {
                        "local_obs": np.stack(episode_local_obs, axis=0),
                        "agent_ids": np.stack(episode_agent_ids, axis=0),
                        "attack_masks": np.stack(episode_attack_masks, axis=0),
                        "alive_mask": np.stack(episode_alive_mask, axis=0),
                        "course_action": np.stack(episode_course_action, axis=0),
                        "attack_action": np.stack(episode_attack_action, axis=0),
                        "mode_label": np.stack(episode_mode_label, axis=0),
                    }
                ]
            )
            accepted_episode_returns.append(float(episode_return))
            accepted_episode_lengths.append(int(episode_len))
            accepted_episode_wins.append(float(win_flag))
            accepted_episode_destroy_balance.append(float(destroy_balance))
            accepted_episode_round_reward.append(float(round_reward))
            accepted_episode_idx += 1
        return accepted, episode_len, win_flag, destroy_balance, round_reward

    try:
        while accepted_episode_idx < args.episodes and attempt_episode_idx < max_attempts:
            teacher_obs = build_teacher_obs(base, teacher_obs_mode)
            try:
                _detector_action, fighter_action = teacher.get_action(teacher_obs, step_cnt)
            except KeyError as exc:
                # Some rule teachers may still require raw observation fields.
                if teacher_obs_mode == "simple" and "detector_obs_list" in str(exc):
                    teacher_obs = build_teacher_obs(base, obs_mode="raw")
                    _detector_action, fighter_action = teacher.get_action(teacher_obs, step_cnt)
                else:
                    raise
            mappo_actions = teacher_actions_to_mappo(env, fighter_action)

            episode_local_obs.append(np.asarray(obs["local_obs"], dtype=np.float32))
            episode_agent_ids.append(np.asarray(obs["agent_ids"], dtype=np.int64))
            episode_attack_masks.append(np.asarray(obs["attack_masks"], dtype=np.bool_))
            episode_alive_mask.append(np.asarray(obs["alive_mask"], dtype=np.float32))
            episode_course_action.append(np.asarray(mappo_actions[:, 0], dtype=np.int64))
            episode_attack_action.append(np.asarray(mappo_actions[:, 1], dtype=np.int64))
            episode_mode_label.append(
                infer_mode_labels(
                    obs,
                    mappo_actions,
                    max_visible_enemies=int(args.maca_max_visible_enemies),
                )
            )

            obs, reward, done, _info = env.step(mappo_actions)
            ep_return += float(reward)
            step_cnt += 1

            if done:
                episode_returns.append(ep_return)
                attempt_episode_idx += 1
                episode_stats = dict(_info.get("episode_extra_stats", {}))
                accepted, episode_len, win_flag, destroy_balance, round_reward = flush_episode_if_accepted(
                    episode_stats,
                    ep_return,
                )
                print(
                    "[collect] accepted=%d/%d attempt=%d/%d return=%.2f win=%.0f destroy_balance=%.1f round_reward=%.1f steps=%d kept=%s"
                    % (
                        accepted_episode_idx,
                        args.episodes,
                        attempt_episode_idx,
                        max_attempts,
                        ep_return,
                        win_flag,
                        destroy_balance,
                        round_reward,
                        episode_len,
                        "yes" if accepted else "no",
                    ),
                    flush=True,
                )
                ep_return = 0.0
                step_cnt = 0
                episode_local_obs = []
                episode_agent_ids = []
                episode_attack_masks = []
                episode_alive_mask = []
                episode_course_action = []
                episode_attack_action = []
                episode_mode_label = []
                obs = env.reset(seed=args.seed + attempt_episode_idx)
    finally:
        env.close()

    if accepted_episode_idx <= 0:
        raise RuntimeError(
            "No teacher episodes matched filters. attempts=%d wins_only=%s min_destroy_balance=%s min_round_reward=%s"
            % (
                attempt_episode_idx,
                bool(args.wins_only),
                str(args.min_destroy_balance),
                str(args.min_round_reward),
            )
        )

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload_steps = {
        "local_obs": np.concatenate([item["local_obs"] for item in accepted_steps], axis=0),
        "agent_ids": np.concatenate([item["agent_ids"] for item in accepted_steps], axis=0),
        "attack_masks": np.concatenate([item["attack_masks"] for item in accepted_steps], axis=0),
        "alive_mask": np.concatenate([item["alive_mask"] for item in accepted_steps], axis=0),
        "course_action": np.concatenate([item["course_action"] for item in accepted_steps], axis=0),
        "attack_action": np.concatenate([item["attack_action"] for item in accepted_steps], axis=0),
        "mode_label": np.concatenate([item["mode_label"] for item in accepted_steps], axis=0),
    }
    payload = {
        **payload_steps,
        "episode_lengths": np.asarray(accepted_episode_lengths, dtype=np.int32),
        "episode_win_flag": np.asarray(accepted_episode_wins, dtype=np.float32),
        "episode_destroy_balance": np.asarray(accepted_episode_destroy_balance, dtype=np.float32),
        "episode_round_reward": np.asarray(accepted_episode_round_reward, dtype=np.float32),
    }
    np.savez_compressed(str(output_path), **payload)

    summary = {
        "requested_episodes": int(args.episodes),
        "accepted_episodes": int(accepted_episode_idx),
        "attempted_episodes": int(attempt_episode_idx),
        "steps": int(payload_steps["local_obs"].shape[0]),
        "mean_return": float(np.mean(episode_returns)) if episode_returns else 0.0,
        "mean_accepted_return": float(np.mean(accepted_episode_returns)) if accepted_episode_returns else 0.0,
        "mean_accepted_win": float(np.mean(accepted_episode_wins)) if accepted_episode_wins else 0.0,
        "mean_accepted_destroy_balance": float(np.mean(accepted_episode_destroy_balance))
        if accepted_episode_destroy_balance
        else 0.0,
        "teacher_agent": args.teacher_agent,
        "teacher_obs_mode": teacher_obs_mode,
        "wins_only": bool(args.wins_only),
        "min_destroy_balance": args.min_destroy_balance,
        "min_round_reward": args.min_round_reward,
        "output_path": str(output_path),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
