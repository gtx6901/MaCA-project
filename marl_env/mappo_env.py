"""Structured multi-agent PPO environment wrapper for MaCA.

This wrapper is intentionally independent from the Sample Factory adapter.
It exposes:

- structured local observations per fighter
- a centralized global state for the critic
- target-specific attack masks built from raw visible enemies

The design goal is to keep the trainer simple while removing two major
limitations of the old training lane:

1. attack legality was collapsed to only the nearest visible target
2. value learning had no access to centralized state
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from fighter_action_utils import (
    ATTACK_IND_NUM,
    COURSE_NUM,
    build_attack_mask_from_raw,
)
from marl_env.maca_parallel_env import EnvConfig, MaCAParallelEnv


@dataclass
class MAPPOMaCAConfig:
    map_path: str = "maps/1000_1000_fighter10v10.map"
    red_obs_ind: str = "simple"
    opponent: str = "fix_rule"
    max_step: int = 1000
    render: bool = False
    random_pos: bool = False
    random_seed: int = -1
    adaptive_support_policy: bool = True
    support_search_hold: int = 6
    delta_course_action: bool = True
    course_delta_deg: float = 45.0
    max_visible_enemies: int = 4
    friendly_attrition_penalty: float = 200.0
    enemy_attrition_reward: float = 100.0


class MAPPOMaCAEnv:
    def __init__(self, config: Optional[MAPPOMaCAConfig] = None):
        self.config = config or MAPPOMaCAConfig()
        self.base_env = MaCAParallelEnv(
            EnvConfig(
                map_path=self.config.map_path,
                red_obs_ind=self.config.red_obs_ind,
                opponent=self.config.opponent,
                max_step=self.config.max_step,
                render=self.config.render,
                random_pos=self.config.random_pos,
                random_seed=self.config.random_seed,
                include_global_state=True,
                adaptive_support_policy=self.config.adaptive_support_policy,
                support_search_hold=self.config.support_search_hold,
                semantic_screen_observation=False,
                screen_track_memory_steps=1,
                delta_course_action=self.config.delta_course_action,
                course_delta_deg=self.config.course_delta_deg,
            )
        )
        self.num_agents = len(self.base_env.agents)
        self.max_visible_enemies = max(1, int(self.config.max_visible_enemies))
        self._size_x, self._size_y = self.base_env.get_map_size()
        self._local_obs_dim = 12 + self.max_visible_enemies * 8
        self._global_state_dim = None
        self._reset_stats()

    @property
    def local_obs_dim(self) -> int:
        return int(self._local_obs_dim)

    @property
    def global_state_dim(self) -> int:
        if self._global_state_dim is None:
            raise RuntimeError("Environment must be reset before reading global_state_dim")
        return int(self._global_state_dim)

    def close(self) -> None:
        self.base_env.close()

    def reset(self, seed: Optional[int] = None):
        obs_dict, infos = self.base_env.reset(seed=seed)
        self._reset_stats()
        structured = self._build_step_output(obs_dict, infos)
        self._global_state_dim = structured["global_state"].shape[0]
        self._last_course_actions = np.full(self.num_agents, -1, dtype=np.int32)
        self._last_red_destroyed_count = int(infos[self.base_env.agents[0]].get("red_fighter_destroyed_count", 0))
        self._last_blue_destroyed_count = int(infos[self.base_env.agents[0]].get("blue_fighter_destroyed_count", 0))
        return structured

    def step(self, actions: np.ndarray):
        actions = np.asarray(actions, dtype=np.int64)
        if actions.shape != (self.num_agents, 2):
            raise ValueError(f"Expected actions shape {(self.num_agents, 2)}, got {actions.shape}")

        red_raw_obs, _ = self.base_env.get_raw_snapshot()
        if red_raw_obs is None:
            raise RuntimeError("Environment must be reset before stepping")

        action_dict: Dict[str, List[int]] = {}
        attack_opportunity_count = 0
        missed_attack_count = 0
        fire_count = 0
        executed_fire_count = 0

        for idx, agent_id in enumerate(self.base_env.agents):
            fighter_raw_obs = red_raw_obs["fighter_obs_list"][idx]
            alive = bool(fighter_raw_obs["alive"])
            course_action = int(np.clip(actions[idx, 0], 0, COURSE_NUM - 1))
            attack_action = int(np.clip(actions[idx, 1], 0, ATTACK_IND_NUM - 1))
            attack_mask = build_attack_mask_from_raw(fighter_raw_obs)

            if attack_mask[1:].any():
                attack_opportunity_count += 1
            if attack_action > 0:
                fire_count += 1
            if attack_action > 0 and not attack_mask[attack_action]:
                attack_action = 0
            if attack_action == 0 and attack_mask[1:].any():
                missed_attack_count += 1
            if attack_action > 0 and attack_mask[attack_action]:
                executed_fire_count += 1

            if alive and self._last_course_actions[idx] >= 0 and self._last_course_actions[idx] != course_action:
                self._course_change_counts[idx] += 1
            self._last_course_actions[idx] = course_action
            if alive:
                self._course_visited[idx, course_action] = True

            action_dict[agent_id] = [course_action, attack_action]

        obs_dict, reward_dict, terminations, truncations, infos = self.base_env.step(action_dict)
        done = bool(any(terminations.values()) or any(truncations.values()))
        raw_info = infos[self.base_env.agents[0]]
        red_destroyed_count = int(raw_info.get("red_fighter_destroyed_count", 0))
        blue_destroyed_count = int(raw_info.get("blue_fighter_destroyed_count", 0))
        red_destroyed_delta = max(0, red_destroyed_count - self._last_red_destroyed_count)
        blue_destroyed_delta = max(0, blue_destroyed_count - self._last_blue_destroyed_count)
        self._last_red_destroyed_count = red_destroyed_count
        self._last_blue_destroyed_count = blue_destroyed_count

        attrition_reward = (
            float(blue_destroyed_delta) * float(self.config.enemy_attrition_reward)
            - float(red_destroyed_delta) * float(self.config.friendly_attrition_penalty)
        )
        team_reward = float(np.mean(list(reward_dict.values()))) + attrition_reward
        self._episode_return += team_reward
        self._episode_len += 1
        self._attack_opportunity_count += attack_opportunity_count
        self._missed_attack_count += missed_attack_count
        self._fire_action_count += fire_count
        self._executed_fire_action_count += executed_fire_count

        structured = self._build_step_output(obs_dict, infos)
        info = {
            "num_frames": 1,
            "team_reward": team_reward,
            "attrition_reward": attrition_reward,
            "is_active": True,
            "round_reward": float(raw_info.get("round_reward", 0.0)),
            "opponent_round_reward": float(raw_info.get("opponent_round_reward", 0.0)),
        }
        if done:
            info["true_reward"] = float(self._episode_return)
            info["episode_extra_stats"] = self._episode_extra_stats(infos)
        return structured, team_reward, done, info

    def _reset_stats(self):
        self._episode_return = 0.0
        self._episode_len = 0
        self._fire_action_count = 0
        self._executed_fire_action_count = 0
        self._attack_opportunity_count = 0
        self._missed_attack_count = 0
        self._contact_counts = np.zeros((self.num_agents,), dtype=np.int32)
        self._visible_enemy_count_totals = np.zeros((self.num_agents,), dtype=np.float32)
        self._nearest_enemy_distance_sums = np.zeros((self.num_agents,), dtype=np.float32)
        self._nearest_enemy_distance_counts = np.zeros((self.num_agents,), dtype=np.int32)
        self._nearest_enemy_distance_mins = np.full((self.num_agents,), np.inf, dtype=np.float32)
        self._course_change_counts = np.zeros((self.num_agents,), dtype=np.int32)
        self._course_visited = np.zeros((self.num_agents, COURSE_NUM), dtype=np.bool_)
        self._last_course_actions = np.full((self.num_agents,), -1, dtype=np.int32)
        self._attack_window_entry_counts = np.zeros((self.num_agents,), dtype=np.int32)
        self._last_attack_available = np.zeros((self.num_agents,), dtype=np.bool_)

    def _build_step_output(self, obs_dict, infos):
        red_raw_obs, blue_raw_obs = self.base_env.get_raw_snapshot()
        if red_raw_obs is None or blue_raw_obs is None:
            raise RuntimeError("Missing raw snapshot in MaCAParallelEnv")

        fighter_obs_list = red_raw_obs["fighter_obs_list"]
        red_alive_count = sum(1 for fighter in fighter_obs_list if fighter["alive"])
        blue_alive_count = sum(1 for fighter in blue_raw_obs["fighter_obs_list"] if fighter["alive"])
        step_frac = float(self._episode_len) / float(max(self.config.max_step, 1))
        local_obs = np.zeros((self.num_agents, self.local_obs_dim), dtype=np.float32)
        attack_masks = np.zeros((self.num_agents, ATTACK_IND_NUM), dtype=np.bool_)
        alive_mask = np.zeros((self.num_agents,), dtype=np.float32)

        for idx, fighter_raw_obs in enumerate(fighter_obs_list):
            attack_mask = build_attack_mask_from_raw(fighter_raw_obs)
            attack_masks[idx] = attack_mask
            alive = bool(fighter_raw_obs["alive"])
            alive_mask[idx] = 1.0 if alive else 0.0
            local_obs[idx] = self._build_local_obs(
                fighter_raw_obs=fighter_raw_obs,
                step_frac=step_frac,
                red_alive_count=red_alive_count,
                blue_alive_count=blue_alive_count,
            )

            visible_count = len(fighter_raw_obs.get("r_visible_list", [])) if alive else 0
            self._visible_enemy_count_totals[idx] += float(visible_count)
            if visible_count > 0:
                self._contact_counts[idx] += 1
            nearest_distance = float(self._nearest_visible_distance(fighter_raw_obs))
            if nearest_distance > 0.0:
                self._nearest_enemy_distance_sums[idx] += nearest_distance
                self._nearest_enemy_distance_counts[idx] += 1
                self._nearest_enemy_distance_mins[idx] = min(self._nearest_enemy_distance_mins[idx], nearest_distance)

            has_attack_opportunity = bool(attack_mask[1:].any())
            if has_attack_opportunity and not self._last_attack_available[idx]:
                self._attack_window_entry_counts[idx] += 1
            self._last_attack_available[idx] = has_attack_opportunity

        global_state = infos[self.base_env.agents[0]].get("global_state")
        global_state = np.asarray(global_state, dtype=np.float32)
        agent_ids = np.arange(self.num_agents, dtype=np.int64)
        return {
            "local_obs": local_obs,
            "global_state": global_state,
            "attack_masks": attack_masks,
            "alive_mask": alive_mask,
            "agent_ids": agent_ids,
        }

    def _build_local_obs(
        self,
        fighter_raw_obs: Dict[str, object],
        step_frac: float,
        red_alive_count: int,
        blue_alive_count: int,
    ) -> np.ndarray:
        obs = np.zeros((self.local_obs_dim,), dtype=np.float32)
        if not fighter_raw_obs["alive"]:
            return obs

        own_x = float(fighter_raw_obs["pos_x"])
        own_y = float(fighter_raw_obs["pos_y"])
        course_rad = float(fighter_raw_obs["course"]) * (np.pi / 180.0)
        recv_count, recv_dir_sin, recv_dir_cos = self._recv_summary(fighter_raw_obs)

        cursor = 0
        base_features = [
            1.0,
            step_frac,
            np.sin(course_rad),
            np.cos(course_rad),
            own_x / float(max(self._size_x, 1.0)),
            own_y / float(max(self._size_y, 1.0)),
            float(fighter_raw_obs["l_missile_left"]) / 4.0,
            float(fighter_raw_obs["s_missile_left"]) / 4.0,
            recv_count / float(max(self.num_agents, 1)),
            recv_dir_sin,
            recv_dir_cos,
            (blue_alive_count - red_alive_count) / float(max(self.num_agents, 1)),
        ]
        obs[cursor : cursor + len(base_features)] = np.asarray(base_features, dtype=np.float32)
        cursor += len(base_features)

        visible_targets = self._sorted_visible_targets(fighter_raw_obs)
        for slot_idx in range(self.max_visible_enemies):
            start = cursor + slot_idx * 8
            if slot_idx >= len(visible_targets):
                continue

            target = visible_targets[slot_idx]
            dx = float(target["pos_x"]) - own_x
            dy = float(target["pos_y"]) - own_y
            distance = float(np.sqrt(dx * dx + dy * dy))
            rel_bearing = float(np.arctan2(dy, dx) - course_rad)
            features = [
                1.0,
                dx / float(max(self._size_x, 1.0)),
                dy / float(max(self._size_y, 1.0)),
                distance / 1500.0,
                np.sin(rel_bearing),
                np.cos(rel_bearing),
                float(target.get("type", 0.0)) / 2.0,
                float(target.get("id", 0.0)) / float(max(self.num_agents, 1)),
            ]
            obs[start : start + 8] = np.asarray(features, dtype=np.float32)
        return obs

    @staticmethod
    def _recv_summary(fighter_raw_obs: Dict[str, object]) -> Tuple[float, float, float]:
        recv_list = list(fighter_raw_obs.get("j_recv_list", []))
        recv_count = float(len(recv_list))
        if recv_count <= 0:
            return 0.0, 0.0, 0.0

        angles = np.asarray([float(item.get("direction", 0.0)) * (np.pi / 180.0) for item in recv_list], dtype=np.float32)
        recv_dir_sin = float(np.mean(np.sin(angles))) if angles.size > 0 else 0.0
        recv_dir_cos = float(np.mean(np.cos(angles))) if angles.size > 0 else 0.0
        return recv_count, recv_dir_sin, recv_dir_cos

    @staticmethod
    def _sorted_visible_targets(fighter_raw_obs: Dict[str, object]) -> List[Dict[str, object]]:
        own_x = float(fighter_raw_obs["pos_x"])
        own_y = float(fighter_raw_obs["pos_y"])

        def sort_key(target):
            dx = own_x - float(target.get("pos_x", own_x))
            dy = own_y - float(target.get("pos_y", own_y))
            return dx * dx + dy * dy

        visible_targets = list(fighter_raw_obs.get("r_visible_list", []))
        visible_targets.sort(key=sort_key)
        return visible_targets

    @staticmethod
    def _nearest_visible_distance(fighter_raw_obs: Dict[str, object]) -> float:
        visible_targets = MAPPOMaCAEnv._sorted_visible_targets(fighter_raw_obs)
        if not visible_targets:
            return 0.0
        own_x = float(fighter_raw_obs["pos_x"])
        own_y = float(fighter_raw_obs["pos_y"])
        target = visible_targets[0]
        dx = own_x - float(target.get("pos_x", own_x))
        dy = own_y - float(target.get("pos_y", own_y))
        return float(np.sqrt(dx * dx + dy * dy))

    def _episode_extra_stats(self, infos):
        raw_info = infos[self.base_env.agents[0]]
        episode_len = float(max(self._episode_len, 1))
        nearest_counts = np.maximum(self._nearest_enemy_distance_counts, 1)
        nearest_mean = float(np.mean(self._nearest_enemy_distance_sums / nearest_counts))
        nearest_min = float(np.mean(np.where(np.isfinite(self._nearest_enemy_distance_mins), self._nearest_enemy_distance_mins, 0.0)))

        return {
            "round_reward": float(raw_info.get("round_reward", 0.0)),
            "opponent_round_reward": float(raw_info.get("opponent_round_reward", 0.0)),
            "invalid_action_frac": 0.0,
            "fire_action_frac": float(self._fire_action_count) / episode_len,
            "executed_fire_action_frac": float(self._executed_fire_action_count) / episode_len,
            "attack_opportunity_frac": float(self._attack_opportunity_count) / episode_len,
            "missed_attack_frac": float(self._missed_attack_count) / episode_len,
            "course_change_frac": float(np.mean(self._course_change_counts)) / float(max(self._episode_len - 1, 1)),
            "course_unique_frac": float(np.mean(np.count_nonzero(self._course_visited, axis=1) / float(max(COURSE_NUM, 1)))),
            "visible_enemy_count_mean": float(np.mean(self._visible_enemy_count_totals)) / episode_len,
            "contact_frac": float(np.mean(self._contact_counts)) / episode_len,
            "attack_window_entry_frac": float(np.mean(self._attack_window_entry_counts)) / episode_len,
            "nearest_enemy_distance_mean": nearest_mean,
            "nearest_enemy_distance_min": nearest_min,
            "engagement_progress_reward_mean": 0.0,
            "episode_len": float(self._episode_len),
            "win_flag": float(raw_info.get("round_reward", 0.0) > raw_info.get("opponent_round_reward", 0.0)),
            "red_fighter_alive_end": float(raw_info.get("red_fighter_alive_count", 0.0)),
            "red_fighter_destroyed_end": float(raw_info.get("red_fighter_destroyed_count", 0.0)),
            "blue_fighter_alive_end": float(raw_info.get("blue_fighter_alive_count", 0.0)),
            "blue_fighter_destroyed_end": float(raw_info.get("blue_fighter_destroyed_count", 0.0)),
            "fighter_destroy_balance_end": float(raw_info.get("blue_fighter_destroyed_count", 0.0))
            - float(raw_info.get("red_fighter_destroyed_count", 0.0)),
        }
