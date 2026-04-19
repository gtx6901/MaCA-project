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
    track_memory_steps: int = 12
    contact_reward: float = 0.1
    progress_reward_scale: float = 0.002
    progress_reward_cap: float = 20.0
    attack_window_reward: float = 0.1
    agent_aux_reward_scale: float = 0.0
    mode_reward_scale: float = 0.5
    exec_reward_scale: float = 0.2
    disengage_penalty: float = 0.05
    bearing_reward_scale: float = 0.05
    boundary_penalty_margin: float = 120.0
    boundary_penalty_scale: float = 0.03
    boundary_stuck_penalty_enabled: bool = True
    boundary_stuck_ramp_steps: int = 20
    semantic_screen_downsample: int = 4
    terminal_ammo_fail_penalty: float = 80.0
    terminal_participation_penalty: float = 40.0


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
                semantic_screen_observation=True,
                screen_track_memory_steps=max(1, int(self.config.track_memory_steps)),
                delta_course_action=self.config.delta_course_action,
                course_delta_deg=self.config.course_delta_deg,
            )
        )
        self.num_agents = len(self.base_env.agents)
        self.red_fighter_num = int(self.base_env.red_fighter_num)
        self.blue_fighter_num = int(self.base_env.blue_fighter_num)
        self.max_visible_enemies = max(1, int(self.config.max_visible_enemies))
        self._size_x, self._size_y = self.base_env.get_map_size()
        self._track_memory_steps = max(1, int(self.config.track_memory_steps))
        self._screen_downsample = max(1, int(self.config.semantic_screen_downsample))
        raw_screen_h, raw_screen_w, raw_screen_c = self.base_env.observation_spec["screen_shape"]
        self._local_screen_shape = (
            int((int(raw_screen_h) + self._screen_downsample - 1) // self._screen_downsample),
            int((int(raw_screen_w) + self._screen_downsample - 1) // self._screen_downsample),
            int(raw_screen_c),
        )
        self._local_obs_dim = 16 + self.max_visible_enemies * 8
        self._global_state_dim = None
        self._reset_stats()

    @property
    def local_obs_dim(self) -> int:
        return int(self._local_obs_dim)

    @property
    def local_screen_shape(self) -> Tuple[int, int, int]:
        return tuple(self._local_screen_shape)

    @property
    def global_state_dim(self) -> int:
        if self._global_state_dim is None:
            raise RuntimeError("Environment must be reset before reading global_state_dim")
        return int(self._global_state_dim)

    def close(self) -> None:
        self.base_env.close()

    def reset(self, seed: Optional[int] = None):
        obs_dict, infos = self.base_env.reset(seed=seed)
        self.red_fighter_num = int(self.base_env.red_fighter_num)
        self.blue_fighter_num = int(self.base_env.blue_fighter_num)
        self._reset_stats()
        structured = self._build_step_output(obs_dict, infos)
        self._global_state_dim = structured["global_state"].shape[0]
        self._last_course_actions = np.full(self.num_agents, -1, dtype=np.int32)
        self._last_red_destroyed_count = int(infos[self.base_env.agents[0]].get("red_fighter_destroyed_count", 0))
        self._last_blue_destroyed_count = int(infos[self.base_env.agents[0]].get("blue_fighter_destroyed_count", 0))
        red_raw_obs, _blue_raw_obs = self.base_env.get_raw_snapshot()
        if red_raw_obs is not None:
            self._initial_team_missile_total = max(
                1.0,
                float(self._compute_team_missile_total(red_raw_obs["fighter_obs_list"])),
            )
        else:
            self._initial_team_missile_total = 1.0
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

        pre_states = [self._extract_agent_state(red_raw_obs["fighter_obs_list"][idx]) for idx in range(self.num_agents)]

        for idx, agent_id in enumerate(self.base_env.agents):
            fighter_raw_obs = red_raw_obs["fighter_obs_list"][idx]
            alive = bool(fighter_raw_obs["alive"])
            course_action = int(np.clip(actions[idx, 0], 0, COURSE_NUM - 1))
            attack_action = int(np.clip(actions[idx, 1], 0, ATTACK_IND_NUM - 1))
            attack_mask = build_attack_mask_from_raw(fighter_raw_obs)

            if attack_mask[1:].any():
                attack_opportunity_count += 1
                if alive:
                    self._agent_attack_opportunity_counts[idx] += 1.0
            if attack_action > 0:
                fire_count += 1
            if attack_action > 0 and not attack_mask[attack_action]:
                attack_action = 0
            if attack_action == 0 and attack_mask[1:].any():
                missed_attack_count += 1
            if attack_action > 0 and attack_mask[attack_action]:
                executed_fire_count += 1
                if alive:
                    self._agent_fire_counts[idx] += 1.0

            if alive and self._last_course_actions[idx] >= 0 and self._last_course_actions[idx] != course_action:
                self._course_change_counts[idx] += 1
            self._last_course_actions[idx] = course_action
            if alive:
                self._course_visited[idx, course_action] = True

            action_dict[agent_id] = [course_action, attack_action]

        obs_dict, reward_dict, terminations, truncations, infos = self.base_env.step(action_dict)
        red_raw_obs_next, _ = self.base_env.get_raw_snapshot()
        done = bool(any(terminations.values()) or any(truncations.values()))
        raw_info = infos[self.base_env.agents[0]]
        red_destroyed_count = int(raw_info.get("red_fighter_destroyed_count", 0))
        blue_destroyed_count = int(raw_info.get("blue_fighter_destroyed_count", 0))
        blue_alive_count = int(
            raw_info.get("blue_fighter_alive_count", max(self.blue_fighter_num - blue_destroyed_count, 0))
        )
        red_destroyed_delta = max(0, red_destroyed_count - self._last_red_destroyed_count)
        blue_destroyed_delta = max(0, blue_destroyed_count - self._last_blue_destroyed_count)
        self._last_red_destroyed_count = red_destroyed_count
        self._last_blue_destroyed_count = blue_destroyed_count

        kill_reward = float(blue_destroyed_delta) * float(self.config.enemy_attrition_reward)
        survival_reward = -float(red_destroyed_delta) * float(self.config.friendly_attrition_penalty)
        attrition_reward = kill_reward + survival_reward
        agent_aux_reward = np.zeros((self.num_agents,), dtype=np.float32)
        agent_mode_reward = np.zeros((self.num_agents,), dtype=np.float32)
        agent_exec_reward = np.zeros((self.num_agents,), dtype=np.float32)
        agent_progress_reward = np.zeros((self.num_agents,), dtype=np.float32)
        agent_boundary_penalty = np.zeros((self.num_agents,), dtype=np.float32)
        agent_near_boundary = np.zeros((self.num_agents,), dtype=np.float32)
        agent_alive_next = np.zeros((self.num_agents,), dtype=np.float32)
        if red_raw_obs_next is not None:
            for idx in range(self.num_agents):
                next_state = self._extract_agent_state(red_raw_obs_next["fighter_obs_list"][idx])
                aux_components = self._agent_aux_reward_components(idx, pre_states[idx], next_state)
                agent_aux_reward[idx] = aux_components["total"]
                agent_mode_reward[idx] = aux_components["mode_total"]
                agent_exec_reward[idx] = aux_components["exec_total"]
                agent_progress_reward[idx] = aux_components["range_improve"]
                agent_boundary_penalty[idx] = aux_components["boundary_penalty"]
                agent_near_boundary[idx] = aux_components["near_boundary"]
                agent_alive_next[idx] = 1.0 if next_state["alive"] else 0.0

        damage_reward = float(np.mean(list(reward_dict.values())))
        reward_env = damage_reward + attrition_reward
        reward_mode = float(np.mean(agent_mode_reward))
        reward_exec = float(np.mean(agent_exec_reward))
        team_reward = (
            reward_env
            + float(self.config.mode_reward_scale) * reward_mode
            + float(self.config.exec_reward_scale) * reward_exec
        )
        if self.config.agent_aux_reward_scale != 0.0:
            team_reward += float(np.mean(agent_aux_reward)) * float(self.config.agent_aux_reward_scale)

        for idx, agent_id in enumerate(self.base_env.agents):
            self._agent_destroy_contribution[idx] += float(reward_dict.get(agent_id, 0.0))

        reward_terminal = 0.0
        self._terminal_ammo_fail_penalty = 0.0
        self._terminal_participation_penalty = 0.0
        if done and red_raw_obs_next is not None:
            fighter_obs_end = red_raw_obs_next["fighter_obs_list"]
            team_missile_left_end = self._compute_team_missile_total(fighter_obs_end)
            self._agent_missile_left_end = np.asarray(
                [self._compute_agent_missile_left(fighter) for fighter in fighter_obs_end],
                dtype=np.float32,
            )
            self._team_missile_left_ratio_end = float(team_missile_left_end / max(self._initial_team_missile_total, 1.0))
            self._team_missile_left_ratio_end = float(np.clip(self._team_missile_left_ratio_end, 0.0, 1.0))
            missile_spent_ratio = float(np.clip(1.0 - self._team_missile_left_ratio_end, 0.0, 1.0))
            self._enemy_alive_ratio_end = float(blue_destroyed_count < self.blue_fighter_num) * float(
                blue_alive_count / float(max(self.blue_fighter_num, 1))
            )

            if blue_alive_count > 0:
                self._terminal_ammo_fail_penalty = -float(self.config.terminal_ammo_fail_penalty) * self._enemy_alive_ratio_end * missile_spent_ratio

                fire_total = float(np.sum(self._agent_fire_counts))
                if fire_total > 1e-6:
                    sorted_fire = np.sort(self._agent_fire_counts)
                    top2_share = float(np.sum(sorted_fire[-2:]) / fire_total)
                else:
                    top2_share = 1.0
                fire_concentration = float(np.clip(top2_share - 0.5, 0.0, 0.5) / 0.5)

                inactive_mask = (self._contact_counts <= 0.0) & (self._agent_fire_counts <= 0.0)
                inactive_frac = float(np.mean(inactive_mask.astype(np.float32)))
                collapse_index = 0.6 * inactive_frac + 0.4 * fire_concentration
                self._terminal_participation_penalty = (
                    -float(self.config.terminal_participation_penalty)
                    * self._enemy_alive_ratio_end
                    * missile_spent_ratio
                    * float(np.clip(collapse_index, 0.0, 1.0))
                )

            reward_terminal = self._terminal_ammo_fail_penalty + self._terminal_participation_penalty
            team_reward += reward_terminal

        self._episode_return += team_reward
        self._episode_env_return += reward_env
        self._episode_mode_return += reward_mode
        self._episode_exec_return += reward_exec
        self._episode_len += 1
        self._episode_agent_aux_return += agent_aux_reward
        self._episode_engagement_progress_return += agent_progress_reward
        self._episode_boundary_penalty_return += agent_boundary_penalty
        self._near_boundary_counts += agent_near_boundary * agent_alive_next
        self._alive_step_counts += agent_alive_next
        self._attack_opportunity_count += attack_opportunity_count
        self._missed_attack_count += missed_attack_count
        self._fire_action_count += fire_count
        self._executed_fire_action_count += executed_fire_count

        structured = self._build_step_output(obs_dict, infos)
        info = {
            "num_frames": 1,
            "team_reward": team_reward,
            "attrition_reward": attrition_reward,
            "damage_reward": damage_reward,
            "kill_reward": kill_reward,
            "survival_reward": survival_reward,
            "reward_env": reward_env,
            "reward_mode": reward_mode,
            "reward_exec": reward_exec,
            "reward_boundary": float(np.mean(agent_boundary_penalty)),
            "reward_terminal": float(reward_terminal),
            "near_boundary_frac": float(
                np.sum(agent_near_boundary * agent_alive_next) / max(float(np.sum(agent_alive_next)), 1.0)
            ),
            "win_indicator": 0.0,
            "is_active": True,
            "round_reward": float(raw_info.get("round_reward", 0.0)),
            "opponent_round_reward": float(raw_info.get("opponent_round_reward", 0.0)),
            "agent_aux_reward": agent_aux_reward.tolist(),
        }
        if done:
            info["true_reward"] = float(self._episode_return)
            info["episode_extra_stats"] = self._episode_extra_stats(infos)
            info["win_indicator"] = float(
                float(raw_info.get("round_reward", 0.0)) > float(raw_info.get("opponent_round_reward", 0.0))
            )
        return structured, team_reward, done, info

    def _reset_stats(self):
        self._episode_return = 0.0
        self._episode_env_return = 0.0
        self._episode_mode_return = 0.0
        self._episode_exec_return = 0.0
        self._episode_len = 0
        self._episode_agent_aux_return = np.zeros((self.num_agents,), dtype=np.float32)
        self._episode_engagement_progress_return = np.zeros((self.num_agents,), dtype=np.float32)
        self._fire_action_count = 0
        self._executed_fire_action_count = 0
        self._attack_opportunity_count = 0
        self._missed_attack_count = 0
        self._first_contact_step = -1
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
        self._track_distance = np.zeros((self.num_agents,), dtype=np.float32)
        self._track_bearing_deg = np.zeros((self.num_agents,), dtype=np.float32)
        self._track_heading_diff = np.zeros((self.num_agents,), dtype=np.float32)
        self._track_age = np.full((self.num_agents,), self._track_memory_steps, dtype=np.int32)
        self._near_boundary_streak = np.zeros((self.num_agents,), dtype=np.int32)
        self._near_boundary_counts = np.zeros((self.num_agents,), dtype=np.float32)
        self._alive_step_counts = np.zeros((self.num_agents,), dtype=np.float32)
        self._agent_fire_counts = np.zeros((self.num_agents,), dtype=np.float32)
        self._agent_attack_opportunity_counts = np.zeros((self.num_agents,), dtype=np.float32)
        self._agent_destroy_contribution = np.zeros((self.num_agents,), dtype=np.float32)
        self._episode_boundary_penalty_return = np.zeros((self.num_agents,), dtype=np.float32)
        self._initial_team_missile_total = 1.0
        self._terminal_ammo_fail_penalty = 0.0
        self._terminal_participation_penalty = 0.0
        self._team_missile_left_ratio_end = 1.0
        self._enemy_alive_ratio_end = 0.0
        self._agent_missile_left_end = np.zeros((self.num_agents,), dtype=np.float32)

    @staticmethod
    def _compute_team_missile_total(fighter_obs_list: List[Dict[str, object]]) -> float:
        total = 0.0
        for fighter in fighter_obs_list:
            total += float(fighter.get("l_missile_left", 0.0)) + float(fighter.get("s_missile_left", 0.0))
        return float(total)

    @staticmethod
    def _compute_agent_missile_left(fighter_obs: Dict[str, object]) -> float:
        return float(fighter_obs.get("l_missile_left", 0.0)) + float(fighter_obs.get("s_missile_left", 0.0))

    def _build_step_output(self, obs_dict, infos):
        red_raw_obs, blue_raw_obs = self.base_env.get_raw_snapshot()
        if red_raw_obs is None or blue_raw_obs is None:
            raise RuntimeError("Missing raw snapshot in MaCAParallelEnv")

        fighter_obs_list = red_raw_obs["fighter_obs_list"]
        red_alive_count = sum(1 for fighter in fighter_obs_list if fighter["alive"])
        blue_alive_count = sum(1 for fighter in blue_raw_obs["fighter_obs_list"] if fighter["alive"])
        step_frac = float(self._episode_len) / float(max(self.config.max_step, 1))
        local_obs = np.zeros((self.num_agents, self.local_obs_dim), dtype=np.float32)
        local_screen = np.zeros((self.num_agents,) + self.local_screen_shape, dtype=np.uint8)
        attack_masks = np.zeros((self.num_agents, ATTACK_IND_NUM), dtype=np.bool_)
        alive_mask = np.zeros((self.num_agents,), dtype=np.float32)
        rule_visible_target_ids = np.zeros((self.num_agents, self.max_visible_enemies), dtype=np.int64)

        for idx, fighter_raw_obs in enumerate(fighter_obs_list):
            agent_id = self.base_env.agents[idx]
            screen_full = np.asarray(obs_dict[agent_id]["screen"], dtype=np.uint8)
            local_screen[idx] = self._downsample_screen(screen_full)

            attack_mask = build_attack_mask_from_raw(fighter_raw_obs)
            attack_masks[idx] = attack_mask
            alive = bool(fighter_raw_obs["alive"])
            alive_mask[idx] = 1.0 if alive else 0.0
            if not alive:
                local_screen[idx].fill(0)
            sorted_targets = self._sorted_visible_targets(fighter_raw_obs) if alive else []
            for slot_idx in range(min(self.max_visible_enemies, len(sorted_targets))):
                rule_visible_target_ids[idx, slot_idx] = int(sorted_targets[slot_idx].get("id", 0))
            local_obs[idx] = self._build_local_obs(
                agent_idx=idx,
                fighter_raw_obs=fighter_raw_obs,
                step_frac=step_frac,
                red_alive_count=red_alive_count,
                blue_alive_count=blue_alive_count,
            )

            visible_count = len(fighter_raw_obs.get("r_visible_list", [])) if alive else 0
            self._visible_enemy_count_totals[idx] += float(visible_count)
            if visible_count > 0:
                self._contact_counts[idx] += 1
                if self._first_contact_step < 0:
                    self._first_contact_step = int(self._episode_len)
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
            "local_screen": local_screen,
            "global_state": global_state,
            "attack_masks": attack_masks,
            "alive_mask": alive_mask,
            "agent_ids": agent_ids,
            "rule_visible_target_ids": rule_visible_target_ids,
        }

    def _downsample_screen(self, screen: np.ndarray) -> np.ndarray:
        if self._screen_downsample <= 1:
            return screen.astype(np.uint8, copy=False)
        return screen[:: self._screen_downsample, :: self._screen_downsample, :].astype(np.uint8, copy=False)

    def _build_local_obs(
        self,
        agent_idx: int,
        fighter_raw_obs: Dict[str, object],
        step_frac: float,
        red_alive_count: int,
        blue_alive_count: int,
    ) -> np.ndarray:
        obs = np.zeros((self.local_obs_dim,), dtype=np.float32)
        if not fighter_raw_obs["alive"]:
            self._track_age[agent_idx] = self._track_memory_steps
            return obs

        own_x = float(fighter_raw_obs["pos_x"])
        own_y = float(fighter_raw_obs["pos_y"])
        course_rad = float(fighter_raw_obs["course"]) * (np.pi / 180.0)
        recv_count, recv_dir_sin, recv_dir_cos = self._recv_summary(fighter_raw_obs)
        range_rate, bearing_rate, heading_diff, last_seen_age = self._dynamic_track_features(
            agent_idx=agent_idx,
            fighter_raw_obs=fighter_raw_obs,
            course_rad=course_rad,
        )

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
            range_rate,
            bearing_rate,
            heading_diff,
            last_seen_age,
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

    def _dynamic_track_features(self, agent_idx: int, fighter_raw_obs: Dict[str, object], course_rad: float):
        prev_age = int(self._track_age[agent_idx])
        prev_distance = float(self._track_distance[agent_idx])
        prev_bearing = float(self._track_bearing_deg[agent_idx])
        prev_heading_diff = float(self._track_heading_diff[agent_idx])

        visible_targets = self._sorted_visible_targets(fighter_raw_obs)
        if visible_targets:
            target = visible_targets[0]
            own_x = float(fighter_raw_obs["pos_x"])
            own_y = float(fighter_raw_obs["pos_y"])
            dx = float(target["pos_x"]) - own_x
            dy = float(target["pos_y"]) - own_y
            distance = float(np.sqrt(dx * dx + dy * dy))
            rel_bearing_deg = float(np.degrees(np.arctan2(dy, dx) - course_rad))
            rel_bearing_deg = ((rel_bearing_deg + 180.0) % 360.0) - 180.0
            heading_diff = float(np.clip(rel_bearing_deg / 180.0, -1.0, 1.0))

            range_rate = 0.0
            bearing_rate = 0.0
            if prev_age < self._track_memory_steps and prev_distance > 0.0:
                range_rate = float(np.clip((prev_distance - distance) / 100.0, -1.0, 1.0))
                bearing_delta = ((rel_bearing_deg - prev_bearing + 180.0) % 360.0) - 180.0
                bearing_rate = float(np.clip(bearing_delta / 45.0, -1.0, 1.0))

            self._track_distance[agent_idx] = distance
            self._track_bearing_deg[agent_idx] = rel_bearing_deg
            self._track_heading_diff[agent_idx] = heading_diff
            self._track_age[agent_idx] = 0
            return range_rate, bearing_rate, heading_diff, 0.0

        self._track_age[agent_idx] = min(self._track_age[agent_idx] + 1, self._track_memory_steps)
        last_seen_age = float(self._track_age[agent_idx]) / float(max(self._track_memory_steps, 1))
        if prev_age >= self._track_memory_steps:
            prev_heading_diff = 0.0
        return 0.0, 0.0, prev_heading_diff, last_seen_age

    def _extract_agent_state(self, fighter_raw_obs: Dict[str, object]) -> Dict[str, float]:
        alive = bool(fighter_raw_obs.get("alive", False))
        if not alive:
            return {
                "alive": False,
                "has_contact": False,
                "nearest_enemy_distance": 0.0,
                "nearest_enemy_bearing_abs": 1.0,
                "has_attack_opportunity": False,
                "nearest_boundary_dist": 0.0,
            }

        own_x = float(np.clip(float(fighter_raw_obs.get("pos_x", 0.0)), 0.0, float(max(self._size_x, 1.0))))
        own_y = float(np.clip(float(fighter_raw_obs.get("pos_y", 0.0)), 0.0, float(max(self._size_y, 1.0))))
        nearest_boundary_dist = float(
            min(
                own_x,
                max(float(self._size_x) - own_x, 0.0),
                own_y,
                max(float(self._size_y) - own_y, 0.0),
            )
        )

        visible_targets = list(fighter_raw_obs.get("r_visible_list", []))
        has_contact = len(visible_targets) > 0
        if has_contact:
            nearest = min(
                visible_targets,
                key=lambda t: (own_x - float(t.get("pos_x", own_x))) ** 2 + (own_y - float(t.get("pos_y", own_y))) ** 2,
            )
            dx = own_x - float(nearest.get("pos_x", own_x))
            dy = own_y - float(nearest.get("pos_y", own_y))
            nearest_distance = float(np.sqrt(dx * dx + dy * dy))
            course_rad = float(fighter_raw_obs["course"]) * (np.pi / 180.0)
            rel_bearing = float(np.arctan2(-dy, -dx) - course_rad)
            rel_bearing = ((rel_bearing + np.pi) % (2.0 * np.pi)) - np.pi
            nearest_bearing_abs = float(abs(rel_bearing) / np.pi)
        else:
            nearest_distance = 0.0
            nearest_bearing_abs = 1.0

        attack_mask = build_attack_mask_from_raw(fighter_raw_obs)
        return {
            "alive": True,
            "has_contact": has_contact,
            "nearest_enemy_distance": nearest_distance,
            "nearest_enemy_bearing_abs": nearest_bearing_abs,
            "has_attack_opportunity": bool(attack_mask[1:].any()),
            "nearest_boundary_dist": nearest_boundary_dist,
        }

    def _agent_aux_reward_components(
        self,
        agent_idx: int,
        prev_state: Dict[str, float],
        next_state: Dict[str, float],
    ) -> Dict[str, float]:
        if not prev_state["alive"]:
            self._near_boundary_streak[agent_idx] = 0
            return {
                "contact": 0.0,
                "attack_window": 0.0,
                "disengage": 0.0,
                "range_improve": 0.0,
                "bearing_improve": 0.0,
                "boundary_penalty": 0.0,
                "near_boundary": 0.0,
                "mode_total": 0.0,
                "exec_total": 0.0,
                "total": 0.0,
            }

        contact_reward = 0.0
        attack_window_reward = 0.0
        disengage_penalty = 0.0
        range_improve_reward = 0.0
        bearing_improve_reward = 0.0
        boundary_penalty = 0.0
        near_boundary = 0.0
        if next_state["has_contact"] and not prev_state["has_contact"]:
            contact_reward += float(self.config.contact_reward)

        if prev_state["has_contact"] and not next_state["has_contact"]:
            disengage_penalty -= float(self.config.disengage_penalty)

        if (
            prev_state["has_contact"]
            and next_state["has_contact"]
            and prev_state["nearest_enemy_distance"] > 0.0
            and next_state["nearest_enemy_distance"] > 0.0
        ):
            distance_delta = prev_state["nearest_enemy_distance"] - next_state["nearest_enemy_distance"]
            clipped_delta = float(
                np.clip(
                    distance_delta,
                    -float(self.config.progress_reward_cap),
                    float(self.config.progress_reward_cap),
                )
            )
            range_improve_reward += clipped_delta * float(self.config.progress_reward_scale)

            bearing_delta = prev_state["nearest_enemy_bearing_abs"] - next_state["nearest_enemy_bearing_abs"]
            bearing_improve_reward += float(
                np.clip(bearing_delta, -1.0, 1.0) * float(self.config.bearing_reward_scale)
            )

        if next_state["has_attack_opportunity"] and not prev_state["has_attack_opportunity"]:
            attack_window_reward += float(self.config.attack_window_reward)

        margin = float(max(float(self.config.boundary_penalty_margin), 1e-6))
        scale = float(max(float(self.config.boundary_penalty_scale), 0.0))
        dist_to_boundary = float(next_state.get("nearest_boundary_dist", margin))
        if next_state["alive"] and scale > 0.0 and dist_to_boundary < margin:
            near_boundary = 1.0
            self._near_boundary_streak[agent_idx] += 1
            edge_ratio = float(np.clip(1.0 - dist_to_boundary / margin, 0.0, 1.0))
            smooth_edge = edge_ratio * edge_ratio
            idle_multiplier = 1.0 if (not next_state["has_contact"] and not next_state["has_attack_opportunity"]) else 0.35
            stuck_multiplier = 1.0
            if bool(self.config.boundary_stuck_penalty_enabled):
                ramp_steps = int(max(int(self.config.boundary_stuck_ramp_steps), 1))
                stuck_ratio = float(np.clip(float(self._near_boundary_streak[agent_idx]) / float(ramp_steps), 0.0, 1.0))
                stuck_multiplier += 0.75 * stuck_ratio
            boundary_penalty = -scale * smooth_edge * idle_multiplier * stuck_multiplier
        else:
            self._near_boundary_streak[agent_idx] = 0

        mode_total = contact_reward + attack_window_reward + disengage_penalty + boundary_penalty
        exec_total = range_improve_reward + bearing_improve_reward
        total_reward = mode_total + exec_total
        return {
            "contact": float(contact_reward),
            "attack_window": float(attack_window_reward),
            "disengage": float(disengage_penalty),
            "range_improve": float(range_improve_reward),
            "bearing_improve": float(bearing_improve_reward),
            "boundary_penalty": float(boundary_penalty),
            "near_boundary": float(near_boundary),
            "mode_total": float(mode_total),
            "exec_total": float(exec_total),
            "total": float(total_reward),
        }

    def _compute_agent_aux_reward(self, prev_state: Dict[str, float], next_state: Dict[str, float], agent_idx: int = 0) -> float:
        return float(self._agent_aux_reward_components(agent_idx, prev_state, next_state)["total"])

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
        contact_frac = float(np.mean(self._contact_counts)) / episode_len
        attack_opportunity_frac = float(self._attack_opportunity_count) / episode_len
        fire_action_frac = float(self._fire_action_count) / episode_len
        executed_fire_action_frac = float(self._executed_fire_action_count) / episode_len

        denom_contact = max(contact_frac, 1e-6)
        denom_opportunity = max(attack_opportunity_frac, 1e-6)
        contact_to_opportunity_ratio = attack_opportunity_frac / denom_contact
        opportunity_to_fire_ratio = executed_fire_action_frac / denom_opportunity

        if self._first_contact_step >= 0:
            time_to_first_contact = float(self._first_contact_step)
        else:
            time_to_first_contact = float(self._episode_len)

        red_alive_end = float(raw_info.get("red_fighter_alive_count", 0.0))
        red_destroyed_end = float(raw_info.get("red_fighter_destroyed_count", 0.0))
        blue_alive_end = float(raw_info.get("blue_fighter_alive_count", 0.0))
        blue_destroyed_end = float(raw_info.get("blue_fighter_destroyed_count", 0.0))
        timeout_flag = float(self._episode_len >= int(self.config.max_step))
        blue_alive_zero_flag = float(blue_alive_end <= 0.0)
        total_win_flag = float(blue_alive_zero_flag > 0.5 and red_alive_end > 0.0 and timeout_flag < 0.5)

        agent_fire_count_std = float(np.std(self._agent_fire_counts))
        agent_contact_count_std = float(np.std(self._contact_counts.astype(np.float32)))
        agent_missile_left_end_std = float(np.std(self._agent_missile_left_end))
        agent_destroy_contribution_std = float(np.std(self._agent_destroy_contribution))
        agent_attack_opportunity_count_std = float(np.std(self._agent_attack_opportunity_counts))
        alive_steps_total = float(max(np.sum(self._alive_step_counts), 1.0))
        near_boundary_frac = float(np.sum(self._near_boundary_counts) / alive_steps_total)
        boundary_penalty_mean = float(np.sum(self._episode_boundary_penalty_return) / alive_steps_total)

        return {
            "round_reward": float(raw_info.get("round_reward", 0.0)),
            "opponent_round_reward": float(raw_info.get("opponent_round_reward", 0.0)),
            "invalid_action_frac": 0.0,
            "fire_action_frac": fire_action_frac,
            "executed_fire_action_frac": executed_fire_action_frac,
            "attack_opportunity_frac": attack_opportunity_frac,
            "contact_to_opportunity_ratio": contact_to_opportunity_ratio,
            "opportunity_to_fire_ratio": opportunity_to_fire_ratio,
            "time_to_first_contact": time_to_first_contact,
            "missed_attack_frac": float(self._missed_attack_count) / episode_len,
            "course_change_frac": float(np.mean(self._course_change_counts)) / float(max(self._episode_len - 1, 1)),
            "course_unique_frac": float(np.mean(np.count_nonzero(self._course_visited, axis=1) / float(max(COURSE_NUM, 1)))),
            "visible_enemy_count_mean": float(np.mean(self._visible_enemy_count_totals)) / episode_len,
            "contact_frac": contact_frac,
            "attack_window_entry_frac": float(np.mean(self._attack_window_entry_counts)) / episode_len,
            "nearest_enemy_distance_mean": nearest_mean,
            "nearest_enemy_distance_min": nearest_min,
            "engagement_progress_reward_mean": float(np.mean(self._episode_engagement_progress_return)) / episode_len,
            "agent_aux_reward_mean": float(np.mean(self._episode_agent_aux_return)) / episode_len,
            "boundary_penalty_mean": boundary_penalty_mean,
            "near_boundary_frac": near_boundary_frac,
            "reward_env_mean": float(self._episode_env_return) / episode_len,
            "reward_mode_mean": float(self._episode_mode_return) / episode_len,
            "reward_exec_mean": float(self._episode_exec_return) / episode_len,
            "episode_len": float(self._episode_len),
            "win_flag": float(raw_info.get("round_reward", 0.0) > raw_info.get("opponent_round_reward", 0.0)),
            "total_win_flag": total_win_flag,
            "blue_alive_zero_flag": blue_alive_zero_flag,
            "timeout_flag": timeout_flag,
            "red_fighter_alive_end": red_alive_end,
            "red_fighter_destroyed_end": red_destroyed_end,
            "blue_fighter_alive_end": blue_alive_end,
            "blue_fighter_destroyed_end": blue_destroyed_end,
            "fighter_destroy_balance_end": blue_destroyed_end - red_destroyed_end,
            "team_missile_left_ratio_end": float(self._team_missile_left_ratio_end),
            "terminal_ammo_fail_penalty": float(self._terminal_ammo_fail_penalty),
            "terminal_participation_penalty": float(self._terminal_participation_penalty),
            "enemy_alive_ratio_end": float(self._enemy_alive_ratio_end),
            "agent_fire_count_std": agent_fire_count_std,
            "agent_contact_count_std": agent_contact_count_std,
            "agent_missile_left_end_std": agent_missile_left_end_std,
            "agent_destroy_contribution_std": agent_destroy_contribution_std,
            "agent_attack_opportunity_count_std": agent_attack_opportunity_count_std,
        }
