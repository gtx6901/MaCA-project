"""Sample Factory 1.x environment adapter for MaCA."""

from __future__ import annotations

from typing import Dict, List, Optional

import gym
import numpy as np
from gym import spaces

from fighter_action_utils import ACTION_NUM, ATTACK_IND_NUM, COURSE_NUM
from marl_env.maca_parallel_env import EnvConfig, MaCAParallelEnv
from marl_env.runtime_tweaks import (
    get_attack_window_reward,
    get_contact_reward,
    get_enemy_attrition_reward,
    get_friendly_attrition_penalty,
    get_missed_attack_penalty,
    get_progress_reward_cap,
    get_progress_reward_scale,
)


# Heading is a circular quantity: 0° and 360° are identical directions.
# We encode it as (sin, cos) to preserve circular topology instead of linear
# scaling, which would make 359° and 1° appear maximally different to the net.
# The remaining 5 fields (missiles ×2, distance, target id, bearing) are scaled
# linearly. Total measurement vector dim = 2 + 5 = 7.
_MEASUREMENTS_REST_SCALE = np.asarray([4.0, 4.0, 1500.0, 10.0, 180.0], dtype=np.float32)
_BASE_MEASUREMENT_DIM = 7
_EXTENDED_MEASUREMENT_DIM = 3
_RADAR_TRACK_MEASUREMENT_DIM = 6
_RELATIVE_MOTION_MEASUREMENT_DIM = 3
_TACTICAL_MODE_MEASUREMENT_DIM = 4
_LOCK_STATE_MEASUREMENT_DIM = 3
_TEAM_STATUS_MEASUREMENT_DIM = 3
_THREAT_STATE_MEASUREMENT_DIM = 2


class SampleFactoryMaCAEnv(gym.Env):
    """Translate the MaCA parallel env to the Sample Factory multi-agent API."""

    metadata = {"render.modes": ["human"]}

    def __init__(self, full_env_name: str, cfg, env_config=None):
        self.name = full_env_name
        self.cfg = cfg
        self.env_config = env_config
        self.maca_env = MaCAParallelEnv(self._build_config(cfg))

        self.num_agents = len(self.maca_env.agents)
        self.is_multiagent = True
        self._extended_measurements = bool(getattr(cfg, "maca_extended_observation", False))
        self._radar_tracking_observation = bool(getattr(cfg, "maca_radar_tracking_observation", False))
        self._relative_motion_observation = bool(getattr(cfg, "maca_relative_motion_observation", False))
        self._tactical_mode_observation = bool(getattr(cfg, "maca_tactical_mode_observation", False))
        self._course_prior_observation = bool(getattr(cfg, "maca_course_prior_observation", False))
        self._lock_state_observation = bool(getattr(cfg, "maca_lock_state_observation", False))
        self._team_status_observation = bool(getattr(cfg, "maca_team_status_observation", False))
        self._threat_state_observation = bool(getattr(cfg, "maca_threat_state_observation", False))
        self._course_hold_steps = max(1, int(getattr(cfg, "maca_course_hold_steps", 1)))
        self._max_course_change_bins = max(0, int(getattr(cfg, "maca_max_course_change_bins", COURSE_NUM - 1)))
        self._intercept_course_assist = bool(getattr(cfg, "maca_intercept_course_assist", False))
        self._intercept_course_blend = float(np.clip(getattr(cfg, "maca_intercept_course_blend", 0.0), 0.0, 1.0))
        self._intercept_break_hold_bins = max(0, int(getattr(cfg, "maca_intercept_break_hold_bins", 0)))
        self._intercept_lead_deg = max(0.0, float(getattr(cfg, "maca_intercept_lead_deg", 20.0)))
        self._commit_distance = max(40.0, float(getattr(cfg, "maca_commit_distance", 140.0)))
        self._track_memory_steps = max(1, int(getattr(cfg, "maca_track_memory_steps", 12)))
        self._decoupled_action_heads = bool(getattr(cfg, "maca_decoupled_action_heads", False))
        self._measurement_dim = _BASE_MEASUREMENT_DIM
        if self._extended_measurements:
            self._measurement_dim += _EXTENDED_MEASUREMENT_DIM
        if self._radar_tracking_observation:
            self._measurement_dim += _RADAR_TRACK_MEASUREMENT_DIM
        if self._relative_motion_observation:
            self._measurement_dim += _RELATIVE_MOTION_MEASUREMENT_DIM
        if self._tactical_mode_observation:
            self._measurement_dim += _TACTICAL_MODE_MEASUREMENT_DIM
        if self._lock_state_observation:
            self._measurement_dim += _LOCK_STATE_MEASUREMENT_DIM
        if self._team_status_observation:
            self._measurement_dim += _TEAM_STATUS_MEASUREMENT_DIM
        if self._threat_state_observation:
            self._measurement_dim += _THREAT_STATE_MEASUREMENT_DIM

        screen_shape = self.maca_env.observation_spec["screen_shape"]
        obs_shape = (screen_shape[2], screen_shape[0], screen_shape[1])

        if self._decoupled_action_heads:
            self.action_space = spaces.Tuple((spaces.Discrete(COURSE_NUM), spaces.Discrete(ATTACK_IND_NUM)))
            action_mask_shape = (COURSE_NUM + ATTACK_IND_NUM,)
        else:
            self.action_space = spaces.Discrete(self.maca_env.action_space_n)
            action_mask_shape = (ACTION_NUM,)
        self.observation_space = spaces.Dict(
            {
                "obs": spaces.Box(low=0, high=255, shape=obs_shape, dtype=np.uint8),
                "measurements": spaces.Box(low=-2.0, high=2.0, shape=(self._measurement_dim,), dtype=np.float32),
                "action_mask": spaces.Box(low=0, high=1, shape=action_mask_shape, dtype=np.uint8),
                "course_prior": spaces.Box(low=-16.0, high=16.0, shape=(COURSE_NUM,), dtype=np.float32),
                "attack_prior": spaces.Box(low=-8.0, high=8.0, shape=(ATTACK_IND_NUM,), dtype=np.float32),
                "is_alive": spaces.Box(low=0, high=1, shape=(1,), dtype=np.uint8),
            }
        )

        self._last_obs_dict: Optional[Dict[str, Dict[str, np.ndarray]]] = None
        self._episode_returns = np.zeros(self.num_agents, dtype=np.float32)
        self._invalid_action_counts = np.zeros(self.num_agents, dtype=np.int32)
        self._fire_action_counts = np.zeros(self.num_agents, dtype=np.int32)
        self._executed_fire_action_counts = np.zeros(self.num_agents, dtype=np.int32)
        self._attack_opportunity_counts = np.zeros(self.num_agents, dtype=np.int32)
        self._missed_attack_counts = np.zeros(self.num_agents, dtype=np.int32)
        self._course_change_counts = np.zeros(self.num_agents, dtype=np.int32)
        self._num_course_actions = ACTION_NUM // ATTACK_IND_NUM
        self._fire_action_mask = (np.arange(ACTION_NUM, dtype=np.int32) % ATTACK_IND_NUM) > 0
        self._course_visited = np.zeros((self.num_agents, self._num_course_actions), dtype=np.bool_)
        self._last_course_actions = np.full(self.num_agents, -1, dtype=np.int32)
        self._last_executed_course_actions = np.full(self.num_agents, -1, dtype=np.int32)
        self._course_hold_remaining = np.zeros(self.num_agents, dtype=np.int32)
        self._visible_enemy_count_totals = np.zeros(self.num_agents, dtype=np.float32)
        self._contact_counts = np.zeros(self.num_agents, dtype=np.int32)
        self._attack_window_entry_counts = np.zeros(self.num_agents, dtype=np.int32)
        self._nearest_enemy_distance_sums = np.zeros(self.num_agents, dtype=np.float32)
        self._nearest_enemy_distance_counts = np.zeros(self.num_agents, dtype=np.int32)
        self._nearest_enemy_distance_mins = np.full(self.num_agents, np.inf, dtype=np.float32)
        self._engagement_progress_reward_totals = np.zeros(self.num_agents, dtype=np.float32)
        self._track_bearing_sin = np.zeros(self.num_agents, dtype=np.float32)
        self._track_bearing_cos = np.zeros(self.num_agents, dtype=np.float32)
        self._track_bearing_deg = np.zeros(self.num_agents, dtype=np.float32)
        self._track_distance = np.zeros(self.num_agents, dtype=np.float32)
        self._track_age = np.full(self.num_agents, self._track_memory_steps, dtype=np.int32)
        self._track_streak = np.zeros(self.num_agents, dtype=np.int32)
        self._lost_track_steps = np.zeros(self.num_agents, dtype=np.int32)
        self._last_closure_rate = np.zeros(self.num_agents, dtype=np.float32)
        self._last_bearing_rate = np.zeros(self.num_agents, dtype=np.float32)
        self._pursuit_fail_streak = np.zeros(self.num_agents, dtype=np.int32)
        self._episode_len = 0
        self._missed_attack_penalty = get_missed_attack_penalty()
        self._contact_reward = get_contact_reward()
        self._progress_reward_scale = get_progress_reward_scale()
        self._progress_reward_cap = max(0.0, get_progress_reward_cap())
        self._attack_window_reward = get_attack_window_reward()
        self._friendly_attrition_penalty = get_friendly_attrition_penalty()
        self._enemy_attrition_reward = get_enemy_attrition_reward()

    @staticmethod
    def _build_config(cfg) -> EnvConfig:
        return EnvConfig(
            map_path=cfg.maca_map_path,
            red_obs_ind=cfg.maca_red_obs_ind,
            opponent=cfg.maca_opponent,
            max_step=cfg.maca_max_step,
            render=cfg.maca_render,
            random_pos=cfg.maca_random_pos,
            random_seed=-1,
            include_global_state=False,
            adaptive_support_policy=bool(getattr(cfg, "maca_adaptive_support_policy", False)),
            support_search_hold=int(getattr(cfg, "maca_support_search_hold", 6)),
            semantic_screen_observation=bool(getattr(cfg, "maca_semantic_screen_observation", False)),
            screen_track_memory_steps=int(getattr(cfg, "maca_screen_track_memory_steps", 12)),
            delta_course_action=bool(getattr(cfg, "maca_delta_course_action", False)),
            course_delta_deg=float(getattr(cfg, "maca_course_delta_deg", 45.0)),
        )

    def seed(self, seed: Optional[int] = None) -> List[int]:
        return self.maca_env.seed(seed)

    def reset(self):
        obs_dict, _ = self.maca_env.reset()
        self._last_obs_dict = obs_dict
        self._episode_returns.fill(0.0)
        self._invalid_action_counts.fill(0)
        self._fire_action_counts.fill(0)
        self._executed_fire_action_counts.fill(0)
        self._attack_opportunity_counts.fill(0)
        self._missed_attack_counts.fill(0)
        self._course_change_counts.fill(0)
        self._course_visited.fill(False)
        self._last_course_actions.fill(-1)
        self._last_executed_course_actions.fill(-1)
        self._course_hold_remaining.fill(0)
        self._visible_enemy_count_totals.fill(0.0)
        self._contact_counts.fill(0)
        self._attack_window_entry_counts.fill(0)
        self._nearest_enemy_distance_sums.fill(0.0)
        self._nearest_enemy_distance_counts.fill(0)
        self._nearest_enemy_distance_mins.fill(np.inf)
        self._engagement_progress_reward_totals.fill(0.0)
        self._track_bearing_sin.fill(0.0)
        self._track_bearing_cos.fill(0.0)
        self._track_bearing_deg.fill(0.0)
        self._track_distance.fill(0.0)
        self._track_age.fill(self._track_memory_steps)
        self._track_streak.fill(0)
        self._lost_track_steps.fill(0)
        self._last_closure_rate.fill(0.0)
        self._last_bearing_rate.fill(0.0)
        self._pursuit_fail_streak.fill(0)
        self._episode_len = 0
        return self._format_obs_list(obs_dict)

    def step(self, actions):
        if self._last_obs_dict is None:
            raise RuntimeError("Environment must be reset before stepping")

        action_dict = {}
        missed_attack_flags = np.zeros(self.num_agents, dtype=np.bool_)
        prev_states = [self._extract_agent_state(self._last_obs_dict[agent_id]) for agent_id in self.maca_env.agents]
        for idx, agent_id in enumerate(self.maca_env.agents):
            chosen_action = actions[idx]
            if isinstance(chosen_action, np.ndarray):
                chosen_action = chosen_action.tolist()
            raw_valid_mask = self._last_obs_dict[agent_id]["action_mask"]
            valid_mask = self._format_action_mask(raw_valid_mask)
            had_attack_opportunity = self._has_attack_opportunity(valid_mask)
            if had_attack_opportunity:
                self._attack_opportunity_counts[idx] += 1
            chose_fire = self._chosen_fire(chosen_action)
            if chose_fire:
                self._fire_action_counts[idx] += 1
            elif had_attack_opportunity:
                self._missed_attack_counts[idx] += 1
                missed_attack_flags[idx] = True
            safe_action = self._sanitize_action(chosen_action, valid_mask)
            if not self._actions_equal(safe_action, chosen_action):
                self._invalid_action_counts[idx] += 1
            executed_action = self._postprocess_action(idx, safe_action, valid_mask)
            executed_course_action = self._extract_course_action(executed_action)
            if self._last_course_actions[idx] >= 0 and self._last_course_actions[idx] != executed_course_action:
                self._course_change_counts[idx] += 1
            self._last_course_actions[idx] = executed_course_action
            if 0 <= executed_course_action < self._num_course_actions:
                self._course_visited[idx, executed_course_action] = True
            if self._chosen_fire(executed_action):
                self._executed_fire_action_counts[idx] += 1
            action_dict[agent_id] = executed_action

        obs_dict, reward_dict, terminations, truncations, infos = self.maca_env.step(action_dict)
        next_states = {agent_id: self._extract_agent_state(obs_dict[agent_id]) for agent_id in self.maca_env.agents}
        rewards = []
        for idx, agent_id in enumerate(self.maca_env.agents):
            reward_value = float(reward_dict[agent_id])
            if missed_attack_flags[idx] and self._missed_attack_penalty != 0.0:
                reward_value -= self._missed_attack_penalty
            progress_bonus, entered_attack_window = self._compute_engagement_reward(
                prev_states[idx], next_states[agent_id]
            )
            reward_value += progress_bonus
            self._engagement_progress_reward_totals[idx] += progress_bonus
            if entered_attack_window:
                self._attack_window_entry_counts[idx] += 1
            rewards.append(reward_value)
        rewards = np.asarray(rewards, dtype=np.float32)
        dones = [bool(terminations[agent_id] or truncations[agent_id]) for agent_id in self.maca_env.agents]

        self._episode_len += 1
        self._episode_returns += rewards
        for idx, agent_id in enumerate(self.maca_env.agents):
            state = next_states[agent_id]
            self._visible_enemy_count_totals[idx] += state["visible_enemy_count"]
            if state["has_contact"]:
                self._contact_counts[idx] += 1
            if state["nearest_enemy_distance"] > 0.0:
                self._nearest_enemy_distance_sums[idx] += state["nearest_enemy_distance"]
                self._nearest_enemy_distance_counts[idx] += 1
                self._nearest_enemy_distance_mins[idx] = min(
                    self._nearest_enemy_distance_mins[idx], state["nearest_enemy_distance"]
                )

        info_list = []
        for idx, agent_id in enumerate(self.maca_env.agents):
            raw_info = infos[agent_id]
            agent_info = {
                "is_active": raw_info["is_active"],
                "num_frames": 1,
                "round_reward": float(raw_info.get("round_reward", 0.0)),
                "opponent_round_reward": float(raw_info.get("opponent_round_reward", 0.0)),
            }
            if dones[idx]:
                agent_info["true_reward"] = float(self._episode_returns[idx])
                agent_info["episode_extra_stats"] = {
                    "round_reward": agent_info["round_reward"],
                    "opponent_round_reward": agent_info["opponent_round_reward"],
                    "invalid_action_frac": float(self._invalid_action_counts[idx]) / float(max(self._episode_len, 1)),
                    "fire_action_frac": float(self._fire_action_counts[idx]) / float(max(self._episode_len, 1)),
                    "executed_fire_action_frac": float(self._executed_fire_action_counts[idx])
                    / float(max(self._episode_len, 1)),
                    "attack_opportunity_frac": float(self._attack_opportunity_counts[idx])
                    / float(max(self._episode_len, 1)),
                    "missed_attack_frac": float(self._missed_attack_counts[idx]) / float(max(self._episode_len, 1)),
                    "course_change_frac": float(self._course_change_counts[idx])
                    / float(max(self._episode_len - 1, 1)),
                    "course_unique_frac": float(np.count_nonzero(self._course_visited[idx]))
                    / float(max(self._num_course_actions, 1)),
                    "visible_enemy_count_mean": float(self._visible_enemy_count_totals[idx])
                    / float(max(self._episode_len, 1)),
                    "contact_frac": float(self._contact_counts[idx]) / float(max(self._episode_len, 1)),
                    "attack_window_entry_frac": float(self._attack_window_entry_counts[idx])
                    / float(max(self._episode_len, 1)),
                    "nearest_enemy_distance_mean": float(self._nearest_enemy_distance_sums[idx])
                    / float(max(self._nearest_enemy_distance_counts[idx], 1)),
                    "nearest_enemy_distance_min": float(self._nearest_enemy_distance_mins[idx])
                    if np.isfinite(self._nearest_enemy_distance_mins[idx])
                    else 0.0,
                    "engagement_progress_reward_mean": float(self._engagement_progress_reward_totals[idx])
                    / float(max(self._episode_len, 1)),
                    "episode_len": float(self._episode_len),
                    "win_flag": float(agent_info["round_reward"] > agent_info["opponent_round_reward"]),
                    "red_fighter_alive_end": float(raw_info.get("red_fighter_alive_count", 0.0)),
                    "red_fighter_destroyed_end": float(raw_info.get("red_fighter_destroyed_count", 0.0)),
                    "blue_fighter_alive_end": float(raw_info.get("blue_fighter_alive_count", 0.0)),
                    "blue_fighter_destroyed_end": float(raw_info.get("blue_fighter_destroyed_count", 0.0)),
                    "fighter_destroy_balance_end": float(raw_info.get("blue_fighter_destroyed_count", 0.0))
                    - float(raw_info.get("red_fighter_destroyed_count", 0.0)),
                }
            info_list.append(agent_info)

        if any(dones):
            next_obs = self.reset()
        else:
            self._last_obs_dict = obs_dict
            next_obs = self._format_obs_list(obs_dict)

        return next_obs, rewards.tolist(), dones, info_list

    def render(self, mode="human"):
        del mode
        return None

    def close(self):
        self.maca_env.close()
        self._last_obs_dict = None

    def _format_obs_list(self, obs_dict: Dict[str, Dict[str, np.ndarray]]):
        return [self._format_agent_obs(obs_dict[agent_id], idx) for idx, agent_id in enumerate(self.maca_env.agents)]

    def _format_agent_obs(self, agent_obs: Dict[str, np.ndarray], agent_idx: int):
        screen = np.moveaxis(agent_obs["screen"], -1, 0)
        measurements = self._normalize_measurements(agent_obs, agent_idx)
        action_mask = self._format_action_mask(agent_obs["action_mask"])
        course_prior = self._build_course_prior(agent_obs, agent_idx)
        attack_prior = self._build_attack_prior(agent_obs, agent_idx)
        is_alive = np.asarray([agent_obs["alive"]], dtype=np.uint8)
        return {
            "obs": screen,
            "measurements": measurements,
            "action_mask": action_mask,
            "course_prior": course_prior,
            "attack_prior": attack_prior,
            "is_alive": is_alive,
        }

    def _normalize_measurements(self, agent_obs: Dict[str, np.ndarray], agent_idx: int) -> np.ndarray:
        info_vec = agent_obs["info"]
        info_vec = np.asarray(info_vec, dtype=np.float32)
        previous_track_age = int(self._track_age[agent_idx])
        previous_track_distance = float(self._track_distance[agent_idx])
        previous_track_bearing_deg = float(self._track_bearing_deg[agent_idx])
        # Encode heading (degrees, 0-360) as (sin, cos) so that 0° and 360°
        # map to the same point and 359° is adjacent to 1°.
        course_rad = float(info_vec[0]) * (np.pi / 180.0)
        sin_cos = np.asarray([np.sin(course_rad), np.cos(course_rad)], dtype=np.float32)
        rest = info_vec[1:] / _MEASUREMENTS_REST_SCALE
        measurements = np.concatenate([sin_cos, rest])
        if self._extended_measurements:
            visible_enemy_count = float(np.asarray(agent_obs.get("visible_enemy_count", [0.0]), dtype=np.float32)[0])
            has_attack_opportunity = float(
                np.asarray(agent_obs.get("has_attack_opportunity", [0.0]), dtype=np.float32)[0]
            )
            extras = np.asarray(
                [
                    visible_enemy_count / float(max(self.num_agents, 1)),
                    1.0 if visible_enemy_count > 0 else 0.0,
                    has_attack_opportunity,
                ],
                dtype=np.float32,
            )
            measurements = np.concatenate([measurements, extras])
        if self._radar_tracking_observation:
            has_contact = float(np.asarray(agent_obs.get("visible_enemy_count", [0.0]), dtype=np.float32)[0]) > 0.0
            nearest_enemy_distance = float(
                np.asarray(agent_obs.get("nearest_enemy_distance", [0.0]), dtype=np.float32)[0]
            )
            nearest_enemy_bearing_sin = float(
                np.asarray(agent_obs.get("nearest_enemy_bearing_sin", [0.0]), dtype=np.float32)[0]
            )
            nearest_enemy_bearing_cos = float(
                np.asarray(agent_obs.get("nearest_enemy_bearing_cos", [0.0]), dtype=np.float32)[0]
            )
            if has_contact and nearest_enemy_distance > 0.0:
                self._track_bearing_sin[agent_idx] = nearest_enemy_bearing_sin
                self._track_bearing_cos[agent_idx] = nearest_enemy_bearing_cos
                self._track_distance[agent_idx] = nearest_enemy_distance
                self._track_age[agent_idx] = 0
            else:
                self._track_age[agent_idx] = min(self._track_age[agent_idx] + 1, self._track_memory_steps)

            # Keep only the minimal track/receive state that is not repeated by
            # lock-state and threat-state features.
            recv_count = float(np.asarray(agent_obs.get("recv_count", [0.0]), dtype=np.float32)[0])
            recv_direction_sin = float(np.asarray(agent_obs.get("recv_direction_sin", [0.0]), dtype=np.float32)[0])
            recv_direction_cos = float(np.asarray(agent_obs.get("recv_direction_cos", [0.0]), dtype=np.float32)[0])
            track_extras = np.asarray(
                [
                    self._track_bearing_sin[agent_idx],
                    self._track_bearing_cos[agent_idx],
                    self._track_distance[agent_idx] / 1500.0,
                    min(recv_count / float(max(self.num_agents, 1)), 1.0),
                    recv_direction_sin,
                    recv_direction_cos,
                ],
                dtype=np.float32,
            )
            measurements = np.concatenate([measurements, track_extras])
        if self._relative_motion_observation:
            has_contact = float(np.asarray(agent_obs.get("visible_enemy_count", [0.0]), dtype=np.float32)[0]) > 0.0
            nearest_enemy_distance = float(
                np.asarray(agent_obs.get("nearest_enemy_distance", [0.0]), dtype=np.float32)[0]
            )
            nearest_enemy_bearing_sin = float(
                np.asarray(agent_obs.get("nearest_enemy_bearing_sin", [0.0]), dtype=np.float32)[0]
            )
            nearest_enemy_bearing_cos = float(
                np.asarray(agent_obs.get("nearest_enemy_bearing_cos", [0.0]), dtype=np.float32)[0]
            )

            closure_rate = 0.0
            bearing_rate = 0.0

            if has_contact and nearest_enemy_distance > 0.0:
                current_track_bearing_deg = float(
                    np.degrees(np.arctan2(nearest_enemy_bearing_sin, nearest_enemy_bearing_cos))
                )
                if previous_track_age < self._track_memory_steps and previous_track_distance > 0.0:
                    closure_rate = float(
                        np.clip((previous_track_distance - nearest_enemy_distance) / 100.0, -1.0, 1.0)
                    )
                    bearing_delta = ((current_track_bearing_deg - previous_track_bearing_deg + 180.0) % 360.0) - 180.0
                    bearing_rate = float(np.clip(bearing_delta / 45.0, -1.0, 1.0))

                self._track_bearing_deg[agent_idx] = current_track_bearing_deg
                if not self._radar_tracking_observation:
                    self._track_distance[agent_idx] = nearest_enemy_distance
                    self._track_age[agent_idx] = 0
                self._track_streak[agent_idx] = min(self._track_streak[agent_idx] + 1, self._track_memory_steps)
            else:
                if not self._radar_tracking_observation:
                    self._track_age[agent_idx] = min(self._track_age[agent_idx] + 1, self._track_memory_steps)
                self._track_streak[agent_idx] = 0

            relative_motion = np.asarray(
                [
                    closure_rate,
                    bearing_rate,
                    float(self._track_streak[agent_idx]) / float(max(self._track_memory_steps, 1)),
                ],
                dtype=np.float32,
            )
            self._last_closure_rate[agent_idx] = closure_rate
            self._last_bearing_rate[agent_idx] = bearing_rate
            if has_contact and nearest_enemy_distance > 0.0:
                self._lost_track_steps[agent_idx] = 0
                if closure_rate > 0.05:
                    self._pursuit_fail_streak[agent_idx] = 0
                else:
                    self._pursuit_fail_streak[agent_idx] = min(
                        self._pursuit_fail_streak[agent_idx] + 1, self._track_memory_steps
                    )
            else:
                if previous_track_age < self._track_memory_steps:
                    self._lost_track_steps[agent_idx] = min(self._lost_track_steps[agent_idx] + 1, self._track_memory_steps)
                else:
                    self._lost_track_steps[agent_idx] = 0
                self._pursuit_fail_streak[agent_idx] = 0
            measurements = np.concatenate([measurements, relative_motion])
        if self._tactical_mode_observation:
            measurements = np.concatenate([measurements, self._build_tactical_mode(agent_obs, agent_idx)])
        if self._lock_state_observation:
            lock_state = np.asarray(
                [
                    float(self._lost_track_steps[agent_idx]) / float(max(self._track_memory_steps, 1)),
                    float(self._pursuit_fail_streak[agent_idx]) / float(max(self._track_memory_steps, 1)),
                    1.0 if self._last_closure_rate[agent_idx] > 0.05 else 0.0,
                ],
                dtype=np.float32,
            )
            measurements = np.concatenate([measurements, lock_state])
        if self._team_status_observation:
            red_destroyed = float(
                np.asarray(agent_obs.get("red_fighter_destroyed_count", [0.0]), dtype=np.float32)[0]
            )
            blue_destroyed = float(
                np.asarray(agent_obs.get("blue_fighter_destroyed_count", [0.0]), dtype=np.float32)[0]
            )
            total_team = float(max(self.num_agents, 1))
            team_status = np.asarray(
                [
                    red_destroyed / total_team,
                    blue_destroyed / total_team,
                    (blue_destroyed - red_destroyed) / total_team,
                ],
                dtype=np.float32,
            )
            measurements = np.concatenate([measurements, team_status])
        if self._threat_state_observation:
            recv_count = float(np.asarray(agent_obs.get("recv_count", [0.0]), dtype=np.float32)[0])
            red_destroyed = float(
                np.asarray(agent_obs.get("red_fighter_destroyed_count", [0.0]), dtype=np.float32)[0]
            )
            blue_destroyed = float(
                np.asarray(agent_obs.get("blue_fighter_destroyed_count", [0.0]), dtype=np.float32)[0]
            )
            team_disadvantage = max(0.0, red_destroyed - blue_destroyed)
            threat_state = np.asarray(
                [
                    min(recv_count / float(max(self.num_agents, 1)), 1.0),
                    min(team_disadvantage / float(max(self.num_agents, 1)), 1.0),
                ],
                dtype=np.float32,
            )
            measurements = np.concatenate([measurements, threat_state])
        return measurements

    def _build_tactical_mode(self, agent_obs: Dict[str, np.ndarray], agent_idx: int) -> np.ndarray:
        visible_enemy_count = float(np.asarray(agent_obs.get("visible_enemy_count", [0.0]), dtype=np.float32)[0])
        has_attack_opportunity = bool(
            np.asarray(agent_obs.get("has_attack_opportunity", [0.0]), dtype=np.float32)[0] > 0.5
        )
        track_is_fresh = self._track_age[agent_idx] < self._track_memory_steps

        mode = np.zeros((_TACTICAL_MODE_MEASUREMENT_DIM,), dtype=np.float32)
        if has_attack_opportunity:
            mode[3] = 1.0
        elif visible_enemy_count > 0.0:
            mode[2] = 1.0
        elif track_is_fresh:
            mode[1] = 1.0
        else:
            mode[0] = 1.0
        return mode

    def _build_course_prior(self, agent_obs: Dict[str, np.ndarray], agent_idx: int) -> np.ndarray:
        prior = np.zeros((COURSE_NUM,), dtype=np.float32)
        if not self._course_prior_observation:
            return prior

        guidance = self._resolve_course_guidance(agent_obs, agent_idx)
        if guidance is None:
            if self._last_executed_course_actions[agent_idx] >= 0:
                prior[self._last_executed_course_actions[agent_idx]] = 0.5
            else:
                prior[COURSE_NUM // 2] = 0.25
            return prior

        best_bin = guidance["course_bin"]
        distances = self._course_bin_distances(best_bin)
        prior = -distances
        mode = guidance["mode"]
        if mode == "attack":
            prior *= 0.2
        elif mode == "pursue":
            pursue_scale = 1.4 if guidance["closure_healthy"] else 2.0
            prior *= pursue_scale
            prior[best_bin] += 1.0 + 0.25 * float(self._pursuit_fail_streak[agent_idx])
        elif mode == "reacquire":
            prior *= 1.8
            prior[best_bin] += 1.25
        elif mode == "search_emit":
            prior *= 0.85
            prior[best_bin] += 0.5
        else:
            prior *= 0.35
        return prior

    def _build_attack_prior(self, agent_obs: Dict[str, np.ndarray], agent_idx: int) -> np.ndarray:
        prior = np.zeros((ATTACK_IND_NUM,), dtype=np.float32)
        has_attack_opportunity = bool(
            np.asarray(agent_obs.get("has_attack_opportunity", [0.0]), dtype=np.float32)[0] > 0.5
        )
        if not has_attack_opportunity:
            return prior

        nearest_enemy_distance = float(
            np.asarray(agent_obs.get("nearest_enemy_distance", [0.0]), dtype=np.float32)[0]
        )
        track_streak = int(self._track_streak[agent_idx])
        closure_healthy = bool(self._last_closure_rate[agent_idx] > 0.05)

        fire_urgency = 1.0
        if nearest_enemy_distance > 0.0 and nearest_enemy_distance <= self._commit_distance:
            fire_urgency += 0.5
        if nearest_enemy_distance > 0.0 and nearest_enemy_distance <= self._commit_distance * 0.75:
            fire_urgency += 0.35
        if track_streak <= 2:
            fire_urgency += 0.25
        if not closure_healthy:
            fire_urgency += 0.2

        prior[0] = -0.25
        prior[1:] = fire_urgency
        return prior

    def _resolve_course_guidance(self, agent_obs: Dict[str, np.ndarray], agent_idx: int):
        visible_enemy_count = float(np.asarray(agent_obs.get("visible_enemy_count", [0.0]), dtype=np.float32)[0])
        has_attack_opportunity = bool(
            np.asarray(agent_obs.get("has_attack_opportunity", [0.0]), dtype=np.float32)[0] > 0.5
        )
        recv_count = float(np.asarray(agent_obs.get("recv_count", [0.0]), dtype=np.float32)[0])

        relative_bearing_deg = None
        mode = "search"

        if visible_enemy_count > 0.0:
            bearing_sin = float(np.asarray(agent_obs.get("nearest_enemy_bearing_sin", [0.0]), dtype=np.float32)[0])
            bearing_cos = float(np.asarray(agent_obs.get("nearest_enemy_bearing_cos", [0.0]), dtype=np.float32)[0])
            relative_bearing_deg = float(np.degrees(np.arctan2(bearing_sin, bearing_cos)))
            mode = "attack" if has_attack_opportunity else "pursue"
            lead_scale = 0.5 if self._last_closure_rate[agent_idx] > 0.05 else 1.0
            relative_bearing_deg += float(
                np.clip(
                    self._last_bearing_rate[agent_idx] * self._intercept_lead_deg * lead_scale,
                    -self._intercept_lead_deg,
                    self._intercept_lead_deg,
                )
            )
            if not has_attack_opportunity and self._last_closure_rate[agent_idx] <= 0.0 and abs(relative_bearing_deg) > 1e-3:
                extra_pull = min(self._intercept_lead_deg * 0.5, max(5.0, abs(relative_bearing_deg) * 0.25))
                relative_bearing_deg += float(np.sign(relative_bearing_deg) * extra_pull)
        elif self._track_age[agent_idx] < self._track_memory_steps:
            relative_bearing_deg = float(self._track_bearing_deg[agent_idx])
            mode = "reacquire"
            if self._lost_track_steps[agent_idx] > 0 and abs(relative_bearing_deg) > 1e-3:
                relative_bearing_deg += float(
                    np.sign(relative_bearing_deg)
                    * min(self._intercept_lead_deg, 5.0 + 3.0 * float(self._lost_track_steps[agent_idx]))
                )
        elif recv_count > 0.0:
            recv_direction_sin = float(np.asarray(agent_obs.get("recv_direction_sin", [0.0]), dtype=np.float32)[0])
            recv_direction_cos = float(np.asarray(agent_obs.get("recv_direction_cos", [0.0]), dtype=np.float32)[0])
            relative_bearing_deg = float(np.degrees(np.arctan2(recv_direction_sin, recv_direction_cos)))
            mode = "search_emit"

        if relative_bearing_deg is None:
            return None

        course_bin = self._bearing_to_course_bin(agent_obs, relative_bearing_deg)
        return {
            "mode": mode,
            "course_bin": course_bin,
            "closure_healthy": bool(self._last_closure_rate[agent_idx] > 0.05),
        }

    def _bearing_to_course_bin(self, agent_obs: Dict[str, np.ndarray], relative_bearing_deg: float) -> int:
        if getattr(self.maca_env.config, "delta_course_action", False):
            max_delta_deg = max(1.0, float(self.maca_env.config.course_delta_deg))
            clamped_bearing = float(np.clip(relative_bearing_deg, -max_delta_deg, max_delta_deg))
            normalized = (clamped_bearing + max_delta_deg) / (2.0 * max_delta_deg)
            return int(np.clip(round(normalized * (COURSE_NUM - 1)), 0, COURSE_NUM - 1))

        current_course = float(np.asarray(agent_obs["info"], dtype=np.float32)[0])
        target_course = (current_course + relative_bearing_deg) % 360.0
        return int(np.clip(round(target_course / (360.0 / COURSE_NUM)) % COURSE_NUM, 0, COURSE_NUM - 1))

    def _course_bin_distances(self, target_bin: int) -> np.ndarray:
        indices = np.arange(COURSE_NUM, dtype=np.int32)
        if getattr(self.maca_env.config, "delta_course_action", False):
            return np.abs(indices - int(target_bin)).astype(np.float32)
        wrapped = np.abs(indices - int(target_bin))
        return np.minimum(wrapped, COURSE_NUM - wrapped).astype(np.float32)

    def _course_bin_distance(self, src_bin: int, dst_bin: int) -> int:
        src = int(src_bin)
        dst = int(dst_bin)
        if getattr(self.maca_env.config, "delta_course_action", False):
            return abs(src - dst)
        wrapped = abs(src - dst)
        return int(min(wrapped, COURSE_NUM - wrapped))

    def _format_action_mask(self, action_mask: np.ndarray) -> np.ndarray:
        action_mask = np.asarray(action_mask, dtype=np.uint8)
        if not self._decoupled_action_heads:
            return action_mask

        reshaped_mask = action_mask.reshape(COURSE_NUM, ATTACK_IND_NUM)
        course_mask = reshaped_mask.any(axis=1).astype(np.uint8)
        attack_mask = reshaped_mask.any(axis=0).astype(np.uint8)
        return np.concatenate([course_mask, attack_mask], axis=0)

    @staticmethod
    def _extract_agent_state(agent_obs: Dict[str, np.ndarray]) -> Dict[str, float]:
        visible_enemy_count = int(np.asarray(agent_obs.get("visible_enemy_count", [0.0]), dtype=np.float32)[0])
        nearest_enemy_distance = float(
            np.asarray(agent_obs.get("nearest_enemy_distance", [0.0]), dtype=np.float32)[0]
        )
        has_attack_opportunity = bool(
            np.asarray(agent_obs.get("has_attack_opportunity", [0.0]), dtype=np.float32)[0] > 0.5
        )
        is_alive = bool(np.asarray(agent_obs.get("alive", False)).item())
        recv_count = float(np.asarray(agent_obs.get("recv_count", [0.0]), dtype=np.float32)[0])
        red_destroyed_count = float(np.asarray(agent_obs.get("red_fighter_destroyed_count", [0.0]), dtype=np.float32)[0])
        blue_destroyed_count = float(
            np.asarray(agent_obs.get("blue_fighter_destroyed_count", [0.0]), dtype=np.float32)[0]
        )
        return {
            "is_alive": is_alive,
            "visible_enemy_count": visible_enemy_count,
            "has_contact": visible_enemy_count > 0,
            "nearest_enemy_distance": nearest_enemy_distance if visible_enemy_count > 0 else 0.0,
            "has_attack_opportunity": has_attack_opportunity,
            "recv_count": recv_count,
            "red_destroyed_count": red_destroyed_count,
            "blue_destroyed_count": blue_destroyed_count,
            "attrition_balance": blue_destroyed_count - red_destroyed_count,
        }

    def _compute_engagement_reward(self, prev_state: Dict[str, float], next_state: Dict[str, float]):
        reward_bonus = 0.0
        entered_attack_window = False
        if not prev_state["is_alive"]:
            return reward_bonus, entered_attack_window

        if next_state["has_contact"] and not prev_state["has_contact"] and self._contact_reward != 0.0:
            reward_bonus += self._contact_reward

        if (
            self._progress_reward_scale != 0.0
            and prev_state["has_contact"]
            and next_state["has_contact"]
            and prev_state["nearest_enemy_distance"] > 0.0
            and next_state["nearest_enemy_distance"] > 0.0
        ):
            distance_delta = prev_state["nearest_enemy_distance"] - next_state["nearest_enemy_distance"]
            if np.isfinite(distance_delta) and distance_delta > 0.0:
                reward_bonus += min(distance_delta, self._progress_reward_cap) * self._progress_reward_scale

        if next_state["has_attack_opportunity"] and not prev_state["has_attack_opportunity"]:
            entered_attack_window = True
            if self._attack_window_reward != 0.0:
                reward_bonus += self._attack_window_reward

        friendly_losses = max(0.0, next_state["red_destroyed_count"] - prev_state["red_destroyed_count"])
        enemy_losses = max(0.0, next_state["blue_destroyed_count"] - prev_state["blue_destroyed_count"])
        if friendly_losses > 0.0 and self._friendly_attrition_penalty != 0.0:
            reward_bonus -= friendly_losses * self._friendly_attrition_penalty
        if enemy_losses > 0.0 and self._enemy_attrition_reward != 0.0:
            reward_bonus += enemy_losses * self._enemy_attrition_reward

        return float(reward_bonus), entered_attack_window

    @staticmethod
    def _sanitize_action(action: int, valid_mask: np.ndarray) -> int:
        valid_mask = np.asarray(valid_mask, dtype=np.uint8)
        if valid_mask.shape[0] == ACTION_NUM:
            return SampleFactoryMaCAEnv._sanitize_flat_action(action, valid_mask)
        return SampleFactoryMaCAEnv._sanitize_decoupled_action(action, valid_mask)

    def _postprocess_action(self, agent_idx: int, action, valid_mask: np.ndarray):
        if valid_mask.shape[0] == ACTION_NUM:
            return self._postprocess_flat_action(agent_idx, action)
        return self._postprocess_decoupled_action(agent_idx, action)

    def _postprocess_flat_action(self, agent_idx: int, action):
        flat_action = int(action)
        attack_action = flat_action % ATTACK_IND_NUM
        course_action = flat_action // ATTACK_IND_NUM
        processed_course = self._postprocess_course_action(agent_idx, course_action)
        return int(processed_course * ATTACK_IND_NUM + attack_action)

    def _postprocess_decoupled_action(self, agent_idx: int, action):
        if isinstance(action, np.ndarray):
            action = action.tolist()
        if isinstance(action, (list, tuple)) and len(action) >= 2:
            course_action = int(action[0])
            attack_action = int(action[1])
        else:
            course_action = int(action) // ATTACK_IND_NUM
            attack_action = int(action) % ATTACK_IND_NUM
        processed_course = self._postprocess_course_action(agent_idx, course_action)
        return [processed_course, attack_action]

    def _postprocess_course_action(self, agent_idx: int, course_action: int) -> int:
        last_executed = int(self._last_executed_course_actions[agent_idx])
        hold_remaining = int(self._course_hold_remaining[agent_idx])
        guidance = None
        if self._intercept_course_assist and self._last_obs_dict is not None:
            guidance = self._resolve_course_guidance(self._last_obs_dict[self.maca_env.agents[agent_idx]], agent_idx)
        guidance_bin = None if guidance is None else int(guidance["course_bin"])

        if hold_remaining > 0 and last_executed >= 0:
            break_hold_bins = self._intercept_break_hold_bins
            if guidance_bin is None or self._course_bin_distance(last_executed, guidance_bin) <= break_hold_bins:
                self._course_hold_remaining[agent_idx] -= 1
                return last_executed

        processed = int(course_action)
        if guidance is not None and guidance_bin is not None and guidance["mode"] != "attack":
            blend = self._intercept_course_blend
            if guidance["mode"] == "reacquire":
                blend = max(blend, 0.8)
            elif guidance["mode"] == "pursue":
                blend = max(blend, 0.6 + 0.08 * float(min(self._pursuit_fail_streak[agent_idx], 3)))
                if guidance["closure_healthy"]:
                    blend = min(blend, 0.55)
            elif guidance["mode"] == "search_emit":
                blend = max(blend, 0.35)
            processed = int(round((1.0 - blend) * float(processed) + blend * float(guidance_bin)))

        if last_executed >= 0 and self._max_course_change_bins >= 0:
            delta = processed - last_executed
            if delta > self._max_course_change_bins:
                processed = last_executed + self._max_course_change_bins
            elif delta < -self._max_course_change_bins:
                processed = last_executed - self._max_course_change_bins

        processed = int(np.clip(processed, 0, COURSE_NUM - 1))
        self._last_executed_course_actions[agent_idx] = processed
        self._course_hold_remaining[agent_idx] = max(0, self._course_hold_steps - 1)
        return processed

    @staticmethod
    def _extract_course_action(action) -> int:
        if isinstance(action, np.ndarray):
            action = action.tolist()
        if isinstance(action, (list, tuple)) and len(action) >= 2:
            return int(action[0])
        return int(action) // ATTACK_IND_NUM

    @staticmethod
    def _sanitize_flat_action(action: int, valid_mask: np.ndarray) -> int:
        action = int(action)
        if 0 <= action < ACTION_NUM and bool(valid_mask[action]):
            return action

        course_action = max(0, min(ACTION_NUM - 1, action)) // ATTACK_IND_NUM
        fallback = course_action * ATTACK_IND_NUM
        if bool(valid_mask[fallback]):
            return fallback

        valid_indices = np.flatnonzero(valid_mask)
        if valid_indices.size == 0:
            return 0
        return int(valid_indices[0])

    @staticmethod
    def _sanitize_decoupled_action(action, valid_mask: np.ndarray):
        if isinstance(action, np.ndarray):
            action = action.tolist()
        if isinstance(action, (list, tuple)) and len(action) >= 2:
            course_action = int(action[0])
            attack_action = int(action[1])
        else:
            course_action = int(action) // ATTACK_IND_NUM
            attack_action = int(action) % ATTACK_IND_NUM

        course_mask = valid_mask[:COURSE_NUM]
        attack_mask = valid_mask[COURSE_NUM:]

        if not (0 <= course_action < COURSE_NUM and bool(course_mask[course_action])):
            valid_courses = np.flatnonzero(course_mask)
            course_action = int(valid_courses[0]) if valid_courses.size > 0 else 0

        if not (0 <= attack_action < ATTACK_IND_NUM and bool(attack_mask[attack_action])):
            attack_action = 0 if bool(attack_mask[0]) else int(np.flatnonzero(attack_mask)[0])

        return [course_action, attack_action]

    def _has_attack_opportunity(self, valid_mask: np.ndarray) -> bool:
        valid_mask = np.asarray(valid_mask, dtype=np.uint8)
        if valid_mask.shape[0] == ACTION_NUM:
            return bool(np.any(valid_mask[self._fire_action_mask]))
        attack_mask = valid_mask[COURSE_NUM:]
        return bool(np.any(attack_mask[1:]))

    def _chosen_fire(self, action) -> bool:
        if isinstance(action, np.ndarray):
            action = action.tolist()
        if isinstance(action, (list, tuple)) and len(action) >= 2:
            return int(action[1]) > 0
        return (int(action) % ATTACK_IND_NUM) > 0

    @staticmethod
    def _actions_equal(a, b) -> bool:
        if isinstance(a, np.ndarray):
            a = a.tolist()
        if isinstance(b, np.ndarray):
            b = b.tolist()
        return a == b
