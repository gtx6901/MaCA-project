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
    get_missed_attack_penalty,
    get_progress_reward_cap,
    get_progress_reward_scale,
)


# Heading is a circular quantity: 0° and 360° are identical directions.
# We encode it as (sin, cos) to preserve circular topology instead of linear
# scaling, which would make 359° and 1° appear maximally different to the net.
# The remaining 5 fields (missiles ×2, distance, target id, bearing) are scaled
# linearly.  Total measurement vector dim = 2 + 5 = 7.
_MEASUREMENTS_REST_SCALE = np.asarray([4.0, 4.0, 1500.0, 10.0, 180.0], dtype=np.float32)
_BASE_MEASUREMENT_DIM = 7
_EXTENDED_MEASUREMENT_DIM = 3
_RADAR_TRACK_MEASUREMENT_DIM = 9


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
        self._track_memory_steps = max(1, int(getattr(cfg, "maca_track_memory_steps", 12)))
        self._decoupled_action_heads = bool(getattr(cfg, "maca_decoupled_action_heads", False))
        self._measurement_dim = _BASE_MEASUREMENT_DIM
        if self._extended_measurements:
            self._measurement_dim += _EXTENDED_MEASUREMENT_DIM
        if self._radar_tracking_observation:
            self._measurement_dim += _RADAR_TRACK_MEASUREMENT_DIM

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
        self._visible_enemy_count_totals = np.zeros(self.num_agents, dtype=np.float32)
        self._contact_counts = np.zeros(self.num_agents, dtype=np.int32)
        self._attack_window_entry_counts = np.zeros(self.num_agents, dtype=np.int32)
        self._nearest_enemy_distance_sums = np.zeros(self.num_agents, dtype=np.float32)
        self._nearest_enemy_distance_counts = np.zeros(self.num_agents, dtype=np.int32)
        self._nearest_enemy_distance_mins = np.full(self.num_agents, np.inf, dtype=np.float32)
        self._engagement_progress_reward_totals = np.zeros(self.num_agents, dtype=np.float32)
        self._track_bearing_sin = np.zeros(self.num_agents, dtype=np.float32)
        self._track_bearing_cos = np.zeros(self.num_agents, dtype=np.float32)
        self._track_distance = np.zeros(self.num_agents, dtype=np.float32)
        self._track_age = np.full(self.num_agents, self._track_memory_steps, dtype=np.int32)
        self._episode_len = 0
        self._missed_attack_penalty = get_missed_attack_penalty()
        self._contact_reward = get_contact_reward()
        self._progress_reward_scale = get_progress_reward_scale()
        self._progress_reward_cap = max(0.0, get_progress_reward_cap())
        self._attack_window_reward = get_attack_window_reward()

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
        self._visible_enemy_count_totals.fill(0.0)
        self._contact_counts.fill(0)
        self._attack_window_entry_counts.fill(0)
        self._nearest_enemy_distance_sums.fill(0.0)
        self._nearest_enemy_distance_counts.fill(0)
        self._nearest_enemy_distance_mins.fill(np.inf)
        self._engagement_progress_reward_totals.fill(0.0)
        self._track_bearing_sin.fill(0.0)
        self._track_bearing_cos.fill(0.0)
        self._track_distance.fill(0.0)
        self._track_age.fill(self._track_memory_steps)
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
            if isinstance(chosen_action, (list, tuple)) and len(chosen_action) >= 2:
                course_action = int(chosen_action[0])
            else:
                chosen_action = int(chosen_action)
                course_action = chosen_action // ATTACK_IND_NUM
            if self._last_course_actions[idx] >= 0 and self._last_course_actions[idx] != course_action:
                self._course_change_counts[idx] += 1
            self._last_course_actions[idx] = course_action
            if 0 <= course_action < self._num_course_actions:
                self._course_visited[idx, course_action] = True
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
            if self._chosen_fire(safe_action):
                self._executed_fire_action_counts[idx] += 1
            action_dict[agent_id] = safe_action

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
        is_alive = np.asarray([agent_obs["alive"]], dtype=np.uint8)
        return {
            "obs": screen,
            "measurements": measurements,
            "action_mask": action_mask,
            "is_alive": is_alive,
        }

    def _normalize_measurements(self, agent_obs: Dict[str, np.ndarray], agent_idx: int) -> np.ndarray:
        info_vec = agent_obs["info"]
        info_vec = np.asarray(info_vec, dtype=np.float32)
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

            track_freshness = 1.0 - float(self._track_age[agent_idx]) / float(max(self._track_memory_steps, 1))
            recv_count = float(np.asarray(agent_obs.get("recv_count", [0.0]), dtype=np.float32)[0])
            recv_direction_sin = float(np.asarray(agent_obs.get("recv_direction_sin", [0.0]), dtype=np.float32)[0])
            recv_direction_cos = float(np.asarray(agent_obs.get("recv_direction_cos", [0.0]), dtype=np.float32)[0])
            recv_dominant_freq = float(np.asarray(agent_obs.get("recv_dominant_freq", [0.0]), dtype=np.float32)[0])
            track_extras = np.asarray(
                [
                    self._track_bearing_sin[agent_idx],
                    self._track_bearing_cos[agent_idx],
                    self._track_distance[agent_idx] / 1500.0,
                    max(0.0, track_freshness),
                    min(recv_count / float(max(self.num_agents, 1)), 1.0),
                    1.0 if recv_count > 0.0 else 0.0,
                    recv_direction_sin,
                    recv_direction_cos,
                    recv_dominant_freq / 10.0,
                ],
                dtype=np.float32,
            )
            measurements = np.concatenate([measurements, track_extras])
        return measurements

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
        return {
            "is_alive": is_alive,
            "visible_enemy_count": visible_enemy_count,
            "has_contact": visible_enemy_count > 0,
            "nearest_enemy_distance": nearest_enemy_distance if visible_enemy_count > 0 else 0.0,
            "has_attack_opportunity": has_attack_opportunity,
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

        return float(reward_bonus), entered_attack_window

    @staticmethod
    def _sanitize_action(action: int, valid_mask: np.ndarray) -> int:
        valid_mask = np.asarray(valid_mask, dtype=np.uint8)
        if valid_mask.shape[0] == ACTION_NUM:
            return SampleFactoryMaCAEnv._sanitize_flat_action(action, valid_mask)
        return SampleFactoryMaCAEnv._sanitize_decoupled_action(action, valid_mask)

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
