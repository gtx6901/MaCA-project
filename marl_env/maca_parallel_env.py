"""Parallel-style multi-agent adapter for the MaCA fighter task.

This module intentionally standardizes the current custom environment into a
dict-based API close to PettingZoo/Gymnasium expectations:

- ``reset() -> observations, infos``
- ``step(action_dict) -> observations, rewards, terminations, truncations, infos``

The wrapper only exposes red-side fighters as learning agents. Opponents remain
pluggable via the existing rule-based agent interface. Dead fighters are kept in
the agent set and marked with ``infos[agent_id]["is_active"] = False`` so
future PPO/APPO integrations can use inactive-agent handling rather than
removing agents mid-episode.
"""

from __future__ import annotations

import importlib
from collections import Counter
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Tuple

import numpy as np

from fighter_action_utils import (
    ACTION_NUM,
    ATTACK_IND_NUM,
    COURSE_NUM,
    DEFAULT_DISTURB_POINT,
    RADAR_POINT_NUM,
    build_valid_action_masks,
    get_valid_attack_indices,
    get_support_action,
)

try:
    import interface
except ImportError as exc:  # pragma: no cover - depends on runtime PYTHONPATH
    raise ImportError(
        "Failed to import MaCA environment interface. "
        "Make sure PYTHONPATH includes both the repo root and ./environment."
    ) from exc


@dataclass(frozen=True)
class EnvConfig:
    map_path: str = "maps/1000_1000_fighter10v10.map"
    red_obs_ind: str = "simple"
    opponent: str = "fix_rule"
    max_step: int = 1000
    render: bool = False
    random_pos: bool = False
    random_seed: int = -1
    include_global_state: bool = True
    adaptive_support_policy: bool = False
    support_search_hold: int = 6
    semantic_screen_observation: bool = False
    screen_track_memory_steps: int = 12
    delta_course_action: bool = False
    course_delta_deg: float = 45.0


class MaCAParallelEnv:
    """Parallel-style wrapper over the current MaCA fighter environment."""

    _RAW_SCREEN_CHANNELS = 5
    _SEMANTIC_SCREEN_CHANNELS = 6
    _SCREEN_SIZE = 100

    def __init__(self, config: Optional[EnvConfig] = None):
        self.config = config or EnvConfig()
        self._current_seed = self.config.random_seed
        self._pending_seed = self.config.random_seed
        self._env = None
        self._opponent_agent = None
        self._last_blue_obs = None
        self._last_red_obs = None
        self._last_red_raw_obs = None
        self._last_blue_raw_obs = None
        self._last_alive_mask = None
        self._last_red_round_reward = 0.0
        self._last_blue_round_reward = 0.0
        self._step_count = 0
        self._rng = np.random.RandomState(0)
        self._support_radar_points = None
        self._support_hold_steps = None
        self._semantic_enemy_memory = None
        self._size_x = None
        self._size_y = None
        self.red_detector_num = 0
        self.red_fighter_num = 0
        self.blue_detector_num = 0
        self.blue_fighter_num = 0
        self.possible_agents: List[str] = []
        self.agents: List[str] = []
        self.action_space_n = ACTION_NUM
        screen_channels = (
            self._SEMANTIC_SCREEN_CHANNELS if self.config.semantic_screen_observation else self._RAW_SCREEN_CHANNELS
        )
        self.observation_spec = {
            "screen_shape": (self._SCREEN_SIZE, self._SCREEN_SIZE, screen_channels),
            "info_shape": (6,),
            "action_mask_shape": (ACTION_NUM,),
        }
        self._build_env(seed=self.config.random_seed)

    def _build_env(self, seed: int = -1) -> None:
        self._current_seed = seed
        self._pending_seed = seed
        self._rng = np.random.RandomState(0 if seed is None or seed < 0 else seed)
        self._opponent_agent = self._load_opponent_agent(self.config.opponent)
        opponent_obs_ind = self._opponent_agent.get_obs_ind()
        self._env = interface.Environment(
            self.config.map_path,
            self.config.red_obs_ind,
            opponent_obs_ind,
            max_step=self.config.max_step,
            render=self.config.render,
            random_pos=self.config.random_pos,
            random_seed=seed,
        )
        self._size_x, self._size_y = self._env.get_map_size()
        (
            self.red_detector_num,
            self.red_fighter_num,
            self.blue_detector_num,
            self.blue_fighter_num,
        ) = self._env.get_unit_num()
        self._opponent_agent.set_map_info(
            self._size_x,
            self._size_y,
            self.blue_detector_num,
            self.blue_fighter_num,
        )
        self.possible_agents = [f"red_fighter_{idx}" for idx in range(self.red_fighter_num)]
        self.agents = list(self.possible_agents)
        self._last_alive_mask = np.zeros(self.red_fighter_num, dtype=np.bool_)
        self._support_radar_points = np.asarray(
            [(idx % RADAR_POINT_NUM) + 1 for idx in range(self.red_fighter_num)], dtype=np.int32
        )
        self._support_hold_steps = np.zeros(self.red_fighter_num, dtype=np.int32)
        self._semantic_enemy_memory = np.zeros(
            (self.red_fighter_num, self._SCREEN_SIZE, self._SCREEN_SIZE), dtype=np.float32
        )

    @staticmethod
    def _load_opponent_agent(agent_name: str):
        module = importlib.import_module(f"agent.{agent_name}.agent")
        return module.Agent()

    def reset(
        self, seed: Optional[int] = None, options: Optional[Mapping[str, object]] = None
    ) -> Tuple[Dict[str, Dict[str, np.ndarray]], Dict[str, Dict[str, object]]]:
        del options
        if seed is not None:
            self._pending_seed = seed
        if self._env is None or self._pending_seed != self._current_seed:
            self._build_env(seed=self._pending_seed)
        self._env.reset()
        self._step_count = 0
        self._last_red_round_reward = 0.0
        self._last_blue_round_reward = 0.0
        if self._semantic_enemy_memory is not None:
            self._semantic_enemy_memory.fill(0.0)
        observations, infos = self._collect_step_output()
        return observations, infos

    def seed(self, seed: Optional[int] = None) -> List[int]:
        if seed is None:
            seed = -1
        self._pending_seed = int(seed)
        return [self._pending_seed]

    def step(
        self, action_dict: Mapping[str, int]
    ) -> Tuple[
        Dict[str, Dict[str, np.ndarray]],
        Dict[str, float],
        Dict[str, bool],
        Dict[str, bool],
        Dict[str, Dict[str, object]],
    ]:
        red_detector_action = []
        red_fighter_action = self._decode_red_actions(action_dict)
        blue_detector_action, blue_fighter_action = self._opponent_agent.get_action(
            self._last_blue_obs, self._step_count
        )
        self._env.step(
            red_detector_action,
            red_fighter_action,
            blue_detector_action,
            blue_fighter_action,
        )
        self._step_count += 1
        observations, infos = self._collect_step_output()
        (
            _red_detector_reward,
            red_fighter_reward,
            red_round_reward,
            _blue_detector_reward,
            _blue_fighter_reward,
            blue_round_reward,
        ) = self._env.get_reward()
        env_done = bool(self._env.get_done())
        timeout = env_done and self._step_count >= self.config.max_step
        red_round_reward_delta = float(red_round_reward) - float(self._last_red_round_reward)
        blue_round_reward_delta = float(blue_round_reward) - float(self._last_blue_round_reward)
        self._last_red_round_reward = float(red_round_reward)
        self._last_blue_round_reward = float(blue_round_reward)

        rewards = {}
        terminations = {}
        truncations = {}
        for idx, agent_id in enumerate(self.agents):
            # Use round-reward delta (instead of absolute round reward) so that
            # victory signal is preserved without being repeatedly accumulated.
            rewards[agent_id] = float(red_fighter_reward[idx]) + red_round_reward_delta
            terminations[agent_id] = bool(env_done and not timeout)
            truncations[agent_id] = bool(timeout)
            infos[agent_id]["round_reward"] = float(red_round_reward)
            infos[agent_id]["round_reward_delta"] = red_round_reward_delta
            infos[agent_id]["opponent_round_reward"] = float(blue_round_reward)
            infos[agent_id]["opponent_round_reward_delta"] = blue_round_reward_delta
        return observations, rewards, terminations, truncations, infos

    def close(self) -> None:
        self._env = None
        self._last_blue_obs = None
        self._last_blue_raw_obs = None
        self._last_red_obs = None
        self._last_red_raw_obs = None
        self._last_red_round_reward = 0.0
        self._last_blue_round_reward = 0.0

    def get_raw_snapshot(self):
        """Expose the latest raw red/blue observations for custom trainers."""
        return self._last_red_raw_obs, self._last_blue_raw_obs

    def get_map_size(self):
        return self._size_x, self._size_y

    def _collect_step_output(
        self,
    ) -> Tuple[Dict[str, Dict[str, np.ndarray]], Dict[str, Dict[str, object]]]:
        red_obs, blue_obs = self._env.get_obs()
        self._last_red_obs = red_obs
        self._last_blue_obs = blue_obs

        red_raw_obs, blue_raw_obs = self._env.get_obs_raw()
        self._last_red_raw_obs = red_raw_obs
        self._last_blue_raw_obs = blue_raw_obs

        global_state = None
        if self.config.include_global_state:
            global_state = self._build_global_state(red_raw_obs, blue_raw_obs)

        fighter_infos = np.stack([fighter["info"] for fighter in red_obs["fighter"]], axis=0)
        raw_fighter_obs = red_raw_obs["fighter_obs_list"]
        valid_masks = build_valid_action_masks(fighter_infos)
        red_alive_count = sum(1 for fighter in red_raw_obs["fighter_obs_list"] if fighter["alive"])
        blue_alive_count = sum(1 for fighter in blue_raw_obs["fighter_obs_list"] if fighter["alive"])
        red_destroyed_count = self.red_fighter_num - red_alive_count
        blue_destroyed_count = self.blue_fighter_num - blue_alive_count

        observations: Dict[str, Dict[str, np.ndarray]] = {}
        infos: Dict[str, Dict[str, object]] = {}
        alive_mask = np.zeros(self.red_fighter_num, dtype=np.bool_)
        for idx, agent_id in enumerate(self.agents):
            fighter_obs = red_obs["fighter"][idx]
            alive = bool(fighter_obs["alive"])
            alive_mask[idx] = alive

            if self.config.semantic_screen_observation:
                screen = self._build_semantic_screen(idx, raw_fighter_obs, red_raw_obs["joint_obs_dict"])
            else:
                screen = np.asarray(fighter_obs["screen"], dtype=np.uint8)
            info_vec = np.asarray(fighter_obs["info"], dtype=np.float32)
            visible_enemy_count = 0
            nearest_enemy_distance = 0.0
            valid_attack_count = 0
            nearest_enemy_bearing_sin = 0.0
            nearest_enemy_bearing_cos = 0.0
            recv_count = 0
            recv_direction_sin = 0.0
            recv_direction_cos = 0.0
            recv_dominant_freq = 0.0
            if not alive:
                screen = np.zeros_like(screen, dtype=np.uint8)
                info_vec = np.zeros_like(info_vec, dtype=np.float32)
                mask = np.zeros((ACTION_NUM,), dtype=np.bool_)
                mask[0] = True
            else:
                mask = valid_masks[idx]
                visible_list = raw_fighter_obs[idx]["r_visible_list"]
                visible_enemy_count = len(visible_list)
                nearest_enemy_distance, nearest_enemy_bearing_sin, nearest_enemy_bearing_cos = (
                    self._extract_nearest_visible_track(raw_fighter_obs[idx], visible_list)
                )
                valid_attack_count = max(0, len(get_valid_attack_indices(info_vec)) - 1)
                recv_count, recv_direction_sin, recv_direction_cos, recv_dominant_freq = self._extract_recv_features(
                    raw_fighter_obs[idx]["j_recv_list"]
                )
            has_attack_opportunity = bool(valid_attack_count > 0)

            observations[agent_id] = {
                "screen": screen,
                "info": info_vec,
                "action_mask": mask.astype(np.bool_),
                "alive": np.asarray(alive, dtype=np.bool_),
                "visible_enemy_count": np.asarray([visible_enemy_count], dtype=np.float32),
                "nearest_enemy_distance": np.asarray([nearest_enemy_distance], dtype=np.float32),
                "has_attack_opportunity": np.asarray([float(has_attack_opportunity)], dtype=np.float32),
                "nearest_enemy_bearing_sin": np.asarray([nearest_enemy_bearing_sin], dtype=np.float32),
                "nearest_enemy_bearing_cos": np.asarray([nearest_enemy_bearing_cos], dtype=np.float32),
                "recv_count": np.asarray([recv_count], dtype=np.float32),
                "recv_direction_sin": np.asarray([recv_direction_sin], dtype=np.float32),
                "recv_direction_cos": np.asarray([recv_direction_cos], dtype=np.float32),
                "recv_dominant_freq": np.asarray([recv_dominant_freq], dtype=np.float32),
                "red_fighter_alive_count": np.asarray([red_alive_count], dtype=np.float32),
                "red_fighter_destroyed_count": np.asarray([red_destroyed_count], dtype=np.float32),
                "blue_fighter_alive_count": np.asarray([blue_alive_count], dtype=np.float32),
                "blue_fighter_destroyed_count": np.asarray([blue_destroyed_count], dtype=np.float32),
            }
            infos[agent_id] = {
                "is_active": alive,
                "valid_action_mask": mask.astype(np.bool_),
                "step_count": self._step_count,
                "visible_enemy_count": visible_enemy_count,
                "nearest_enemy_distance": nearest_enemy_distance,
                "has_attack_opportunity": has_attack_opportunity,
                "recv_count": recv_count,
                "recv_dominant_freq": recv_dominant_freq,
                "red_fighter_alive_count": red_alive_count,
                "red_fighter_destroyed_count": red_destroyed_count,
                "blue_fighter_alive_count": blue_alive_count,
                "blue_fighter_destroyed_count": blue_destroyed_count,
            }
            if global_state is not None:
                infos[agent_id]["global_state"] = global_state

        self._last_alive_mask = alive_mask
        return observations, infos

    def _build_semantic_screen(
        self,
        fighter_idx: int,
        raw_fighter_obs: List[Mapping[str, object]],
        joint_obs_dict: Mapping[str, object],
    ) -> np.ndarray:
        screen = np.zeros((self._SCREEN_SIZE, self._SCREEN_SIZE, self._SEMANTIC_SCREEN_CHANNELS), dtype=np.uint8)

        team_visible_targets: List[Mapping[str, object]] = []
        for fighter in raw_fighter_obs:
            if fighter["alive"]:
                team_visible_targets.extend(fighter["r_visible_list"])
        passive_targets = list(joint_obs_dict.get("passive_detection_enemy_list", []))
        own_visible_targets = list(raw_fighter_obs[fighter_idx]["r_visible_list"])

        # Channel 0: current self radar contacts
        self._draw_targets(screen[:, :, 0], own_visible_targets, value=255)
        # Channel 1: current team active contacts union
        self._draw_targets(screen[:, :, 1], team_visible_targets, value=255)
        # Channel 2: current passive detections
        self._draw_targets(screen[:, :, 2], passive_targets, value=255)

        # Channel 3: decayed enemy track memory from active+passive detections
        memory = self._semantic_enemy_memory[fighter_idx]
        decay = 255.0 / float(max(self.config.screen_track_memory_steps, 1))
        np.maximum(memory - decay, 0.0, out=memory)
        self._draw_targets(memory, team_visible_targets, value=255.0)
        self._draw_targets(memory, passive_targets, value=255.0)
        screen[:, :, 3] = np.asarray(memory, dtype=np.uint8)

        # Channel 4: alive friendly teammates (excluding self)
        for idx, fighter in enumerate(raw_fighter_obs):
            if not fighter["alive"] or idx == fighter_idx:
                continue
            self._draw_point(screen[:, :, 4], fighter["pos_x"], fighter["pos_y"], value=255)

        # Channel 5: self location
        self._draw_point(
            screen[:, :, 5],
            raw_fighter_obs[fighter_idx]["pos_x"],
            raw_fighter_obs[fighter_idx]["pos_y"],
            value=255,
        )
        return screen

    def _draw_targets(self, plane: np.ndarray, targets: Iterable[Mapping[str, object]], value: float) -> None:
        for target in targets:
            self._draw_point(plane, target["pos_x"], target["pos_y"], value=value)

    def _draw_point(self, plane: np.ndarray, pos_x: float, pos_y: float, value: float) -> None:
        cell_x = int(float(pos_y) / 10.0)
        cell_y = int(float(pos_x) / 10.0)
        x0 = max(cell_x - 1, 0)
        x1 = min(cell_x + 2, self._SCREEN_SIZE)
        y0 = max(cell_y - 1, 0)
        y1 = min(cell_y + 2, self._SCREEN_SIZE)
        plane[x0:x1, y0:y1] = value

    def _decode_red_actions(self, action_dict: Mapping[str, int]) -> np.ndarray:
        fighter_action = np.zeros((self.red_fighter_num, 4), dtype=np.int32)
        delta_course_enabled = bool(self.config.delta_course_action)
        max_course_delta = max(1e-6, float(self.config.course_delta_deg))
        for idx, agent_id in enumerate(self.agents):
            fighter_raw_obs = None
            if self._last_red_raw_obs is not None:
                fighter_raw_obs = self._last_red_raw_obs["fighter_obs_list"][idx]
            radar_point, disturb_point = self._get_support_action(idx, fighter_raw_obs)
            fighter_action[idx][1] = radar_point
            fighter_action[idx][2] = disturb_point

            if not self._last_alive_mask[idx]:
                continue

            action = action_dict.get(agent_id, 0)
            if isinstance(action, np.ndarray):
                action = action.tolist()
            if isinstance(action, (list, tuple)) and len(action) >= 2:
                course_action = int(action[0])
                attack_action = int(action[1])
            else:
                flat_action = int(action)
                course_action = int(flat_action / ATTACK_IND_NUM)
                attack_action = int(flat_action % ATTACK_IND_NUM)

            if delta_course_enabled and fighter_raw_obs is not None:
                current_course = float(fighter_raw_obs["course"])
                if COURSE_NUM <= 1:
                    course_delta = 0.0
                else:
                    normalized = float(course_action) / float(COURSE_NUM - 1)
                    course_delta = (normalized * 2.0 - 1.0) * max_course_delta
                fighter_action[idx][0] = int((current_course + course_delta) % 360.0)
            else:
                fighter_action[idx][0] = int(360 / COURSE_NUM * course_action)
            fighter_action[idx][3] = attack_action
        return fighter_action

    def _get_support_action(self, fighter_idx: int, fighter_raw_obs: Optional[Mapping[str, object]]):
        if not self.config.adaptive_support_policy or fighter_raw_obs is None:
            return get_support_action(self._step_count, fighter_idx)

        recv_list = list(fighter_raw_obs.get("j_recv_list", []))
        visible_list = list(fighter_raw_obs.get("r_visible_list", []))
        current_radar_point = int(fighter_raw_obs.get("r_fre_point", 0))
        if len(visible_list) > 0 and 1 <= current_radar_point <= RADAR_POINT_NUM:
            radar_point = current_radar_point
            self._support_radar_points[fighter_idx] = radar_point
            self._support_hold_steps[fighter_idx] = max(1, int(self.config.support_search_hold))
        else:
            if self._support_hold_steps[fighter_idx] <= 0:
                self._support_radar_points[fighter_idx] = int(self._rng.randint(1, RADAR_POINT_NUM + 1))
                self._support_hold_steps[fighter_idx] = max(1, int(self.config.support_search_hold))
            radar_point = int(self._support_radar_points[fighter_idx])
            self._support_hold_steps[fighter_idx] -= 1

        disturb_point = self._get_disturb_point(recv_list)
        return radar_point, disturb_point

    @staticmethod
    def _get_dominant_recv_freq(recv_list: List[Mapping[str, object]]) -> Optional[int]:
        if len(recv_list) == 1:
            return int(recv_list[0].get("r_fp", DEFAULT_DISTURB_POINT))
        if len(recv_list) > 1:
            freq_counts = Counter(int(item.get("r_fp", DEFAULT_DISTURB_POINT)) for item in recv_list)
            dominant_freq, dominant_count = freq_counts.most_common(1)[0]
            if dominant_count * 2 > len(recv_list):
                return int(dominant_freq)
        return None

    @staticmethod
    def _get_disturb_point(recv_list: List[Mapping[str, object]]) -> int:
        dominant_freq = MaCAParallelEnv._get_dominant_recv_freq(recv_list)
        return int(dominant_freq) if dominant_freq is not None else DEFAULT_DISTURB_POINT

    @staticmethod
    def _extract_recv_features(recv_list: List[Mapping[str, object]]):
        recv_count = len(recv_list)
        if recv_count <= 0:
            return 0.0, 0.0, 0.0, 0.0

        dominant_freq = MaCAParallelEnv._get_dominant_recv_freq(recv_list)
        if dominant_freq is not None:
            selected = [item for item in recv_list if int(item.get("r_fp", DEFAULT_DISTURB_POINT)) == dominant_freq]
        else:
            selected = recv_list

        angles = np.asarray([float(item.get("direction", 0.0)) * (np.pi / 180.0) for item in selected], dtype=np.float32)
        recv_direction_sin = float(np.mean(np.sin(angles))) if angles.size > 0 else 0.0
        recv_direction_cos = float(np.mean(np.cos(angles))) if angles.size > 0 else 0.0
        recv_freq = float(dominant_freq if dominant_freq is not None else selected[0].get("r_fp", 0.0))
        return float(recv_count), recv_direction_sin, recv_direction_cos, recv_freq

    @staticmethod
    def _extract_nearest_visible_track(fighter_raw_obs: Mapping[str, object], visible_list: List[Mapping[str, object]]):
        if len(visible_list) <= 0:
            return 0.0, 0.0, 0.0

        own_x = float(fighter_raw_obs["pos_x"])
        own_y = float(fighter_raw_obs["pos_y"])
        own_course = float(fighter_raw_obs["course"])
        closest = None
        closest_dist_sq = None
        for target in visible_list:
            dx = float(target["pos_x"]) - own_x
            dy = float(target["pos_y"]) - own_y
            dist_sq = dx * dx + dy * dy
            if closest_dist_sq is None or dist_sq < closest_dist_sq:
                closest = target
                closest_dist_sq = dist_sq

        if closest is None or closest_dist_sq is None:
            return 0.0, 0.0, 0.0

        dx = float(closest["pos_x"]) - own_x
        dy = float(closest["pos_y"]) - own_y
        target_bearing = float(np.degrees(np.arctan2(dy, dx)))
        relative_bearing = ((target_bearing - own_course + 180.0) % 360.0) - 180.0
        relative_rad = relative_bearing * (np.pi / 180.0)
        return float(np.sqrt(closest_dist_sq)), float(np.sin(relative_rad)), float(np.cos(relative_rad))

    def _build_global_state(self, red_raw_obs: Mapping[str, object], blue_raw_obs: Mapping[str, object]) -> np.ndarray:
        features: List[float] = [self._step_count / float(max(self.config.max_step, 1))]
        features.extend(self._encode_raw_side(red_raw_obs["fighter_obs_list"]))
        features.extend(self._encode_raw_side(blue_raw_obs["fighter_obs_list"]))
        return np.asarray(features, dtype=np.float32)

    def _encode_raw_side(self, fighter_obs_list: Iterable[Mapping[str, object]]) -> List[float]:
        values: List[float] = []
        for fighter in fighter_obs_list:
            alive = 1.0 if fighter["alive"] else 0.0
            values.extend(
                [
                    alive,
                    float(fighter["pos_x"]) / float(self._size_x),
                    float(fighter["pos_y"]) / float(self._size_y),
                    float(fighter["course"]) / 359.0,
                    float(fighter["l_missile_left"]) / 4.0,
                    float(fighter["s_missile_left"]) / 4.0,
                    float(fighter["r_iswork"]),
                    float(fighter["j_iswork"]),
                ]
            )
        return values
