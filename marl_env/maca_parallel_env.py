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
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Tuple

import numpy as np

from fighter_action_utils import (
    ACTION_NUM,
    ATTACK_IND_NUM,
    COURSE_NUM,
    build_valid_action_masks,
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
    max_step: int = 650
    render: bool = False
    random_pos: bool = False
    random_seed: int = -1


class MaCAParallelEnv:
    """Parallel-style wrapper over the current MaCA fighter environment."""

    def __init__(self, config: Optional[EnvConfig] = None):
        self.config = config or EnvConfig()
        self._current_seed = self.config.random_seed
        self._pending_seed = self.config.random_seed
        self._env = None
        self._opponent_agent = None
        self._last_blue_obs = None
        self._last_alive_mask = None
        self._step_count = 0
        self._size_x = None
        self._size_y = None
        self.red_detector_num = 0
        self.red_fighter_num = 0
        self.blue_detector_num = 0
        self.blue_fighter_num = 0
        self.possible_agents: List[str] = []
        self.agents: List[str] = []
        self.action_space_n = ACTION_NUM
        self.observation_spec = {
            "screen_shape": (100, 100, 5),
            "info_shape": (6,),
            "action_mask_shape": (ACTION_NUM,),
        }
        self._build_env(seed=self.config.random_seed)

    def _build_env(self, seed: int = -1) -> None:
        self._current_seed = seed
        self._pending_seed = seed
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

        rewards = {}
        terminations = {}
        truncations = {}
        for idx, agent_id in enumerate(self.agents):
            rewards[agent_id] = float(red_fighter_reward[idx] + red_round_reward)
            terminations[agent_id] = bool(env_done and not timeout)
            truncations[agent_id] = bool(timeout)
            infos[agent_id]["round_reward"] = float(red_round_reward)
            infos[agent_id]["opponent_round_reward"] = float(blue_round_reward)
        return observations, rewards, terminations, truncations, infos

    def close(self) -> None:
        self._env = None
        self._last_blue_obs = None

    def _collect_step_output(
        self,
    ) -> Tuple[Dict[str, Dict[str, np.ndarray]], Dict[str, Dict[str, object]]]:
        red_obs, blue_obs = self._env.get_obs()
        red_raw_obs, blue_raw_obs = self._env.get_obs_raw()
        self._last_blue_obs = blue_obs
        global_state = self._build_global_state(red_raw_obs, blue_raw_obs)

        fighter_infos = np.stack([fighter["info"] for fighter in red_obs["fighter"]], axis=0)
        valid_masks = build_valid_action_masks(fighter_infos)

        observations: Dict[str, Dict[str, np.ndarray]] = {}
        infos: Dict[str, Dict[str, object]] = {}
        alive_mask = np.zeros(self.red_fighter_num, dtype=np.bool_)
        for idx, agent_id in enumerate(self.agents):
            fighter_obs = red_obs["fighter"][idx]
            alive = bool(fighter_obs["alive"])
            alive_mask[idx] = alive

            screen = np.asarray(fighter_obs["screen"], dtype=np.float32)
            info_vec = np.asarray(fighter_obs["info"], dtype=np.float32)
            if not alive:
                screen = np.zeros_like(screen, dtype=np.float32)
                info_vec = np.zeros_like(info_vec, dtype=np.float32)
                mask = np.zeros((ACTION_NUM,), dtype=np.bool_)
                mask[0] = True
            else:
                mask = valid_masks[idx]

            observations[agent_id] = {
                "screen": screen,
                "info": info_vec,
                "action_mask": mask.astype(np.bool_),
                "alive": np.asarray(alive, dtype=np.bool_),
            }
            infos[agent_id] = {
                "is_active": alive,
                "global_state": global_state,
                "valid_action_mask": mask.astype(np.bool_),
                "step_count": self._step_count,
            }

        self._last_alive_mask = alive_mask
        return observations, infos

    def _decode_red_actions(self, action_dict: Mapping[str, int]) -> np.ndarray:
        fighter_action = np.zeros((self.red_fighter_num, 4), dtype=np.int32)
        for idx, agent_id in enumerate(self.agents):
            radar_point, disturb_point = get_support_action(self._step_count, idx)
            fighter_action[idx][1] = radar_point
            fighter_action[idx][2] = disturb_point

            if not self._last_alive_mask[idx]:
                continue

            action = int(action_dict.get(agent_id, 0))
            fighter_action[idx][0] = int(360 / COURSE_NUM * int(action / ATTACK_IND_NUM))
            fighter_action[idx][3] = int(action % ATTACK_IND_NUM)
        return fighter_action

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
