"""Sample Factory 1.x environment adapter for MaCA."""

from __future__ import annotations

from typing import Dict, List, Optional

import gym
import numpy as np
from gym import spaces

from fighter_action_utils import ACTION_NUM, ATTACK_IND_NUM
from marl_env.maca_parallel_env import EnvConfig, MaCAParallelEnv


_MEASUREMENTS_SCALE = np.asarray([359.0, 4.0, 4.0, 1500.0, 10.0, 180.0], dtype=np.float32)
_MEASUREMENTS_BIAS = np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)


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

        screen_shape = self.maca_env.observation_spec["screen_shape"]
        info_shape = self.maca_env.observation_spec["info_shape"]
        obs_shape = (screen_shape[2], screen_shape[0], screen_shape[1])

        self.action_space = spaces.Discrete(self.maca_env.action_space_n)
        self.observation_space = spaces.Dict(
            {
                "obs": spaces.Box(low=0, high=255, shape=obs_shape, dtype=np.uint8),
                "measurements": spaces.Box(low=-10.0, high=10.0, shape=info_shape, dtype=np.float32),
                "action_mask": spaces.Box(low=0, high=1, shape=(ACTION_NUM,), dtype=np.uint8),
                "is_alive": spaces.Box(low=0, high=1, shape=(1,), dtype=np.uint8),
            }
        )

        self._last_obs_dict: Optional[Dict[str, Dict[str, np.ndarray]]] = None
        self._episode_returns = np.zeros(self.num_agents, dtype=np.float32)
        self._invalid_action_counts = np.zeros(self.num_agents, dtype=np.int32)
        self._episode_len = 0

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
        )

    def seed(self, seed: Optional[int] = None) -> List[int]:
        return self.maca_env.seed(seed)

    def reset(self):
        obs_dict, _ = self.maca_env.reset()
        self._last_obs_dict = obs_dict
        self._episode_returns.fill(0.0)
        self._invalid_action_counts.fill(0)
        self._episode_len = 0
        return self._format_obs_list(obs_dict)

    def step(self, actions):
        if self._last_obs_dict is None:
            raise RuntimeError("Environment must be reset before stepping")

        action_dict = {}
        for idx, agent_id in enumerate(self.maca_env.agents):
            chosen_action = int(actions[idx])
            valid_mask = self._last_obs_dict[agent_id]["action_mask"]
            safe_action = self._sanitize_action(chosen_action, valid_mask)
            if safe_action != chosen_action:
                self._invalid_action_counts[idx] += 1
            action_dict[agent_id] = safe_action

        obs_dict, reward_dict, terminations, truncations, infos = self.maca_env.step(action_dict)
        rewards = np.asarray([reward_dict[agent_id] for agent_id in self.maca_env.agents], dtype=np.float32)
        dones = [bool(terminations[agent_id] or truncations[agent_id]) for agent_id in self.maca_env.agents]

        self._episode_len += 1
        self._episode_returns += rewards

        info_list = []
        for idx, agent_id in enumerate(self.maca_env.agents):
            agent_info = dict(infos[agent_id])
            agent_info.setdefault("num_frames", 1)
            if dones[idx]:
                agent_info["true_reward"] = float(self._episode_returns[idx])
                agent_info["episode_extra_stats"] = {
                    "round_reward": float(agent_info.get("round_reward", 0.0)),
                    "opponent_round_reward": float(agent_info.get("opponent_round_reward", 0.0)),
                    "invalid_action_frac": float(self._invalid_action_counts[idx]) / float(max(self._episode_len, 1)),
                    "episode_len": float(self._episode_len),
                    "win_flag": float(agent_info.get("round_reward", 0.0) > agent_info.get("opponent_round_reward", 0.0)),
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
        return [self._format_agent_obs(obs_dict[agent_id]) for agent_id in self.maca_env.agents]

    def _format_agent_obs(self, agent_obs: Dict[str, np.ndarray]):
        screen = np.asarray(agent_obs["screen"], dtype=np.uint8)
        screen = np.transpose(screen, (2, 0, 1))
        measurements = self._normalize_measurements(agent_obs["info"])
        action_mask = np.asarray(agent_obs["action_mask"], dtype=np.uint8)
        is_alive = np.asarray([agent_obs["alive"]], dtype=np.uint8)
        return {
            "obs": screen,
            "measurements": measurements,
            "action_mask": action_mask,
            "is_alive": is_alive,
        }

    def _normalize_measurements(self, info_vec: np.ndarray) -> np.ndarray:
        info_vec = np.asarray(info_vec, dtype=np.float32)
        return (info_vec - _MEASUREMENTS_BIAS) / _MEASUREMENTS_SCALE

    @staticmethod
    def _sanitize_action(action: int, valid_mask: np.ndarray) -> int:
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
