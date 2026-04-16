#!/usr/bin/env python
"""Train MaCA with recurrent MAPPO (chunked PPO + burn-in)."""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import random
import time
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from marl_env.mappo_env import MAPPOMaCAConfig, MAPPOMaCAEnv
from marl_env.mappo_model import MAPPOModelConfig, TeamActorCritic


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
        std = np.sqrt(self.var + self.epsilon)
        return np.clip((np.asarray(values) - self.mean) / std, -self.clip, self.clip)

    def denormalize(self, values):
        std = np.sqrt(self.var + self.epsilon)
        return np.asarray(values) * std + self.mean

    def state_dict(self):
        return {"mean": self.mean, "var": self.var, "count": self.count}

    def load_state_dict(self, state):
        self.mean = float(state.get("mean", 0.0))
        self.var = float(state.get("var", 1.0))
        self.count = float(state.get("count", self.epsilon))


def str2bool(value) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=str, default="maca_mappo_recurrent")
    parser.add_argument("--train_dir", type=str, default="train_dir/mappo")
    parser.add_argument("--device", type=str, default="gpu")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--num_envs", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)

    # P0: recurrent PPO with sequence chunks + burn-in
    parser.add_argument("--rollout", type=int, default=64)
    parser.add_argument("--chunk_len", type=int, default=16)
    parser.add_argument("--burn_in", type=int, default=8)
    parser.add_argument("--train_for_env_steps", type=int, default=20_000_000)

    parser.add_argument("--ppo_epochs", type=int, default=5)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae_lambda", type=float, default=0.95)
    parser.add_argument("--clip_ratio", type=float, default=0.1)
    parser.add_argument("--value_loss_coeff", type=float, default=0.5)
    parser.add_argument("--entropy_coeff", type=float, default=0.01)
    parser.add_argument("--max_grad_norm", type=float, default=5.0)
    parser.add_argument("--num_mini_batches", type=int, default=4)

    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--role_embed_dim", type=int, default=8)
    parser.add_argument("--course_embed_dim", type=int, default=16)

    parser.add_argument("--save_every_sec", type=int, default=900)
    parser.add_argument("--log_every_sec", type=int, default=30)
    parser.add_argument("--resume", action="store_true")

    # P1: team + agent-aware credit blending
    parser.add_argument("--team_adv_weight", type=float, default=1.0)
    parser.add_argument("--aux_adv_weight", type=float, default=1.0)

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

    # P1: dynamics features + agent aux reward knobs
    parser.add_argument("--maca_track_memory_steps", type=int, default=12)
    parser.add_argument("--maca_contact_reward", type=float, default=0.1)
    parser.add_argument("--maca_progress_reward_scale", type=float, default=0.002)
    parser.add_argument("--maca_progress_reward_cap", type=float, default=20.0)
    parser.add_argument("--maca_attack_window_reward", type=float, default=0.1)
    parser.add_argument("--maca_agent_aux_reward_scale", type=float, default=0.0)

    args = parser.parse_args()
    if args.chunk_len <= 0:
        raise ValueError("chunk_len must be > 0")
    if args.burn_in < 0:
        raise ValueError("burn_in must be >= 0")
    if args.rollout <= 0:
        raise ValueError("rollout must be > 0")
    return args


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def device_from_arg(device_arg: str):
    if device_arg == "gpu" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def experiment_dir(args) -> Path:
    return Path(args.train_dir) / args.experiment


def checkpoint_dir(args) -> Path:
    return experiment_dir(args) / "checkpoint"


def latest_checkpoint(args) -> Optional[Path]:
    ckpt_dir = checkpoint_dir(args)
    if not ckpt_dir.exists():
        return None
    checkpoints = sorted(ckpt_dir.glob("checkpoint_*.pt"))
    return checkpoints[-1] if checkpoints else None


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


def load_checkpoint(args, model, optimizer, value_normalizer=None):
    ckpt_path = latest_checkpoint(args)
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
    if value_normalizer is not None and "value_normalizer" in state:
        value_normalizer.load_state_dict(state["value_normalizer"])
        print("[checkpoint] value normalizer restored", flush=True)
    print("[checkpoint] resumed from %s" % ckpt_path, flush=True)
    return int(state.get("env_steps", 0)), int(state.get("update_idx", 0))

def masked_categorical(logits: torch.Tensor, masks: torch.Tensor) -> Categorical:
    invalid_logit = torch.finfo(logits.dtype).min
    masked_logits = logits.masked_fill(~masks, invalid_logit)
    return Categorical(logits=masked_logits)


def sample_actions(model, batch, device, deterministic: bool = False):
    local_obs = torch.as_tensor(batch["local_obs"], dtype=torch.float32, device=device)
    agent_ids = torch.as_tensor(batch["agent_ids"], dtype=torch.long, device=device)
    attack_masks = torch.as_tensor(batch["attack_masks"], dtype=torch.bool, device=device)
    actor_h = torch.as_tensor(batch["actor_h"], dtype=torch.float32, device=device)

    flat_local = local_obs.reshape(-1, local_obs.shape[-1])
    flat_ids = agent_ids.reshape(-1)
    flat_attack_masks = attack_masks.reshape(-1, attack_masks.shape[-1])
    flat_actor_h = actor_h.reshape(-1, actor_h.shape[-1])

    # Sequential two-head sampling: sample course first, then condition attack.
    course_logits, _attack_logits_unused, next_actor_h = model.actor_step(flat_local, flat_ids, flat_actor_h)
    course_dist = Categorical(logits=course_logits)
    if deterministic:
        course_action = torch.argmax(course_logits, dim=-1)
    else:
        course_action = course_dist.sample()

    attack_logits = model.attack_logits(next_actor_h, course_action)
    attack_dist = masked_categorical(attack_logits, flat_attack_masks)
    if deterministic:
        attack_action = torch.argmax(attack_dist.logits, dim=-1)
    else:
        attack_action = attack_dist.sample()

    log_prob = course_dist.log_prob(course_action) + attack_dist.log_prob(attack_action)
    entropy = course_dist.entropy() + attack_dist.entropy()

    course_action = course_action.reshape(local_obs.shape[0], local_obs.shape[1])
    attack_action = attack_action.reshape(local_obs.shape[0], local_obs.shape[1])
    log_prob = log_prob.reshape(local_obs.shape[0], local_obs.shape[1])
    entropy = entropy.reshape(local_obs.shape[0], local_obs.shape[1])
    next_actor_h = next_actor_h.reshape(local_obs.shape[0], local_obs.shape[1], -1)

    return {
        "course_action": course_action,
        "attack_action": attack_action,
        "log_prob": log_prob,
        "entropy": entropy,
        "next_actor_h": next_actor_h,
    }


def export_actor_state_cpu(model) -> dict:
    actor_state = {}
    for key, value in model.state_dict().items():
        if key.startswith("critic."):
            continue
        actor_state[key] = value.detach().cpu().numpy()
    return actor_state


def build_env(args, seed_offset: int) -> MAPPOMaCAEnv:
    return MAPPOMaCAEnv(
        MAPPOMaCAConfig(
            map_path=args.maca_map_path,
            red_obs_ind=args.maca_red_obs_ind,
            opponent=args.maca_opponent,
            max_step=args.maca_max_step,
            render=args.maca_render,
            random_pos=args.maca_random_pos,
            random_seed=args.seed + seed_offset,
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


def _collector_process_main(args, worker_idx: int, env_count: int, conn):
    torch.set_num_threads(1)
    set_seed(args.seed + worker_idx * 1009)
    envs = [build_env(args, seed_offset=worker_idx * 100000 + env_idx * 9973) for env_idx in range(env_count)]
    worker_model = None
    obs_batch = None

    try:
        while True:
            command, payload = conn.recv()
            if command == "reset":
                base_seed = payload
                obs_batch = []
                for env_idx, env in enumerate(envs):
                    seed = None if base_seed is None else int(base_seed + worker_idx * 1000 + env_idx)
                    obs_batch.append(env.reset(seed=seed))
                conn.send(obs_batch)
                continue

            if command == "collect":
                actor_state, rollout_steps = payload
                if obs_batch is None:
                    obs_batch = [env.reset() for env in envs]

                if worker_model is None:
                    local_obs_dim = obs_batch[0]["local_obs"].shape[1]
                    global_state_dim = obs_batch[0]["global_state"].shape[0]
                    num_agents = obs_batch[0]["local_obs"].shape[0]
                    worker_model = TeamActorCritic(
                        MAPPOModelConfig(
                            local_obs_dim=local_obs_dim,
                            global_state_dim=global_state_dim,
                            num_agents=num_agents,
                            hidden_size=args.hidden_size,
                            role_embed_dim=args.role_embed_dim,
                            course_embed_dim=args.course_embed_dim,
                        )
                    )
                    worker_model.eval()

                actor_state_tensors = {key: torch.from_numpy(value) for key, value in actor_state.items()}
                worker_model.load_state_dict(actor_state_tensors, strict=False)

                num_agents = obs_batch[0]["local_obs"].shape[0]
                local_obs_dim = obs_batch[0]["local_obs"].shape[1]
                global_dim = obs_batch[0]["global_state"].shape[0]
                attack_dim = obs_batch[0]["attack_masks"].shape[1]
                actor_hidden_dim = worker_model.actor_hidden_dim

                actor_h_batch = np.zeros((env_count, num_agents, actor_hidden_dim), dtype=np.float32)
                buffer = {
                    "local_obs": np.zeros((rollout_steps, env_count, num_agents, local_obs_dim), dtype=np.float32),
                    "global_state": np.zeros((rollout_steps, env_count, global_dim), dtype=np.float32),
                    "agent_ids": np.zeros((rollout_steps, env_count, num_agents), dtype=np.int64),
                    "attack_masks": np.zeros((rollout_steps, env_count, num_agents, attack_dim), dtype=np.bool_),
                    "alive_mask": np.zeros((rollout_steps, env_count, num_agents), dtype=np.float32),
                    "actor_h": np.zeros((rollout_steps, env_count, num_agents, actor_hidden_dim), dtype=np.float32),
                    "course_action": np.zeros((rollout_steps, env_count, num_agents), dtype=np.int64),
                    "attack_action": np.zeros((rollout_steps, env_count, num_agents), dtype=np.int64),
                    "log_prob": np.zeros((rollout_steps, env_count, num_agents), dtype=np.float32),
                    "reward": np.zeros((rollout_steps, env_count), dtype=np.float32),
                    "done": np.zeros((rollout_steps, env_count), dtype=np.float32),
                    "agent_aux_reward": np.zeros((rollout_steps, env_count, num_agents), dtype=np.float32),
                }
                episodes = []

                for step_idx in range(rollout_steps):
                    stacked = {
                        "local_obs": np.stack([obs["local_obs"] for obs in obs_batch], axis=0),
                        "global_state": np.stack([obs["global_state"] for obs in obs_batch], axis=0),
                        "attack_masks": np.stack([obs["attack_masks"] for obs in obs_batch], axis=0),
                        "alive_mask": np.stack([obs["alive_mask"] for obs in obs_batch], axis=0),
                        "agent_ids": np.stack([obs["agent_ids"] for obs in obs_batch], axis=0),
                        "actor_h": actor_h_batch.copy(),
                    }

                    with torch.no_grad():
                        actions = sample_actions(worker_model, stacked, torch.device("cpu"), deterministic=False)

                    buffer["local_obs"][step_idx] = stacked["local_obs"]
                    buffer["global_state"][step_idx] = stacked["global_state"]
                    buffer["attack_masks"][step_idx] = stacked["attack_masks"]
                    buffer["alive_mask"][step_idx] = stacked["alive_mask"]
                    buffer["agent_ids"][step_idx] = stacked["agent_ids"]
                    buffer["actor_h"][step_idx] = stacked["actor_h"]
                    buffer["course_action"][step_idx] = actions["course_action"].cpu().numpy()
                    buffer["attack_action"][step_idx] = actions["attack_action"].cpu().numpy()
                    buffer["log_prob"][step_idx] = actions["log_prob"].cpu().numpy()

                    next_obs_batch = []
                    rewards = np.zeros((env_count,), dtype=np.float32)
                    dones = np.zeros((env_count,), dtype=np.float32)
                    aux_rewards = np.zeros((env_count, num_agents), dtype=np.float32)

                    for env_idx, env in enumerate(envs):
                        env_actions = np.stack(
                            [
                                actions["course_action"][env_idx].cpu().numpy(),
                                actions["attack_action"][env_idx].cpu().numpy(),
                            ],
                            axis=-1,
                        )
                        next_obs, reward, done, info = env.step(env_actions)
                        rewards[env_idx] = float(reward)
                        dones[env_idx] = 1.0 if done else 0.0

                        info_aux = info.get("agent_aux_reward", None)
                        if info_aux is not None:
                            info_aux = np.asarray(info_aux, dtype=np.float32)
                            if info_aux.shape == (num_agents,):
                                aux_rewards[env_idx] = info_aux

                        if done:
                            episodes.append(info["episode_extra_stats"])
                            next_obs = env.reset()
                        next_obs_batch.append(next_obs)

                    buffer["reward"][step_idx] = rewards
                    buffer["done"][step_idx] = dones
                    buffer["agent_aux_reward"][step_idx] = aux_rewards

                    actor_h_batch = actions["next_actor_h"].cpu().numpy()
                    if np.any(dones > 0.5):
                        done_rows = np.where(dones > 0.5)[0]
                        actor_h_batch[done_rows, :, :] = 0.0
                    obs_batch = next_obs_batch

                final_global_state = np.stack([obs["global_state"] for obs in obs_batch], axis=0)

                conn.send(
                    {
                        "buffer": buffer,
                        "final_global_state": final_global_state,
                        "episodes": episodes,
                    }
                )
                continue

            if command == "close":
                break

            raise ValueError("Unknown collector command: %s" % command)
    finally:
        for env in envs:
            env.close()
        conn.close()


class CollectorPool:
    def __init__(self, args):
        total_envs = max(1, int(args.num_envs))
        requested_workers = max(1, int(args.num_workers))
        actual_workers = min(requested_workers, total_envs)

        base_envs = total_envs // actual_workers
        remainder = total_envs % actual_workers
        self.worker_env_counts = [
            base_envs + (1 if worker_idx < remainder else 0) for worker_idx in range(actual_workers)
        ]

        ctx = mp.get_context("spawn")
        self.parent_conns = []
        self.processes = []
        self.num_workers = actual_workers
        self.num_envs = total_envs

        for worker_idx, env_count in enumerate(self.worker_env_counts):
            parent_conn, child_conn = ctx.Pipe()
            process = ctx.Process(
                target=_collector_process_main,
                args=(args, worker_idx, env_count, child_conn),
                daemon=True,
            )
            process.start()
            child_conn.close()
            self.parent_conns.append(parent_conn)
            self.processes.append(process)

    def reset(self, seed=None):
        for worker_idx, conn in enumerate(self.parent_conns):
            worker_seed = None if seed is None else int(seed + worker_idx * 1000)
            conn.send(("reset", worker_seed))

        obs_batch = []
        for conn in self.parent_conns:
            obs_batch.extend(conn.recv())
        return obs_batch

    def collect(self, actor_state: dict, rollout_steps: int):
        for conn in self.parent_conns:
            conn.send(("collect", (actor_state, rollout_steps)))

        results = []
        for conn in self.parent_conns:
            results.append(conn.recv())
        return results

    def close(self):
        for conn in self.parent_conns:
            try:
                conn.send(("close", None))
            except (BrokenPipeError, EOFError):
                pass
        for process in self.processes:
            process.join(timeout=5)
            if process.is_alive():
                process.terminate()


def summarize_episode_stats(episodes):
    if not episodes:
        return {}
    keys = sorted(episodes[0].keys())
    summary = {"episodes": len(episodes)}
    for key in keys:
        values = [float(ep[key]) for ep in episodes if key in ep]
        if not values:
            continue
        summary_key = key if key.endswith("_mean") else "%s_mean" % key
        summary[summary_key] = float(np.mean(values))
    if "win_flag_mean" in summary:
        summary["win_rate"] = summary["win_flag_mean"]
    return summary


def rollout(collectors: CollectorPool, actor_state: dict, model, device, args, episode_stats, value_normalizer=None):
    worker_results = collectors.collect(actor_state, args.rollout)

    buffer_keys = [
        "local_obs",
        "global_state",
        "agent_ids",
        "attack_masks",
        "alive_mask",
        "actor_h",
        "course_action",
        "attack_action",
        "log_prob",
        "reward",
        "done",
        "agent_aux_reward",
    ]
    buffer = {
        key: np.concatenate([result["buffer"][key] for result in worker_results], axis=1) for key in buffer_keys
    }
    final_global_state = np.concatenate([result["final_global_state"] for result in worker_results], axis=0)

    for result in worker_results:
        for episode in result["episodes"]:
            episode_stats.append(episode)

    global_state_t = torch.as_tensor(buffer["global_state"], dtype=torch.float32, device=device)
    with torch.no_grad():
        raw_values = (
            model.value(global_state_t.reshape(-1, global_state_t.shape[-1]))
            .reshape(args.rollout, collectors.num_envs)
            .cpu()
            .numpy()
        )
        raw_next_value = model.value(
            torch.as_tensor(final_global_state, dtype=torch.float32, device=device)
        ).cpu().numpy()
        if value_normalizer is not None:
            buffer["value"] = value_normalizer.denormalize(raw_values)
            next_value = value_normalizer.denormalize(raw_next_value)
        else:
            buffer["value"] = raw_values
            next_value = raw_next_value

    return buffer, next_value


def compute_gae(buffer, next_value, gamma, gae_lambda):
    rewards = buffer["reward"]
    dones = buffer["done"]
    values = buffer["value"]
    advantages = np.zeros_like(rewards, dtype=np.float32)
    last_gae = np.zeros_like(next_value, dtype=np.float32)

    for step in reversed(range(rewards.shape[0])):
        next_nonterminal = 1.0 - dones[step]
        next_values = next_value if step == rewards.shape[0] - 1 else values[step + 1]
        delta = rewards[step] + gamma * next_values * next_nonterminal - values[step]
        last_gae = delta + gamma * gae_lambda * next_nonterminal * last_gae
        advantages[step] = last_gae

    returns = advantages + values
    return advantages, returns


def compute_aux_advantages(buffer, gamma):
    aux_rewards = buffer["agent_aux_reward"]
    dones = buffer["done"]

    returns = np.zeros_like(aux_rewards, dtype=np.float32)
    running = np.zeros_like(aux_rewards[0], dtype=np.float32)
    for step in reversed(range(aux_rewards.shape[0])):
        nonterminal = (1.0 - dones[step])[:, None]
        running = aux_rewards[step] + gamma * nonterminal * running
        returns[step] = running

    # Center using alive-agent mean to form proper pseudo-advantages
    alive = buffer["alive_mask"]
    active_mask = alive > 0.5
    active_returns = returns[active_mask]
    if active_returns.size > 0:
        returns = returns - float(np.mean(active_returns))
    return returns


def build_recurrent_chunks(rollout_steps: int, num_envs: int, num_agents: int, chunk_len: int):
    chunks = []
    for env_idx in range(num_envs):
        for agent_idx in range(num_agents):
            start = 0
            while start < rollout_steps:
                end = min(rollout_steps, start + chunk_len)
                chunks.append((env_idx, agent_idx, start, end))
                start = end
    return chunks


def ppo_update(model, optimizer, buffer, advantages_team, advantages_aux, returns, device, args, value_normalizer=None):
    local_obs = torch.as_tensor(buffer["local_obs"], dtype=torch.float32, device=device)
    global_state = torch.as_tensor(buffer["global_state"], dtype=torch.float32, device=device)
    agent_ids = torch.as_tensor(buffer["agent_ids"], dtype=torch.long, device=device)
    attack_masks = torch.as_tensor(buffer["attack_masks"], dtype=torch.bool, device=device)
    alive_mask = torch.as_tensor(buffer["alive_mask"], dtype=torch.float32, device=device)
    actor_h = torch.as_tensor(buffer["actor_h"], dtype=torch.float32, device=device)
    old_log_prob = torch.as_tensor(buffer["log_prob"], dtype=torch.float32, device=device)
    course_action = torch.as_tensor(buffer["course_action"], dtype=torch.long, device=device)
    attack_action = torch.as_tensor(buffer["attack_action"], dtype=torch.long, device=device)

    adv_team_t = torch.as_tensor(advantages_team, dtype=torch.float32, device=device)
    adv_aux_t = torch.as_tensor(advantages_aux, dtype=torch.float32, device=device)

    # Value normalization: normalize returns for critic training
    if value_normalizer is not None:
        norm_returns = value_normalizer.normalize(returns)
        returns_t = torch.as_tensor(norm_returns, dtype=torch.float32, device=device)
    else:
        returns_t = torch.as_tensor(returns, dtype=torch.float32, device=device)

    combined_adv = args.team_adv_weight * adv_team_t.unsqueeze(-1) + args.aux_adv_weight * adv_aux_t
    active_mask_bool = alive_mask > 0.5
    active_adv = combined_adv[active_mask_bool]
    if active_adv.numel() > 0:
        adv_mean = active_adv.mean()
        adv_std = active_adv.std(unbiased=False)
        combined_adv = (combined_adv - adv_mean) / torch.clamp(adv_std, min=1e-6)

    rollout_steps, num_envs, num_agents = local_obs.shape[0], local_obs.shape[1], local_obs.shape[2]
    chunks = build_recurrent_chunks(rollout_steps, num_envs, num_agents, args.chunk_len)
    num_chunks = len(chunks)
    accum_size = max(1, num_chunks // max(1, args.num_mini_batches))

    total_policy = 0.0
    total_value = 0.0
    total_entropy = 0.0
    total_chunk_count = 0
    total_active = 0
    total_grad_steps = 0

    for _ in range(args.ppo_epochs):
        order = np.random.permutation(num_chunks)
        optimizer.zero_grad()
        chunks_in_batch = 0

        for i in range(num_chunks):
            chunk_idx = int(order[i])
            env_idx, agent_idx, start, end = chunks[chunk_idx]
            burn_start = max(0, start - args.burn_in)

            h = actor_h[burn_start, env_idx, agent_idx].unsqueeze(0).detach()

            new_log_probs = []
            old_log_probs = []
            entropies = []
            advantages = []
            value_preds = []
            value_targets = []

            for t in range(burn_start, end):
                obs_t = local_obs[t, env_idx, agent_idx].unsqueeze(0)
                id_t = agent_ids[t, env_idx, agent_idx].unsqueeze(0)
                course_t = course_action[t, env_idx, agent_idx].unsqueeze(0)
                attack_t = attack_action[t, env_idx, agent_idx].unsqueeze(0)

                course_logits, attack_logits, h = model.actor_step(obs_t, id_t, h, course_actions=course_t)

                if t < start:
                    continue

                mask_t = attack_masks[t, env_idx, agent_idx].unsqueeze(0)
                alive_t = bool(alive_mask[t, env_idx, agent_idx].item() > 0.5)

                course_dist = Categorical(logits=course_logits)
                attack_dist = masked_categorical(attack_logits, mask_t)
                lp = course_dist.log_prob(course_t) + attack_dist.log_prob(attack_t)
                ent = course_dist.entropy() + attack_dist.entropy()

                value_pred = model.value(global_state[t, env_idx].unsqueeze(0)).squeeze(0)
                value_target = returns_t[t, env_idx]
                value_preds.append(value_pred)
                value_targets.append(value_target)

                if alive_t:
                    new_log_probs.append(lp.squeeze(0))
                    old_log_probs.append(old_log_prob[t, env_idx, agent_idx])
                    entropies.append(ent.squeeze(0))
                    advantages.append(combined_adv[t, env_idx, agent_idx])

            if not value_preds:
                continue

            value_pred_t = torch.stack(value_preds)
            value_target_t = torch.stack(value_targets)
            # FIX: removed erroneous /num_agents scaling
            value_loss = 0.5 * (value_pred_t - value_target_t).pow(2).mean() * args.value_loss_coeff

            if new_log_probs:
                new_log_prob_t = torch.stack(new_log_probs)
                old_log_prob_t = torch.stack(old_log_probs)
                adv_t = torch.stack(advantages)
                entropy_t = torch.stack(entropies)

                ratio = torch.exp(new_log_prob_t - old_log_prob_t)
                clipped_ratio = torch.clamp(ratio, 1.0 - args.clip_ratio, 1.0 + args.clip_ratio)
                policy_loss = -torch.min(ratio * adv_t, clipped_ratio * adv_t).mean()
                entropy_loss = -args.entropy_coeff * entropy_t.mean()
                entropy_mean = float(entropy_t.mean().item())
                total_active += int(adv_t.numel())
            else:
                policy_loss = value_loss * 0.0
                entropy_loss = value_loss * 0.0
                entropy_mean = 0.0

            # Gradient accumulation: scale loss by mini-batch size
            chunk_loss = (policy_loss + value_loss + entropy_loss) / float(accum_size)
            chunk_loss.backward()
            chunks_in_batch += 1

            total_policy += float(policy_loss.item())
            total_value += float(value_loss.item())
            total_entropy += entropy_mean
            total_chunk_count += 1

            # Step optimizer once per accumulated mini-batch
            if chunks_in_batch >= accum_size or i == num_chunks - 1:
                nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()
                chunks_in_batch = 0
                total_grad_steps += 1

    if total_chunk_count <= 0:
        return {
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy": 0.0,
            "active_samples": 0,
            "grad_steps": 0,
        }

    return {
        "policy_loss": total_policy / float(total_chunk_count),
        "value_loss": total_value / float(total_chunk_count),
        "entropy": total_entropy / float(total_chunk_count),
        "active_samples": total_active,
        "grad_steps": total_grad_steps,
    }


def main():
    args = parse_args()
    device = device_from_arg(args.device)
    set_seed(args.seed)

    collectors = CollectorPool(args)
    initial_obs = collectors.reset(seed=args.seed)
    local_obs_dim = initial_obs[0]["local_obs"].shape[1]
    global_state_dim = initial_obs[0]["global_state"].shape[0]
    num_agents = initial_obs[0]["local_obs"].shape[0]

    save_config(args, local_obs_dim, global_state_dim, num_agents, collectors.worker_env_counts)
    print(
        "[collector] num_workers=%d num_envs=%d worker_env_counts=%s"
        % (collectors.num_workers, collectors.num_envs, collectors.worker_env_counts),
        flush=True,
    )

    model_cfg = MAPPOModelConfig(
        local_obs_dim=local_obs_dim,
        global_state_dim=global_state_dim,
        num_agents=num_agents,
        hidden_size=args.hidden_size,
        role_embed_dim=args.role_embed_dim,
        course_embed_dim=args.course_embed_dim,
    )
    model = TeamActorCritic(model_cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    value_normalizer = ValueNormalizer()

    env_steps = 0
    update_idx = 0
    if args.resume:
        env_steps, update_idx = load_checkpoint(args, model, optimizer, value_normalizer)

    last_log_time = time.time()
    last_save_time = time.time()
    episode_stats = deque(maxlen=100)

    try:
        while env_steps < args.train_for_env_steps:
            update_start = time.time()

            actor_state = export_actor_state_cpu(model)
            buffer, next_value = rollout(
                collectors, actor_state, model, device, args, episode_stats, value_normalizer
            )
            advantages_team, returns = compute_gae(buffer, next_value, args.gamma, args.gae_lambda)
            advantages_aux = compute_aux_advantages(buffer, args.gamma)

            # Update value normalizer BEFORE ppo_update (so critic trains on fresh stats)
            value_normalizer.update(returns)

            stats = ppo_update(
                model,
                optimizer,
                buffer,
                advantages_team,
                advantages_aux,
                returns,
                device,
                args,
                value_normalizer,
            )

            env_steps += args.rollout * collectors.num_envs
            update_idx += 1
            sample_fps = (args.rollout * collectors.num_envs) / max(time.time() - update_start, 1e-6)

            if time.time() - last_log_time >= args.log_every_sec:
                summary = summarize_episode_stats(list(episode_stats))
                reward_mean = summary.get("round_reward_mean", 0.0)
                win_rate = summary.get("win_rate", 0.0)
                print(
                    "[train] env_steps=%d update=%d reward_mean=%.2f win_rate=%.3f fps=%.1f policy_loss=%.4f value_loss=%.4f entropy=%.4f active=%d grad_steps=%d"
                    % (
                        env_steps,
                        update_idx,
                        reward_mean,
                        win_rate,
                        sample_fps,
                        stats.get("policy_loss", 0.0),
                        stats.get("value_loss", 0.0),
                        stats.get("entropy", 0.0),
                        int(stats.get("active_samples", 0)),
                        int(stats.get("grad_steps", 0)),
                    ),
                    flush=True,
                )
                last_log_time = time.time()

            if time.time() - last_save_time >= args.save_every_sec:
                save_checkpoint(args, model, optimizer, env_steps, update_idx, value_normalizer)
                last_save_time = time.time()

        save_checkpoint(args, model, optimizer, env_steps, update_idx, value_normalizer)
    finally:
        collectors.close()


if __name__ == "__main__":
    main()
