#!/usr/bin/env python
"""Train MaCA with a lightweight parameter-sharing PPO + team critic stack."""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import random
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from marl_env.mappo_env import MAPPOMaCAConfig, MAPPOMaCAEnv
from marl_env.mappo_model import MAPPOModelConfig, TeamActorCritic


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=str, default="maca_mappo_baseline")
    parser.add_argument("--train_dir", type=str, default="train_dir/mappo")
    parser.add_argument("--device", type=str, default="gpu")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--num_envs", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--rollout", type=int, default=128)
    parser.add_argument("--train_for_env_steps", type=int, default=20_000_000)
    parser.add_argument("--ppo_epochs", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae_lambda", type=float, default=0.95)
    parser.add_argument("--clip_ratio", type=float, default=0.2)
    parser.add_argument("--value_loss_coeff", type=float, default=0.5)
    parser.add_argument("--entropy_coeff", type=float, default=0.01)
    parser.add_argument("--max_grad_norm", type=float, default=0.5)
    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--role_embed_dim", type=int, default=8)
    parser.add_argument("--save_every_sec", type=int, default=900)
    parser.add_argument("--log_every_sec", type=int, default=30)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--maca_map_path", type=str, default="maps/1000_1000_fighter10v10.map")
    parser.add_argument("--maca_red_obs_ind", type=str, default="simple")
    parser.add_argument("--maca_opponent", type=str, default="fix_rule")
    parser.add_argument("--maca_max_step", type=int, default=1000)
    parser.add_argument("--maca_render", action="store_true")
    parser.add_argument("--maca_random_pos", action="store_true")
    parser.add_argument("--maca_adaptive_support_policy", action="store_true", default=True)
    parser.add_argument("--maca_support_search_hold", type=int, default=6)
    parser.add_argument("--maca_delta_course_action", action="store_true", default=True)
    parser.add_argument("--maca_course_delta_deg", type=float, default=45.0)
    parser.add_argument("--maca_max_visible_enemies", type=int, default=4)
    parser.add_argument("--maca_friendly_attrition_penalty", type=float, default=200.0)
    parser.add_argument("--maca_enemy_attrition_reward", type=float, default=100.0)
    return parser.parse_args()


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


def latest_checkpoint(args) -> Path:
    ckpt_dir = checkpoint_dir(args)
    if not ckpt_dir.exists():
        return None
    checkpoints = sorted(ckpt_dir.glob("checkpoint_*.pt"))
    return checkpoints[-1] if checkpoints else None


def save_config(args, local_obs_dim: int, global_state_dim: int, num_agents: int):
    exp_dir = experiment_dir(args)
    exp_dir.mkdir(parents=True, exist_ok=True)
    cfg = vars(args).copy()
    cfg["local_obs_dim"] = int(local_obs_dim)
    cfg["global_state_dim"] = int(global_state_dim)
    cfg["num_agents"] = int(num_agents)
    (exp_dir / "cfg.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=2))


def save_checkpoint(args, model, optimizer, env_steps, update_idx):
    ckpt_dir = checkpoint_dir(args)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    path = ckpt_dir / f"checkpoint_{update_idx:09d}_{env_steps}.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "env_steps": env_steps,
            "update_idx": update_idx,
        },
        path,
    )
    print(f"[checkpoint] saved {path}", flush=True)


def load_checkpoint(args, model, optimizer):
    ckpt_path = latest_checkpoint(args)
    if ckpt_path is None:
        return 0, 0
    state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    print(f"[checkpoint] resumed from {ckpt_path}", flush=True)
    return int(state.get("env_steps", 0)), int(state.get("update_idx", 0))


def masked_categorical(logits: torch.Tensor, masks: torch.Tensor) -> Categorical:
    invalid_logit = torch.finfo(logits.dtype).min
    masked_logits = logits.masked_fill(~masks, invalid_logit)
    return Categorical(logits=masked_logits)


def sample_actions(model, batch, device, deterministic: bool = False):
    local_obs = torch.as_tensor(batch["local_obs"], dtype=torch.float32, device=device)
    agent_ids = torch.as_tensor(batch["agent_ids"], dtype=torch.long, device=device)
    attack_masks = torch.as_tensor(batch["attack_masks"], dtype=torch.bool, device=device)
    alive_mask = torch.as_tensor(batch["alive_mask"], dtype=torch.float32, device=device)
    actor_h = torch.as_tensor(batch["actor_h"], dtype=torch.float32, device=device)

    flat_local = local_obs.reshape(-1, local_obs.shape[-1])
    flat_ids = agent_ids.reshape(-1)
    flat_attack_masks = attack_masks.reshape(-1, attack_masks.shape[-1])
    flat_alive = alive_mask.reshape(-1)
    flat_actor_h = actor_h.reshape(-1, actor_h.shape[-1])

    course_logits, attack_logits, next_actor_h = model.actor_step(flat_local, flat_ids, flat_actor_h)
    course_dist = Categorical(logits=course_logits)
    attack_dist = masked_categorical(attack_logits, flat_attack_masks)

    if deterministic:
        course_action = torch.argmax(course_logits, dim=-1)
        attack_action = torch.argmax(attack_dist.logits, dim=-1)
    else:
        course_action = course_dist.sample()
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
        "alive_mask": flat_alive.reshape(local_obs.shape[0], local_obs.shape[1]),
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
                        if done:
                            episodes.append(info["episode_extra_stats"])
                            next_obs = env.reset()
                            actor_h_batch[env_idx, :, :] = 0.0
                        next_obs_batch.append(next_obs)

                    buffer["reward"][step_idx] = rewards
                    buffer["done"][step_idx] = dones
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

            raise ValueError(f"Unknown collector command: {command}")
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
        summary_key = key if key.endswith("_mean") else f"{key}_mean"
        summary[summary_key] = float(np.mean(values))
    if "win_flag_mean" in summary:
        summary["win_rate"] = summary["win_flag_mean"]
    return summary


def rollout(collectors: CollectorPool, actor_state: dict, model, device, args, episode_stats):
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
        buffer["value"] = (
            model.value(global_state_t.reshape(-1, global_state_t.shape[-1]))
            .reshape(args.rollout, collectors.num_envs)
            .cpu()
            .numpy()
        )
        next_value = model.value(torch.as_tensor(final_global_state, dtype=torch.float32, device=device)).cpu().numpy()

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


def ppo_update(model, optimizer, buffer, advantages, returns, device, args):
    local_obs = torch.as_tensor(buffer["local_obs"], dtype=torch.float32, device=device)
    global_state = torch.as_tensor(buffer["global_state"], dtype=torch.float32, device=device)
    agent_ids = torch.as_tensor(buffer["agent_ids"], dtype=torch.long, device=device)
    attack_masks = torch.as_tensor(buffer["attack_masks"], dtype=torch.bool, device=device)
    alive_mask = torch.as_tensor(buffer["alive_mask"], dtype=torch.float32, device=device)
    actor_h = torch.as_tensor(buffer["actor_h"], dtype=torch.float32, device=device)
    old_log_prob = torch.as_tensor(buffer["log_prob"], dtype=torch.float32, device=device)
    course_action = torch.as_tensor(buffer["course_action"], dtype=torch.long, device=device)
    attack_action = torch.as_tensor(buffer["attack_action"], dtype=torch.long, device=device)
    advantages_t = torch.as_tensor(advantages, dtype=torch.float32, device=device)
    returns_t = torch.as_tensor(returns, dtype=torch.float32, device=device)

    adv_mean = advantages_t.mean()
    adv_std = advantages_t.std(unbiased=False)
    advantages_t = (advantages_t - adv_mean) / torch.clamp(adv_std, min=1e-6)

    stats = {}
    for _ in range(args.ppo_epochs):
        flat_local = local_obs.reshape(-1, local_obs.shape[-1])
        flat_ids = agent_ids.reshape(-1)
        flat_attack_masks = attack_masks.reshape(-1, attack_masks.shape[-1])
        flat_alive = alive_mask.reshape(-1) > 0.5
        flat_actor_h = actor_h.reshape(-1, actor_h.shape[-1])
        flat_course_action = course_action.reshape(-1)
        flat_attack_action = attack_action.reshape(-1)
        flat_old_log_prob = old_log_prob.reshape(-1)
        flat_advantages = advantages_t.unsqueeze(-1).repeat(1, 1, local_obs.shape[2]).reshape(-1)

        course_logits, attack_logits, _ = model.actor_step(flat_local, flat_ids, flat_actor_h)
        course_dist = Categorical(logits=course_logits)
        attack_dist = masked_categorical(attack_logits, flat_attack_masks)
        new_log_prob = course_dist.log_prob(flat_course_action) + attack_dist.log_prob(flat_attack_action)
        entropy = course_dist.entropy() + attack_dist.entropy()

        ratio = torch.exp(new_log_prob - flat_old_log_prob)
        clipped_ratio = torch.clamp(ratio, 1.0 - args.clip_ratio, 1.0 + args.clip_ratio)
        policy_loss_terms = torch.min(ratio * flat_advantages, clipped_ratio * flat_advantages)
        active_policy_terms = torch.masked_select(policy_loss_terms, flat_alive)
        active_entropy_terms = torch.masked_select(entropy, flat_alive)
        if active_policy_terms.numel() > 0:
            policy_loss = -active_policy_terms.mean()
            entropy_loss = -args.entropy_coeff * active_entropy_terms.mean()
            entropy_mean = float(active_entropy_terms.mean().item())
        else:
            policy_loss = new_log_prob.sum() * 0.0
            entropy_loss = entropy.sum() * 0.0
            entropy_mean = 0.0

        new_values = model.value(global_state.reshape(-1, global_state.shape[-1]))
        value_loss = 0.5 * (new_values - returns_t.reshape(-1)).pow(2).mean() * args.value_loss_coeff

        total_loss = policy_loss + value_loss + entropy_loss
        optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        optimizer.step()

        stats = {
            "policy_loss": float(policy_loss.item()),
            "value_loss": float(value_loss.item()),
            "entropy": entropy_mean,
            "adv_mean": float(advantages_t.mean().item()),
        }
    return stats


def main():
    args = parse_args()
    device = device_from_arg(args.device)
    set_seed(args.seed)

    collectors = CollectorPool(args)
    initial_obs = collectors.reset(seed=args.seed)
    local_obs_dim = initial_obs[0]["local_obs"].shape[1]
    global_state_dim = initial_obs[0]["global_state"].shape[0]
    num_agents = initial_obs[0]["local_obs"].shape[0]
    save_config(args, local_obs_dim, global_state_dim, num_agents)
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
    )
    model = TeamActorCritic(model_cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    env_steps = 0
    update_idx = 0
    if args.resume:
        env_steps, update_idx = load_checkpoint(args, model, optimizer)

    last_log_time = time.time()
    last_save_time = time.time()
    episode_stats = deque(maxlen=100)

    try:
        while env_steps < args.train_for_env_steps:
            update_start = time.time()
            actor_state = export_actor_state_cpu(model)
            buffer, next_value = rollout(collectors, actor_state, model, device, args, episode_stats)
            advantages, returns = compute_gae(buffer, next_value, args.gamma, args.gae_lambda)
            stats = ppo_update(model, optimizer, buffer, advantages, returns, device, args)

            env_steps += args.rollout * collectors.num_envs
            update_idx += 1
            sample_fps = (args.rollout * collectors.num_envs) / max(time.time() - update_start, 1e-6)

            if time.time() - last_log_time >= args.log_every_sec:
                summary = summarize_episode_stats(list(episode_stats))
                reward_mean = summary.get("round_reward_mean", 0.0)
                win_rate = summary.get("win_rate", 0.0)
                print(
                    "[train] env_steps=%d update=%d reward_mean=%.2f win_rate=%.3f fps=%.1f policy_loss=%.4f value_loss=%.4f entropy=%.4f"
                    % (
                        env_steps,
                        update_idx,
                        reward_mean,
                        win_rate,
                        sample_fps,
                        stats.get("policy_loss", 0.0),
                        stats.get("value_loss", 0.0),
                        stats.get("entropy", 0.0),
                    ),
                    flush=True,
                )
                last_log_time = time.time()

            if time.time() - last_save_time >= args.save_every_sec:
                save_checkpoint(args, model, optimizer, env_steps, update_idx)
                last_save_time = time.time()

        save_checkpoint(args, model, optimizer, env_steps, update_idx)
    finally:
        collectors.close()


if __name__ == "__main__":
    main()
