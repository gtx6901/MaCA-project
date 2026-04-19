from __future__ import annotations

import time

import numpy as np
import torch

from marl_train.collector import ROLLOUT_BUFFER_DTYPES


def rollout(collectors, model, device, args, episode_stats, value_normalizer=None):
    t_collect_start = time.time()
    worker_results = collectors.collect(args.rollout)
    rollout_collect_time = time.time() - t_collect_start

    buffer_keys = [
        "local_obs",
        "global_state",
        "agent_ids",
        "attack_masks",
        "alive_mask",
        "actor_h",
        "course_action",
        "attack_action",
        "policy_attack_action",
        "mode_action",
        "log_prob",
        "mode_log_prob",
        "mode_decision",
        "mode_duration",
        "reward",
        "reward_env",
        "reward_mode",
        "reward_exec",
        "damage_reward",
        "kill_reward",
        "survival_reward",
        "win_indicator",
        "done",
        "agent_aux_reward",
    ]
    t_post_start = time.time()
    first_worker_buffer = worker_results[0]["buffer"]
    buffer = {}
    for key in buffer_keys:
        worker_shape = first_worker_buffer[key].shape
        target_shape = (worker_shape[0], collectors.num_envs) + worker_shape[2:]
        buffer[key] = np.empty(target_shape, dtype=np.dtype(ROLLOUT_BUFFER_DTYPES[key]))

    final_global_dim = int(worker_results[0]["final_global_state"].shape[-1])
    final_global_state = np.empty((collectors.num_envs, final_global_dim), dtype=np.float32)

    for worker_idx, result in enumerate(worker_results):
        env_begin, env_end = collectors.worker_env_slices[worker_idx]
        for key in buffer_keys:
            buffer[key][:, env_begin:env_end] = result["buffer"][key]
        final_global_state[env_begin:env_end] = result["final_global_state"]

    for result in worker_results:
        for episode in result["episodes"]:
            episode_stats.append(episode)

    rollout_postprocess_time = time.time() - t_post_start

    hidden_mismatch_count = int(sum(int(result.get("rnn_hidden_mismatch_count", 0)) for result in worker_results))
    hidden_max_abs_diff = float(max(float(result.get("rnn_hidden_max_abs_diff", 0.0)) for result in worker_results))

    t_value_bootstrap_start = time.time()
    global_state_t = torch.as_tensor(buffer["global_state"], dtype=torch.float32, device=device)
    mode_action_t = torch.as_tensor(buffer["mode_action"], dtype=torch.long, device=device)
    with torch.no_grad():
        value_heads = model.value_heads(global_state_t, mode_actions=mode_action_t)
        raw_values = value_heads["team"].cpu().numpy()
        final_mode_action = mode_action_t[-1]
        raw_next_value = model.value(
            torch.as_tensor(final_global_state, dtype=torch.float32, device=device),
            mode_actions=final_mode_action,
        ).cpu().numpy()
        if value_normalizer is not None:
            buffer["value"] = value_normalizer.denormalize(raw_values)
            next_value = value_normalizer.denormalize(raw_next_value)
        else:
            buffer["value"] = raw_values
            next_value = raw_next_value
    value_bootstrap_time = time.time() - t_value_bootstrap_start

    rollout_diag = {
        "rnn_hidden_mismatch_count": hidden_mismatch_count,
        "rnn_hidden_max_abs_diff": hidden_max_abs_diff,
        "damage_reward_mean": float(np.mean(buffer["damage_reward"])),
        "reward_env_mean": float(np.mean(buffer["reward_env"])),
        "reward_mode_mean": float(np.mean(buffer["reward_mode"])),
        "reward_exec_mean": float(np.mean(buffer["reward_exec"])),
        "kill_reward_mean": float(np.mean(buffer["kill_reward"])),
        "survival_reward_mean": float(np.mean(buffer["survival_reward"])),
        "win_indicator_mean": float(np.mean(buffer["win_indicator"])),
        "value_contact_mean": float(value_heads["contact"].mean().item()),
        "value_opportunity_mean": float(value_heads["opportunity"].mean().item()),
        "value_survival_mean": float(value_heads["survival"].mean().item()),
    }
    rollout_timing = {
        "rollout_collect_time": float(rollout_collect_time),
        "rollout_postprocess_time": float(rollout_postprocess_time),
        "value_bootstrap_time": float(value_bootstrap_time),
    }
    return buffer, next_value, rollout_diag, rollout_timing


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

    alive = buffer["alive_mask"]
    active_mask = alive > 0.5
    active_returns = returns[active_mask]
    if active_returns.size > 0:
        returns = returns - float(np.mean(active_returns))
    return returns


def compute_mode_advantages(buffer, team_advantages):
    mode_decision = np.asarray(buffer["mode_decision"], dtype=np.float32)
    mode_duration = np.asarray(buffer["mode_duration"], dtype=np.float32)
    alive_mask = np.asarray(buffer["alive_mask"], dtype=np.float32)

    high_adv = team_advantages[:, :, None] * np.maximum(mode_duration, 1.0)
    high_adv = high_adv * mode_decision * alive_mask

    active = high_adv[mode_decision > 0.5]
    if active.size > 0:
        mean = float(np.mean(active))
        std = float(np.std(active))
        if std > 1e-6:
            high_adv = (high_adv - mean) / std
        else:
            high_adv = high_adv - mean
        high_adv = high_adv * mode_decision * alive_mask
    return high_adv.astype(np.float32, copy=False)
