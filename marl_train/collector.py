from __future__ import annotations

import ctypes
import multiprocessing as mp
import random
from typing import Dict

import numpy as np
import torch
from torch.distributions import Categorical

from fighter_action_utils import FIGHTER_NUM
from marl_env.mappo_env import MAPPOMaCAConfig, MAPPOMaCAEnv
from marl_env.mappo_model import MAPPOModelConfig, TeamActorCritic


ROLLOUT_BUFFER_DTYPES = {
    "local_obs": np.float32,
    "local_screen": np.uint8,
    "global_state": np.float32,
    "attack_masks": np.uint8,
    "alive_mask": np.float32,
    "actor_h": np.float32,
    "course_action": np.int64,
    "attack_action": np.int64,
    "policy_attack_action": np.int64,
    "mode_action": np.int64,
    "log_prob": np.float32,
    "mode_log_prob": np.float32,
    "mode_decision": np.float32,
    "mode_duration": np.int64,
    "reward": np.float32,
    "reward_env": np.float32,
    "reward_mode": np.float32,
    "reward_exec": np.float32,
    "reward_terminal": np.float32,
    "contact_signal": np.float32,
    "opportunity_signal": np.float32,
    "damage_reward": np.float32,
    "kill_reward": np.float32,
    "survival_reward": np.float32,
    "win_indicator": np.float32,
    "done": np.float32,
    "agent_aux_reward": np.float32,
}


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sanitize_logits(logits: torch.Tensor, clamp_abs: float = 30.0) -> torch.Tensor:
    logits = torch.nan_to_num(logits, nan=0.0, posinf=clamp_abs, neginf=-clamp_abs)
    return torch.clamp(logits, -clamp_abs, clamp_abs)


def ensure_valid_action_mask(masks: torch.Tensor) -> torch.Tensor:
    if masks.numel() <= 0:
        return masks
    has_any = torch.any(masks, dim=-1, keepdim=True)
    safe_masks = masks.clone()
    safe_masks[..., 0] = safe_masks[..., 0] | (~has_any.squeeze(-1))
    return safe_masks


def masked_categorical(logits: torch.Tensor, masks: torch.Tensor) -> Categorical:
    invalid_logit = torch.finfo(logits.dtype).min
    masked_logits = logits.masked_fill(~masks, invalid_logit)
    return Categorical(logits=masked_logits)


def _fire_decision_logits_from_attack_logits(attack_logits: torch.Tensor, attack_masks: torch.Tensor) -> torch.Tensor:
    invalid_logit = torch.finfo(attack_logits.dtype).min
    masked_logits = attack_logits.masked_fill(~attack_masks, invalid_logit)
    no_fire_logit = masked_logits[:, 0]
    fire_logit = torch.logsumexp(masked_logits[:, 1:], dim=-1)
    has_legal_nonzero = torch.any(attack_masks[:, 1:], dim=-1)
    fire_logit = torch.where(has_legal_nonzero, fire_logit, torch.full_like(fire_logit, invalid_logit))
    return torch.stack([no_fire_logit, fire_logit], dim=-1)


def _choose_nearest_target_attack(
    attack_mask_row: np.ndarray,
    visible_target_ids_row: np.ndarray,
    prefer_long: bool,
) -> int:
    legal_nonzero = np.flatnonzero(attack_mask_row[1:]) + 1
    if legal_nonzero.size <= 0:
        return 0

    # Action mapping: [1..10]=long missile on target id, [11..20]=short missile on target id.
    for target_id_raw in visible_target_ids_row:
        target_id = int(target_id_raw)
        if target_id <= 0 or target_id > FIGHTER_NUM:
            continue
        long_idx = int(target_id)
        short_idx = int(target_id + FIGHTER_NUM)
        long_legal = long_idx < attack_mask_row.shape[0] and bool(attack_mask_row[long_idx])
        short_legal = short_idx < attack_mask_row.shape[0] and bool(attack_mask_row[short_idx])
        if prefer_long:
            if long_legal:
                return long_idx
            if short_legal:
                return short_idx
        else:
            if short_legal:
                return short_idx
            if long_legal:
                return long_idx

    long_candidates = legal_nonzero[(legal_nonzero >= 1) & (legal_nonzero <= FIGHTER_NUM)]
    short_candidates = legal_nonzero[(legal_nonzero > FIGHTER_NUM)]
    if prefer_long and long_candidates.size > 0:
        return int(np.min(long_candidates))
    if (not prefer_long) and short_candidates.size > 0:
        return int(np.min(short_candidates))
    return int(np.min(legal_nonzero))


def _choose_best_logit_attack(attack_logits_row: torch.Tensor, attack_mask_row: np.ndarray) -> int:
    legal_nonzero = np.flatnonzero(attack_mask_row[1:]) + 1
    if legal_nonzero.size <= 0:
        return 0
    legal_idx_t = torch.as_tensor(legal_nonzero, dtype=torch.long, device=attack_logits_row.device)
    best_rel = int(torch.argmax(attack_logits_row[legal_idx_t]).item())
    return int(legal_nonzero[best_rel])


def _map_fire_or_not_to_attack_actions(
    fire_decision: np.ndarray,
    attack_masks: np.ndarray,
    visible_target_ids: np.ndarray,
    attack_logits: torch.Tensor,
    rule_mode: str,
    prefer_long: bool,
) -> np.ndarray:
    out = np.zeros((fire_decision.shape[0],), dtype=np.int64)
    for row_idx in range(fire_decision.shape[0]):
        mask_row = attack_masks[row_idx]
        if not np.any(mask_row[1:]):
            out[row_idx] = 0
            continue
        if int(fire_decision[row_idx]) <= 0:
            out[row_idx] = 0
            continue

        if rule_mode == "nearest_target":
            out[row_idx] = _choose_nearest_target_attack(
                attack_mask_row=mask_row,
                visible_target_ids_row=visible_target_ids[row_idx],
                prefer_long=prefer_long,
            )
        else:
            out[row_idx] = _choose_best_logit_attack(attack_logits[row_idx], mask_row)
    return out


def _remap_full_discrete_attack_with_rule(
    sampled_attack_action: np.ndarray,
    attack_masks: np.ndarray,
    visible_target_ids: np.ndarray,
    rule_mode: str,
    prefer_long: bool,
) -> np.ndarray:
    if rule_mode != "nearest_target":
        return sampled_attack_action.astype(np.int64, copy=False)

    remapped = sampled_attack_action.astype(np.int64, copy=True)
    for row_idx in range(remapped.shape[0]):
        action = int(remapped[row_idx])
        mask_row = attack_masks[row_idx]
        if action <= 0 or not np.any(mask_row[1:]):
            remapped[row_idx] = 0
            continue
        remapped[row_idx] = _choose_nearest_target_attack(
            attack_mask_row=mask_row,
            visible_target_ids_row=visible_target_ids[row_idx],
            prefer_long=prefer_long,
        )
    return remapped


def build_env(args, seed_offset: int) -> MAPPOMaCAEnv:
    runtime_opponent = str(getattr(args, "runtime_maca_opponent", args.maca_opponent))
    runtime_max_step = int(getattr(args, "runtime_maca_max_step", args.maca_max_step))
    runtime_random_pos = bool(getattr(args, "runtime_maca_random_pos", args.maca_random_pos))
    return MAPPOMaCAEnv(
        MAPPOMaCAConfig(
            map_path=args.maca_map_path,
            red_obs_ind=args.maca_red_obs_ind,
            opponent=runtime_opponent,
            max_step=runtime_max_step,
            render=args.maca_render,
            random_pos=runtime_random_pos,
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
            mode_reward_scale=args.maca_mode_reward_scale,
            exec_reward_scale=args.maca_exec_reward_scale,
            disengage_penalty=args.maca_disengage_penalty,
            bearing_reward_scale=args.maca_bearing_reward_scale,
            semantic_screen_downsample=args.maca_semantic_screen_downsample,
            terminal_ammo_fail_penalty=args.maca_terminal_ammo_fail_penalty,
            terminal_participation_penalty=args.maca_terminal_participation_penalty,
        )
    )


def sample_actions(model, batch, device, deterministic: bool = False, args=None):
    if torch.is_tensor(batch["local_obs"]):
        local_obs = batch["local_obs"].to(device=device, dtype=torch.float32)
    else:
        local_obs = torch.as_tensor(batch["local_obs"], dtype=torch.float32, device=device)

    if torch.is_tensor(batch["local_screen"]):
        local_screen = batch["local_screen"].to(device=device, dtype=torch.uint8)
    else:
        local_screen = torch.as_tensor(batch["local_screen"], dtype=torch.uint8, device=device)

    if torch.is_tensor(batch["attack_masks"]):
        attack_masks = batch["attack_masks"].to(device=device, dtype=torch.bool)
    else:
        attack_masks = torch.as_tensor(batch["attack_masks"], dtype=torch.bool, device=device)

    if torch.is_tensor(batch["actor_h"]):
        actor_h = batch["actor_h"].to(device=device, dtype=torch.float32)
    else:
        actor_h = torch.as_tensor(batch["actor_h"], dtype=torch.float32, device=device)

    flat_local = local_obs.reshape(-1, local_obs.shape[-1])
    flat_screen = local_screen.reshape(
        local_screen.shape[0] * local_screen.shape[1],
        local_screen.shape[2],
        local_screen.shape[3],
        local_screen.shape[4],
    )
    flat_attack_masks = attack_masks.reshape(-1, attack_masks.shape[-1])
    flat_attack_masks = ensure_valid_action_mask(flat_attack_masks)
    flat_attack_masks_np = flat_attack_masks.cpu().numpy().astype(np.bool_, copy=False)
    flat_actor_h = actor_h.reshape(-1, actor_h.shape[-1])

    if "rule_visible_target_ids" in batch:
        if torch.is_tensor(batch["rule_visible_target_ids"]):
            rule_visible_target_ids = batch["rule_visible_target_ids"].to(device=device, dtype=torch.long)
        else:
            rule_visible_target_ids = torch.as_tensor(
                batch["rule_visible_target_ids"], dtype=torch.long, device=device
            )
        flat_visible_target_ids = rule_visible_target_ids.reshape(
            rule_visible_target_ids.shape[0] * rule_visible_target_ids.shape[1],
            rule_visible_target_ids.shape[-1],
        )
    else:
        flat_visible_target_ids = torch.zeros((flat_attack_masks.shape[0], 0), dtype=torch.long, device=device)

    mode_actions = None
    if "mode_action" in batch:
        if torch.is_tensor(batch["mode_action"]):
            mode_actions = batch["mode_action"].to(device=device, dtype=torch.long)
        else:
            mode_actions = torch.as_tensor(batch["mode_action"], dtype=torch.long, device=device)
        mode_actions = mode_actions.reshape(-1)

    course_logits, _attack_logits_unused, next_actor_h = model.actor_step(
        flat_local,
        flat_screen,
        flat_actor_h,
        mode_actions=mode_actions,
    )
    course_logits = sanitize_logits(course_logits)
    course_dist = Categorical(logits=course_logits)
    if deterministic:
        course_action = torch.argmax(course_logits, dim=-1)
    else:
        course_action = course_dist.sample()

    attack_logits = model.attack_logits(next_actor_h, course_action, mode_actions=mode_actions)
    attack_logits = sanitize_logits(attack_logits)
    attack_policy_mode = str(getattr(args, "attack_policy_mode", "full_discrete")).lower()
    attack_rule_mode = str(getattr(args, "attack_rule_mode", "none")).lower()
    attack_rule_prefer_long = bool(getattr(args, "attack_rule_prefer_long", True))

    if attack_policy_mode == "fire_or_not":
        fire_decision_logits = _fire_decision_logits_from_attack_logits(attack_logits, flat_attack_masks)
        fire_dist = Categorical(logits=fire_decision_logits)
        if deterministic:
            policy_attack_action = torch.argmax(fire_decision_logits, dim=-1)
        else:
            policy_attack_action = fire_dist.sample()

        executed_attack_np = _map_fire_or_not_to_attack_actions(
            fire_decision=policy_attack_action.cpu().numpy().astype(np.int64, copy=False),
            attack_masks=flat_attack_masks_np,
            visible_target_ids=flat_visible_target_ids.cpu().numpy().astype(np.int64, copy=False),
            attack_logits=attack_logits,
            rule_mode=attack_rule_mode,
            prefer_long=attack_rule_prefer_long,
        )
        attack_action = torch.as_tensor(executed_attack_np, dtype=torch.long, device=device)
        log_prob = course_dist.log_prob(course_action) + fire_dist.log_prob(policy_attack_action)
        entropy = course_dist.entropy() + fire_dist.entropy()
    else:
        attack_dist = masked_categorical(attack_logits, flat_attack_masks)
        if deterministic:
            sampled_attack = torch.argmax(attack_dist.logits, dim=-1)
        else:
            sampled_attack = attack_dist.sample()
        attack_action = sampled_attack
        policy_attack_action = sampled_attack
        log_prob = course_dist.log_prob(course_action) + attack_dist.log_prob(policy_attack_action)
        entropy = course_dist.entropy() + attack_dist.entropy()

    course_action = course_action.reshape(local_obs.shape[0], local_obs.shape[1])
    attack_action = attack_action.reshape(local_obs.shape[0], local_obs.shape[1])
    policy_attack_action = policy_attack_action.reshape(local_obs.shape[0], local_obs.shape[1])
    log_prob = log_prob.reshape(local_obs.shape[0], local_obs.shape[1])
    entropy = entropy.reshape(local_obs.shape[0], local_obs.shape[1])
    next_actor_h = next_actor_h.reshape(local_obs.shape[0], local_obs.shape[1], -1)

    return {
        "course_action": course_action,
        "attack_action": attack_action,
        "policy_attack_action": policy_attack_action,
        "log_prob": log_prob,
        "entropy": entropy,
        "next_actor_h": next_actor_h,
    }


def actor_parameter_items(model):
    return [(name, p) for name, p in model.named_parameters() if not name.startswith("critic")]


def build_actor_param_layout(model):
    layout = []
    offset = 0
    for name, p in actor_parameter_items(model):
        numel = int(p.numel())
        layout.append(
            {
                "name": name,
                "shape": tuple(p.shape),
                "offset": int(offset),
                "numel": int(numel),
            }
        )
        offset += numel
    return layout, int(offset)


def allocate_shared_actor_params(model):
    layout, total_numel = build_actor_param_layout(model)
    raw = mp.RawArray(ctypes.c_float, total_numel)
    version = mp.RawValue(ctypes.c_longlong, 0)
    return {
        "raw": raw,
        "size": int(total_numel),
        "layout": layout,
        "version": version,
    }


def shared_actor_flat_view(shared_actor):
    return np.frombuffer(shared_actor["raw"], dtype=np.float32, count=int(shared_actor["size"]))


def write_shared_actor_params(model, shared_actor, increment_version: bool = True):
    flat = shared_actor_flat_view(shared_actor)
    actor_params = dict(actor_parameter_items(model))
    for entry in shared_actor["layout"]:
        name = entry["name"]
        begin = int(entry["offset"])
        end = begin + int(entry["numel"])
        tensor = actor_params[name].detach().to(device="cpu", dtype=torch.float32).reshape(-1)
        flat[begin:end] = tensor.numpy()
    if increment_version:
        shared_actor["version"].value = int(shared_actor["version"].value) + 1


def load_shared_actor_params_into_model(model, shared_actor):
    flat = shared_actor_flat_view(shared_actor)
    params = dict(model.named_parameters())
    with torch.no_grad():
        for entry in shared_actor["layout"]:
            name = entry["name"]
            begin = int(entry["offset"])
            end = begin + int(entry["numel"])
            shape = entry["shape"]
            src = torch.from_numpy(flat[begin:end].reshape(shape))
            dst = params[name]
            dst.copy_(src.to(device=dst.device, dtype=dst.dtype))


def make_shared_ndarray(shape, dtype):
    dtype = np.dtype(dtype)
    c_type = np.ctypeslib.as_ctypes_type(dtype)
    size = int(np.prod(shape))
    raw = mp.RawArray(c_type, size)
    array = np.frombuffer(raw, dtype=dtype, count=size).reshape(shape)
    return raw, array


def shared_ndarray_view(raw, shape, dtype):
    dtype = np.dtype(dtype)
    size = int(np.prod(shape))
    return np.frombuffer(raw, dtype=dtype, count=size).reshape(shape)


def build_worker_buffer_shapes(rollout_steps: int, env_count: int, env_spec: dict):
    num_agents = int(env_spec["num_agents"])
    local_obs_dim = int(env_spec["local_obs_dim"])
    local_screen_h = int(env_spec["local_screen_shape"][0])
    local_screen_w = int(env_spec["local_screen_shape"][1])
    local_screen_c = int(env_spec["local_screen_shape"][2])
    global_state_dim = int(env_spec["global_state_dim"])
    attack_dim = int(env_spec["attack_dim"])
    actor_hidden_dim = int(env_spec["actor_hidden_dim"])
    return {
        "local_obs": (rollout_steps, env_count, num_agents, local_obs_dim),
        "local_screen": (rollout_steps, env_count, num_agents, local_screen_h, local_screen_w, local_screen_c),
        "global_state": (rollout_steps, env_count, global_state_dim),
        "attack_masks": (rollout_steps, env_count, num_agents, attack_dim),
        "alive_mask": (rollout_steps, env_count, num_agents),
        "actor_h": (rollout_steps, env_count, num_agents, actor_hidden_dim),
        "course_action": (rollout_steps, env_count, num_agents),
        "attack_action": (rollout_steps, env_count, num_agents),
        "policy_attack_action": (rollout_steps, env_count, num_agents),
        "mode_action": (rollout_steps, env_count, num_agents),
        "log_prob": (rollout_steps, env_count, num_agents),
        "mode_log_prob": (rollout_steps, env_count, num_agents),
        "mode_decision": (rollout_steps, env_count, num_agents),
        "mode_duration": (rollout_steps, env_count, num_agents),
        "reward": (rollout_steps, env_count),
        "reward_env": (rollout_steps, env_count),
        "reward_mode": (rollout_steps, env_count),
        "reward_exec": (rollout_steps, env_count),
        "reward_terminal": (rollout_steps, env_count),
        "contact_signal": (rollout_steps, env_count),
        "opportunity_signal": (rollout_steps, env_count),
        "damage_reward": (rollout_steps, env_count),
        "kill_reward": (rollout_steps, env_count),
        "survival_reward": (rollout_steps, env_count),
        "win_indicator": (rollout_steps, env_count),
        "done": (rollout_steps, env_count),
        "agent_aux_reward": (rollout_steps, env_count, num_agents),
        "final_global_state": (env_count, global_state_dim),
    }


def allocate_worker_shared_buffers(rollout_steps: int, env_count: int, env_spec: dict):
    shapes = build_worker_buffer_shapes(rollout_steps, env_count, env_spec)
    shared = {}
    for key, shape in shapes.items():
        dtype = np.float32 if key == "final_global_state" else ROLLOUT_BUFFER_DTYPES[key]
        raw, _view = make_shared_ndarray(shape, dtype)
        shared[key] = {
            "raw": raw,
            "shape": shape,
            "dtype": np.dtype(dtype).str,
        }
    return shared


def attach_worker_shared_buffers(shared_buffers):
    attached = {}
    for key, meta in shared_buffers.items():
        attached[key] = shared_ndarray_view(meta["raw"], meta["shape"], np.dtype(meta["dtype"]))
    return attached


def _collector_process_main(args, worker_idx: int, env_count: int, conn, env_spec: dict, shared_buffers, shared_actor):
    torch.set_num_threads(1)
    set_seed(args.seed + worker_idx * 1009)
    envs = [build_env(args, seed_offset=worker_idx * 100000 + env_idx * 9973) for env_idx in range(env_count)]
    worker_model = None
    obs_batch = None
    shared_views = attach_worker_shared_buffers(shared_buffers)
    local_actor_version = -1

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
                rollout_steps = int(payload)
                if obs_batch is None:
                    obs_batch = [env.reset() for env in envs]

                if worker_model is None:
                    local_obs_dim = int(env_spec["local_obs_dim"])
                    local_screen_shape = tuple(env_spec["local_screen_shape"])
                    global_state_dim = obs_batch[0]["global_state"].shape[0]
                    num_agents = obs_batch[0]["local_obs"].shape[0]
                    worker_model = TeamActorCritic(
                        MAPPOModelConfig(
                            local_obs_dim=local_obs_dim,
                            local_screen_shape=local_screen_shape,
                            global_state_dim=global_state_dim,
                            num_agents=num_agents,
                            hidden_size=args.hidden_size,
                            screen_embed_dim=args.screen_embed_dim,
                            course_embed_dim=args.course_embed_dim,
                        )
                    )
                    worker_model.eval()

                shared_version = int(shared_actor["version"].value)
                if shared_version != local_actor_version:
                    load_shared_actor_params_into_model(worker_model, shared_actor)
                    local_actor_version = shared_version

                actor_hidden_dim = worker_model.actor_hidden_dim
                num_agents = int(env_spec["num_agents"])
                disable_high_level_mode = bool(getattr(args, "disable_high_level_mode", False))
                mode_interval = int(env_spec.get("mode_interval", 8))
                attack_dim = int(env_spec["attack_dim"])
                final_local_obs_dim = int(env_spec["local_obs_dim"])
                local_screen_shape = tuple(env_spec["local_screen_shape"])
                global_state_dim = int(obs_batch[0]["global_state"].shape[0])
                visible_target_slots = int(obs_batch[0].get("rule_visible_target_ids", np.zeros((num_agents, 0))).shape[1])

                local_obs_batch = np.empty((env_count, num_agents, final_local_obs_dim), dtype=np.float32)
                local_screen_batch = np.empty((env_count, num_agents) + local_screen_shape, dtype=np.uint8)
                global_state_batch = np.empty((env_count, global_state_dim), dtype=np.float32)
                attack_masks_batch = np.empty((env_count, num_agents, attack_dim), dtype=np.bool_)
                alive_mask_batch = np.empty((env_count, num_agents), dtype=np.float32)
                rule_visible_target_ids_batch = np.zeros((env_count, num_agents, visible_target_slots), dtype=np.int64)
                env_actions_batch = np.empty((env_count, num_agents, 2), dtype=np.int64)

                rewards = np.zeros((env_count,), dtype=np.float32)
                rewards_env = np.zeros((env_count,), dtype=np.float32)
                rewards_mode = np.zeros((env_count,), dtype=np.float32)
                rewards_exec = np.zeros((env_count,), dtype=np.float32)
                rewards_terminal = np.zeros((env_count,), dtype=np.float32)
                contact_signals = np.zeros((env_count,), dtype=np.float32)
                opportunity_signals = np.zeros((env_count,), dtype=np.float32)
                damage_rewards = np.zeros((env_count,), dtype=np.float32)
                kill_rewards = np.zeros((env_count,), dtype=np.float32)
                survival_rewards = np.zeros((env_count,), dtype=np.float32)
                win_indicators = np.zeros((env_count,), dtype=np.float32)
                dones = np.zeros((env_count,), dtype=np.float32)
                aux_rewards = np.zeros((env_count, num_agents), dtype=np.float32)
                local_obs_tensor = torch.from_numpy(local_obs_batch)
                local_screen_tensor = torch.from_numpy(local_screen_batch)
                global_state_tensor = torch.from_numpy(global_state_batch)
                attack_masks_tensor = torch.from_numpy(attack_masks_batch)
                alive_mask_tensor = torch.from_numpy(alive_mask_batch)
                rule_visible_target_ids_tensor = torch.from_numpy(rule_visible_target_ids_batch)

                actor_h_batch = np.zeros((env_count, num_agents, actor_hidden_dim), dtype=np.float32)
                actor_h_tensor = torch.from_numpy(actor_h_batch)
                mode_action_batch = np.zeros((env_count, num_agents), dtype=np.int64)
                mode_duration_batch = np.zeros((env_count, num_agents), dtype=np.int64)
                mode_steps_to_refresh = np.zeros((env_count, num_agents), dtype=np.int64)
                episodes = []
                rnn_hidden_mismatch_count = 0
                rnn_hidden_max_abs_diff = 0.0
                prev_expected_actor_h = None

                for step_idx in range(rollout_steps):
                    for env_idx, obs in enumerate(obs_batch):
                        local_obs_batch[env_idx] = obs["local_obs"]
                        local_screen_batch[env_idx] = obs["local_screen"]
                        global_state_batch[env_idx] = obs["global_state"]
                        attack_masks_batch[env_idx] = obs["attack_masks"]
                        alive_mask_batch[env_idx] = obs["alive_mask"]
                        if visible_target_slots > 0:
                            rule_visible_target_ids_batch[env_idx] = obs.get(
                                "rule_visible_target_ids",
                                np.zeros((num_agents, visible_target_slots), dtype=np.int64),
                            )

                    alive_bool = alive_mask_batch > 0.5
                    alive_count = np.maximum(np.sum(alive_bool, axis=1), 1)
                    contact_agent = np.any(rule_visible_target_ids_batch > 0, axis=-1) & alive_bool
                    opportunity_agent = np.any(attack_masks_batch[:, :, 1:], axis=-1) & alive_bool
                    contact_signals[:] = np.sum(contact_agent, axis=1) / alive_count
                    opportunity_signals[:] = np.sum(opportunity_agent, axis=1) / alive_count

                    if prev_expected_actor_h is not None:
                        diff = np.abs(actor_h_batch - prev_expected_actor_h)
                        max_diff = float(np.max(diff))
                        if max_diff > 1e-5:
                            rnn_hidden_mismatch_count += int(np.count_nonzero(diff > 1e-5))
                            rnn_hidden_max_abs_diff = max(rnn_hidden_max_abs_diff, max_diff)

                    if disable_high_level_mode:
                        mode_decision_mask = np.zeros((env_count, num_agents), dtype=np.bool_)
                        mode_action_batch.fill(0)
                        mode_log_prob_batch = np.zeros((env_count, num_agents), dtype=np.float32)
                        mode_duration_batch.fill(0)
                        mode_steps_to_refresh.fill(0)
                    else:
                        mode_decision_mask = mode_steps_to_refresh <= 0
                        with torch.no_grad():
                            flat_actor_h_t = torch.as_tensor(
                                actor_h_batch.reshape(-1, actor_hidden_dim),
                                dtype=torch.float32,
                                device=torch.device("cpu"),
                            )
                            mode_logits = worker_model.mode_head(flat_actor_h_t)
                            mode_dist = Categorical(logits=mode_logits)
                            sampled_mode_t = mode_dist.sample()
                            sampled_mode = sampled_mode_t.reshape(env_count, num_agents).cpu().numpy()
                            sampled_mode_log_prob = (
                                mode_dist.log_prob(sampled_mode_t).reshape(env_count, num_agents).cpu().numpy()
                            )
                            prev_mode_t = torch.as_tensor(
                                mode_action_batch.reshape(-1),
                                dtype=torch.long,
                                device=torch.device("cpu"),
                            )
                            prev_mode_log_prob = (
                                mode_dist.log_prob(prev_mode_t).reshape(env_count, num_agents).cpu().numpy()
                            )

                        mode_action_batch = np.where(mode_decision_mask, sampled_mode, mode_action_batch)
                        mode_log_prob_batch = np.where(
                            mode_decision_mask,
                            sampled_mode_log_prob,
                            prev_mode_log_prob,
                        ).astype(np.float32, copy=False)
                        mode_duration_batch = np.where(mode_decision_mask, 1, mode_duration_batch + 1).astype(
                            np.int64, copy=False
                        )
                        mode_steps_to_refresh = np.where(
                            mode_decision_mask,
                            max(mode_interval - 1, 0),
                            mode_steps_to_refresh - 1,
                        ).astype(np.int64, copy=False)

                    stacked = {
                        "local_obs": local_obs_tensor,
                        "local_screen": local_screen_tensor,
                        "global_state": global_state_tensor,
                        "attack_masks": attack_masks_tensor,
                        "alive_mask": alive_mask_tensor,
                        "rule_visible_target_ids": rule_visible_target_ids_tensor,
                        "actor_h": actor_h_tensor,
                        "mode_action": torch.from_numpy(mode_action_batch),
                    }

                    with torch.no_grad():
                        actions = sample_actions(worker_model, stacked, torch.device("cpu"), deterministic=False, args=args)

                    course_action_np = actions["course_action"].numpy()
                    attack_action_np = actions["attack_action"].numpy()
                    policy_attack_action_np = actions["policy_attack_action"].numpy()
                    log_prob_np = actions["log_prob"].numpy()
                    next_actor_h_np = actions["next_actor_h"].numpy()
                    env_actions_batch[:, :, 0] = course_action_np
                    env_actions_batch[:, :, 1] = attack_action_np

                    shared_views["local_obs"][step_idx] = local_obs_batch
                    shared_views["local_screen"][step_idx] = local_screen_batch
                    shared_views["global_state"][step_idx] = global_state_batch
                    shared_views["attack_masks"][step_idx] = attack_masks_batch.astype(np.uint8, copy=False)
                    shared_views["alive_mask"][step_idx] = alive_mask_batch
                    shared_views["actor_h"][step_idx] = actor_h_batch
                    shared_views["course_action"][step_idx] = course_action_np
                    shared_views["attack_action"][step_idx] = attack_action_np
                    shared_views["policy_attack_action"][step_idx] = policy_attack_action_np
                    shared_views["mode_action"][step_idx] = mode_action_batch
                    shared_views["log_prob"][step_idx] = log_prob_np
                    shared_views["mode_log_prob"][step_idx] = mode_log_prob_batch
                    shared_views["mode_decision"][step_idx] = mode_decision_mask.astype(np.float32, copy=False)
                    shared_views["mode_duration"][step_idx] = mode_duration_batch

                    next_obs_batch = []
                    rewards.fill(0.0)
                    rewards_env.fill(0.0)
                    rewards_mode.fill(0.0)
                    rewards_exec.fill(0.0)
                    rewards_terminal.fill(0.0)
                    damage_rewards.fill(0.0)
                    kill_rewards.fill(0.0)
                    survival_rewards.fill(0.0)
                    win_indicators.fill(0.0)
                    dones.fill(0.0)
                    aux_rewards.fill(0.0)

                    for env_idx, env in enumerate(envs):
                        next_obs, reward, done, info = env.step(env_actions_batch[env_idx])
                        rewards[env_idx] = float(reward)
                        rewards_env[env_idx] = float(info.get("reward_env", 0.0))
                        rewards_mode[env_idx] = float(info.get("reward_mode", 0.0))
                        rewards_exec[env_idx] = float(info.get("reward_exec", 0.0))
                        rewards_terminal[env_idx] = float(info.get("reward_terminal", 0.0))
                        damage_rewards[env_idx] = float(info.get("damage_reward", 0.0))
                        kill_rewards[env_idx] = float(info.get("kill_reward", 0.0))
                        survival_rewards[env_idx] = float(info.get("survival_reward", 0.0))
                        win_indicators[env_idx] = float(info.get("win_indicator", 0.0))
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

                    shared_views["reward"][step_idx] = rewards
                    shared_views["reward_env"][step_idx] = rewards_env
                    shared_views["reward_mode"][step_idx] = rewards_mode
                    shared_views["reward_exec"][step_idx] = rewards_exec
                    shared_views["reward_terminal"][step_idx] = rewards_terminal
                    shared_views["contact_signal"][step_idx] = contact_signals
                    shared_views["opportunity_signal"][step_idx] = opportunity_signals
                    shared_views["damage_reward"][step_idx] = damage_rewards
                    shared_views["kill_reward"][step_idx] = kill_rewards
                    shared_views["survival_reward"][step_idx] = survival_rewards
                    shared_views["win_indicator"][step_idx] = win_indicators
                    shared_views["done"][step_idx] = dones
                    shared_views["agent_aux_reward"][step_idx] = aux_rewards

                    actor_h_batch[:] = next_actor_h_np
                    if np.any(dones > 0.5):
                        done_rows = np.where(dones > 0.5)[0]
                        actor_h_batch[done_rows, :, :] = 0.0
                        mode_action_batch[done_rows, :] = 0
                        mode_duration_batch[done_rows, :] = 0
                        mode_steps_to_refresh[done_rows, :] = 0

                    if prev_expected_actor_h is None:
                        prev_expected_actor_h = np.zeros_like(actor_h_batch)
                    prev_expected_actor_h[:] = actor_h_batch
                    obs_batch = next_obs_batch

                final_global_state = np.stack([obs["global_state"] for obs in obs_batch], axis=0)
                shared_views["final_global_state"][:] = final_global_state

                conn.send(
                    {
                        "episodes": episodes,
                        "rnn_hidden_mismatch_count": int(rnn_hidden_mismatch_count),
                        "rnn_hidden_max_abs_diff": float(rnn_hidden_max_abs_diff),
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
    def __init__(self, args, env_spec: Dict, shared_actor):
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
        self.rollout_steps = int(args.rollout)
        self.env_spec = env_spec
        self.shared_actor = shared_actor
        self.worker_shared_meta = []
        self.worker_shared_views = []
        self.worker_env_slices = []

        env_begin = 0
        for env_count in self.worker_env_counts:
            env_end = env_begin + int(env_count)
            self.worker_env_slices.append((env_begin, env_end))
            env_begin = env_end

        for worker_idx, env_count in enumerate(self.worker_env_counts):
            shared_buffers = allocate_worker_shared_buffers(self.rollout_steps, env_count, env_spec)
            parent_conn, child_conn = ctx.Pipe()
            process = ctx.Process(
                target=_collector_process_main,
                args=(args, worker_idx, env_count, child_conn, env_spec, shared_buffers, shared_actor),
                daemon=True,
            )
            process.start()
            child_conn.close()
            self.parent_conns.append(parent_conn)
            self.processes.append(process)
            self.worker_shared_meta.append(shared_buffers)
            self.worker_shared_views.append(attach_worker_shared_buffers(shared_buffers))

    def reset(self, seed=None):
        for worker_idx, conn in enumerate(self.parent_conns):
            worker_seed = None if seed is None else int(seed + worker_idx * 1000)
            conn.send(("reset", worker_seed))

        obs_batch = []
        for conn in self.parent_conns:
            obs_batch.extend(conn.recv())
        return obs_batch

    def collect(self, rollout_steps: int):
        for conn in self.parent_conns:
            conn.send(("collect", int(rollout_steps)))

        results = []
        for worker_idx, conn in enumerate(self.parent_conns):
            worker_result = conn.recv()
            worker_result["buffer"] = self.worker_shared_views[worker_idx]
            worker_result["final_global_state"] = self.worker_shared_views[worker_idx]["final_global_state"]
            results.append(worker_result)
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
