from __future__ import annotations

import json
import time

import numpy as np
import torch

from fighter_action_utils import FIGHTER_NUM
from marl_env.mappo_env import MAPPOMaCAConfig, MAPPOMaCAEnv
from marl_train.checkpoint import eval_dir
from marl_train.logging_utils import summarize_episode_stats


def append_agent_id_onehot(local_obs: np.ndarray, agent_ids: np.ndarray, num_agents: int) -> np.ndarray:
    if local_obs.ndim != 3:
        raise ValueError("Expected local_obs shape [env, agent, dim], got %s" % (local_obs.shape,))
    one_hot = np.eye(num_agents, dtype=np.float32)[agent_ids]
    return np.concatenate([local_obs.astype(np.float32, copy=False), one_hot], axis=-1)


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


def fire_decision_logits_from_attack_logits(attack_logits: torch.Tensor, attack_masks: torch.Tensor) -> torch.Tensor:
    invalid_logit = torch.finfo(attack_logits.dtype).min
    masked_logits = attack_logits.masked_fill(~attack_masks, invalid_logit)
    no_fire_logit = masked_logits[:, 0]
    fire_logit = torch.logsumexp(masked_logits[:, 1:], dim=-1)
    has_legal_nonzero = torch.any(attack_masks[:, 1:], dim=-1)
    fire_logit = torch.where(has_legal_nonzero, fire_logit, torch.full_like(fire_logit, invalid_logit))
    return torch.stack([no_fire_logit, fire_logit], dim=-1)


def choose_nearest_target_attack(mask_row: np.ndarray, visible_target_ids_row: np.ndarray, prefer_long: bool) -> int:
    legal_nonzero = np.flatnonzero(mask_row[1:]) + 1
    if legal_nonzero.size <= 0:
        return 0

    for target_id_raw in visible_target_ids_row:
        target_id = int(target_id_raw)
        if target_id <= 0 or target_id > FIGHTER_NUM:
            continue
        long_idx = int(target_id)
        short_idx = int(target_id + FIGHTER_NUM)
        long_legal = long_idx < mask_row.shape[0] and bool(mask_row[long_idx])
        short_legal = short_idx < mask_row.shape[0] and bool(mask_row[short_idx])
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


def select_eval_actions(
    model,
    obs,
    actor_h,
    device,
    deterministic: bool,
    concat_agent_id_onehot: bool,
    num_agents: int,
    attack_rule_mode: str,
    attack_policy_mode: str,
    attack_rule_prefer_long: bool,
):
    local_obs_np = obs["local_obs"]
    if concat_agent_id_onehot:
        local_obs_np = append_agent_id_onehot(local_obs_np[None, ...], obs["agent_ids"][None, ...], num_agents)[0]
    local_obs = torch.as_tensor(local_obs_np, dtype=torch.float32, device=device)
    agent_ids = torch.as_tensor(obs["agent_ids"], dtype=torch.long, device=device)
    attack_masks = torch.as_tensor(obs["attack_masks"], dtype=torch.bool, device=device)
    attack_masks = ensure_valid_action_mask(attack_masks)
    visible_target_ids = np.asarray(obs.get("rule_visible_target_ids", np.zeros((num_agents, 0), dtype=np.int64)), dtype=np.int64)
    actor_h_t = torch.as_tensor(actor_h, dtype=torch.float32, device=device)

    with torch.no_grad():
        course_logits, _attack_logits_unused, next_actor_h = model.actor_step(local_obs, agent_ids, actor_h_t)
        course_logits = sanitize_logits(course_logits)
        if deterministic:
            course_action = torch.argmax(course_logits, dim=-1)
        else:
            course_action = torch.distributions.Categorical(logits=course_logits).sample()

        attack_logits = model.attack_logits(next_actor_h, course_action)
        attack_logits = sanitize_logits(attack_logits)
        attack_policy_mode = str(attack_policy_mode).lower()
        attack_rule_mode = str(attack_rule_mode).lower()

        if attack_policy_mode == "fire_or_not":
            fire_logits = fire_decision_logits_from_attack_logits(attack_logits, attack_masks)
            if deterministic:
                fire_decision = torch.argmax(fire_logits, dim=-1).cpu().numpy().astype(np.int64, copy=False)
            else:
                fire_decision = (
                    torch.distributions.Categorical(logits=fire_logits).sample().cpu().numpy().astype(np.int64, copy=False)
                )

            attack_masks_np = attack_masks.cpu().numpy().astype(np.bool_, copy=False)
            attack_logits_cpu = attack_logits.detach().cpu()
            attack_action_np = np.zeros((num_agents,), dtype=np.int64)
            for idx in range(num_agents):
                if not np.any(attack_masks_np[idx, 1:]) or int(fire_decision[idx]) <= 0:
                    attack_action_np[idx] = 0
                    continue
                if attack_rule_mode == "nearest_target":
                    attack_action_np[idx] = choose_nearest_target_attack(
                        mask_row=attack_masks_np[idx],
                        visible_target_ids_row=visible_target_ids[idx],
                        prefer_long=bool(attack_rule_prefer_long),
                    )
                else:
                    legal_nonzero = np.flatnonzero(attack_masks_np[idx, 1:]) + 1
                    legal_idx_t = torch.as_tensor(legal_nonzero, dtype=torch.long)
                    best_rel = int(torch.argmax(attack_logits_cpu[idx][legal_idx_t]).item())
                    attack_action_np[idx] = int(legal_nonzero[best_rel])
            attack_action = torch.as_tensor(attack_action_np, dtype=torch.long, device=device)
        else:
            invalid_logit = torch.finfo(attack_logits.dtype).min
            masked_attack_logits = attack_logits.masked_fill(~attack_masks, invalid_logit)
            if deterministic:
                attack_action = torch.argmax(masked_attack_logits, dim=-1)
            else:
                attack_action = torch.distributions.Categorical(logits=masked_attack_logits).sample()

            if attack_rule_mode == "nearest_target":
                attack_masks_np = attack_masks.cpu().numpy().astype(np.bool_, copy=False)
                sampled_np = attack_action.cpu().numpy().astype(np.int64, copy=False)
                remap_np = sampled_np.copy()
                for idx in range(num_agents):
                    if int(sampled_np[idx]) <= 0 or not np.any(attack_masks_np[idx, 1:]):
                        remap_np[idx] = 0
                    else:
                        remap_np[idx] = choose_nearest_target_attack(
                            mask_row=attack_masks_np[idx],
                            visible_target_ids_row=visible_target_ids[idx],
                            prefer_long=bool(attack_rule_prefer_long),
                        )
                attack_action = torch.as_tensor(remap_np, dtype=torch.long, device=device)

    actions = np.stack([course_action.cpu().numpy(), attack_action.cpu().numpy()], axis=-1)
    return actions, next_actor_h.cpu().numpy()


def run_evaluation(model, device, args, env_steps: int):
    runtime_opponent = str(getattr(args, "runtime_maca_opponent", args.maca_opponent))
    runtime_max_step = int(getattr(args, "runtime_maca_max_step", args.maca_max_step))
    runtime_random_pos = bool(getattr(args, "runtime_maca_random_pos", args.maca_random_pos))
    eval_opponent = args.eval_opponent or runtime_opponent
    eval_config = MAPPOMaCAConfig(
        map_path=args.maca_map_path,
        red_obs_ind=args.maca_red_obs_ind,
        opponent=eval_opponent,
        max_step=runtime_max_step,
        render=False,
        random_pos=runtime_random_pos,
        random_seed=args.seed + 900000 + int(env_steps),
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
    )
    env = MAPPOMaCAEnv(eval_config)
    actor_h = np.zeros((env.num_agents, model.actor_hidden_dim), dtype=np.float32)
    episode_results = []
    start_time = time.time()

    try:
        obs = env.reset(seed=eval_config.random_seed)
        while len(episode_results) < args.eval_episodes:
            actions, actor_h = select_eval_actions(
                model,
                obs,
                actor_h,
                device,
                args.eval_deterministic,
                concat_agent_id_onehot=bool(args.concat_agent_id_onehot),
                num_agents=env.num_agents,
                attack_rule_mode=str(getattr(args, "attack_rule_mode", "none")),
                attack_policy_mode=str(getattr(args, "attack_policy_mode", "full_discrete")),
                attack_rule_prefer_long=bool(getattr(args, "attack_rule_prefer_long", True)),
            )
            obs, _reward, done, info = env.step(actions)
            if not done:
                continue
            episode_results.append(info["episode_extra_stats"])
            actor_h.fill(0.0)
            obs = env.reset(seed=eval_config.random_seed + len(episode_results))
    finally:
        env.close()

    summary = summarize_episode_stats(episode_results)
    payload = {
        "env_steps": int(env_steps),
        "episodes": int(args.eval_episodes),
        "maca_opponent": eval_opponent,
        "deterministic": bool(args.eval_deterministic),
        "eval_wall_time_sec": float(time.time() - start_time),
        "summary": summary,
        "episodes_detail": episode_results,
    }
    eval_output_dir = eval_dir(args)
    eval_output_dir.mkdir(parents=True, exist_ok=True)
    output_path = eval_output_dir / ("eval_%09d.json" % int(env_steps))
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    return payload, output_path
