from __future__ import annotations

import json
import time

import numpy as np
import torch
from torch.distributions import Categorical

from marl_env.mappo_env import MAPPOMaCAConfig, MAPPOMaCAEnv
from marl_train.collector import sample_actions
from marl_train.checkpoint import eval_dir
from marl_train.logging_utils import summarize_episode_stats


def init_eval_policy_state(num_agents: int, actor_hidden_dim: int):
    return {
        "actor_h": np.zeros((int(num_agents), int(actor_hidden_dim)), dtype=np.float32),
        "mode_action": np.zeros((int(num_agents),), dtype=np.int64),
        "mode_duration": np.zeros((int(num_agents),), dtype=np.int64),
        "mode_steps_to_refresh": np.zeros((int(num_agents),), dtype=np.int64),
    }


def reset_eval_policy_state(policy_state) -> None:
    policy_state["actor_h"].fill(0.0)
    policy_state["mode_action"].fill(0)
    policy_state["mode_duration"].fill(0)
    policy_state["mode_steps_to_refresh"].fill(0)


def _update_eval_mode_state(model, policy_state, device, deterministic: bool, disable_high_level_mode: bool, mode_interval: int):
    mode_action = policy_state["mode_action"]
    mode_duration = policy_state["mode_duration"]
    mode_steps_to_refresh = policy_state["mode_steps_to_refresh"]

    if disable_high_level_mode:
        mode_action.fill(0)
        mode_duration.fill(0)
        mode_steps_to_refresh.fill(0)
        return

    mode_decision = mode_steps_to_refresh <= 0
    if not np.any(mode_decision):
        mode_duration += 1
        mode_steps_to_refresh -= 1
        return

    flat_actor_h = torch.as_tensor(policy_state["actor_h"], dtype=torch.float32, device=device)
    with torch.no_grad():
        mode_logits = model.mode_head(flat_actor_h)
        if deterministic:
            selected_mode = torch.argmax(mode_logits, dim=-1)
        else:
            selected_mode = Categorical(logits=mode_logits).sample()
    selected_mode_np = selected_mode.cpu().numpy().astype(np.int64, copy=False)

    mode_action[:] = np.where(mode_decision, selected_mode_np, mode_action)
    mode_duration[:] = np.where(mode_decision, 1, mode_duration + 1).astype(np.int64, copy=False)
    mode_steps_to_refresh[:] = np.where(
        mode_decision,
        max(int(mode_interval) - 1, 0),
        mode_steps_to_refresh - 1,
    ).astype(np.int64, copy=False)


def select_eval_actions(
    model,
    obs,
    policy_state,
    device,
    deterministic: bool,
    args,
):
    disable_high_level_mode = bool(getattr(args, "disable_high_level_mode", False))
    mode_interval = int(max(1, int(getattr(args, "mode_interval", 8))))
    _update_eval_mode_state(
        model=model,
        policy_state=policy_state,
        device=device,
        deterministic=bool(deterministic),
        disable_high_level_mode=disable_high_level_mode,
        mode_interval=mode_interval,
    )

    stacked = {
        "local_obs": torch.as_tensor(obs["local_obs"][None, ...], dtype=torch.float32, device=device),
        "local_screen": torch.as_tensor(obs["local_screen"][None, ...], dtype=torch.uint8, device=device),
        "attack_masks": torch.as_tensor(obs["attack_masks"][None, ...], dtype=torch.bool, device=device),
        "alive_mask": torch.as_tensor(obs["alive_mask"][None, ...], dtype=torch.float32, device=device),
        "assigned_region_obs": torch.as_tensor(
            obs.get("assigned_region_obs", np.zeros((obs["local_obs"].shape[0], 5), dtype=np.float32))[None, ...],
            dtype=torch.float32,
            device=device,
        ),
        "actor_h": torch.as_tensor(policy_state["actor_h"][None, ...], dtype=torch.float32, device=device),
        "mode_action": torch.as_tensor(policy_state["mode_action"][None, ...], dtype=torch.long, device=device),
        "rule_visible_target_ids": torch.as_tensor(
            obs.get("rule_visible_target_ids", np.zeros((obs["local_obs"].shape[0], 0), dtype=np.int64))[None, ...],
            dtype=torch.long,
            device=device,
        ),
    }

    with torch.no_grad():
        out = sample_actions(model, stacked, device=device, deterministic=bool(deterministic), args=args)

    policy_state["actor_h"][:] = out["next_actor_h"][0].detach().cpu().numpy()
    actions = np.stack(
        [
            out["course_action"][0].detach().cpu().numpy(),
            out["attack_action"][0].detach().cpu().numpy(),
        ],
        axis=-1,
    )
    return actions


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
        render=bool(getattr(args, "eval_maca_render", True)),
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
        boundary_penalty_margin=args.maca_boundary_penalty_margin,
        boundary_penalty_scale=args.maca_boundary_penalty_scale,
        boundary_stuck_penalty_enabled=args.maca_boundary_stuck_penalty_enabled,
        boundary_stuck_trigger_steps=args.maca_boundary_stuck_trigger_steps,
        boundary_stuck_ramp_steps=args.maca_boundary_stuck_ramp_steps,
        search_reward_scale=args.maca_search_reward_scale,
        reacquire_reward_scale=args.maca_reacquire_reward_scale,
        priority_grid_h=args.maca_priority_grid_h,
        priority_grid_w=args.maca_priority_grid_w,
        priority_top_k=args.maca_priority_top_k,
        priority_evidence_weight=args.maca_priority_evidence_weight,
        priority_uncertainty_weight=args.maca_priority_uncertainty_weight,
        priority_diffusion_weight=args.maca_priority_diffusion_weight,
        priority_crowding_penalty=args.maca_priority_crowding_penalty,
        priority_assignment_penalty=args.maca_priority_assignment_penalty,
        priority_distance_penalty=args.maca_priority_distance_penalty,
        priority_unseen_cap_steps=args.maca_priority_unseen_cap_steps,
        priority_memory_decay=args.maca_priority_memory_decay,
        priority_diffusion_rate=args.maca_priority_diffusion_rate,
        priority_passive_recv_weight=args.maca_priority_passive_recv_weight,
        priority_known_enemy_boost=args.maca_priority_known_enemy_boost,
        priority_unseen_threshold=args.maca_priority_unseen_threshold,
        semantic_screen_downsample=args.maca_semantic_screen_downsample,
        terminal_ammo_fail_penalty=args.maca_terminal_ammo_fail_penalty,
        terminal_participation_penalty=args.maca_terminal_participation_penalty,
    )
    env = MAPPOMaCAEnv(eval_config)
    policy_state = init_eval_policy_state(env.num_agents, model.actor_hidden_dim)
    episode_results = []
    start_time = time.time()

    try:
        obs = env.reset(seed=eval_config.random_seed)
        while len(episode_results) < args.eval_episodes:
            actions = select_eval_actions(
                model,
                obs,
                policy_state,
                device,
                args.eval_deterministic,
                args,
            )
            obs, _reward, done, info = env.step(actions)
            if not done:
                continue
            episode_results.append(info["episode_extra_stats"])
            reset_eval_policy_state(policy_state)
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
