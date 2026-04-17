#!/usr/bin/env python
"""Train MaCA with recurrent MAPPO (chunked PPO + burn-in)."""

from __future__ import annotations

import argparse
import sys
import time
from collections import deque
from pathlib import Path

import torch

ROOT_DIR = Path(__file__).resolve().parent.parent
ENV_DIR = ROOT_DIR / "environment"
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(ENV_DIR) not in sys.path:
    sys.path.insert(0, str(ENV_DIR))

from marl_env.mappo_model import MAPPOModelConfig, TeamActorCritic
from marl_train.checkpoint import ValueNormalizer, load_checkpoint, save_checkpoint, save_config
from marl_train.collector import CollectorPool, allocate_shared_actor_params, build_env, write_shared_actor_params
from marl_train.eval import run_evaluation
from marl_train.logging_utils import (
    build_summary_writer,
    log_summary,
    log_train_scalars,
    summarize_episode_stats,
)
from marl_train.ppo_update import action_distribution_stats, ppo_update
from marl_train.rollout import compute_aux_advantages, compute_gae, rollout


def str2bool(value) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=str, default="maca_mappo_recurrent")
    parser.add_argument("--train_dir", type=str, default="train_dir/mappo")
    parser.add_argument("--device", type=str, default="gpu")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--num_envs", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)

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

    parser.add_argument("--mode_interval", type=int, default=8)
    parser.add_argument("--high_level_loss_coeff", type=float, default=0.25)
    parser.add_argument("--high_level_entropy_coeff", type=float, default=0.001)
    parser.add_argument("--teacher_bc_checkpoint", type=str, default="")
    parser.add_argument("--imitation_coef", type=float, default=0.0)
    parser.add_argument("--imitation_warmup_updates", type=int, default=0)

    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--role_embed_dim", type=int, default=8)
    parser.add_argument("--course_embed_dim", type=int, default=16)

    parser.add_argument("--save_every_sec", type=int, default=900)
    parser.add_argument("--log_every_sec", type=int, default=30)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--resume_checkpoint", type=str, default="")
    parser.add_argument("--resume_env_steps", type=int, default=-1)
    parser.add_argument("--tensorboard", type=str2bool, default=True)
    parser.add_argument("--eval_every_env_steps", type=int, default=1_000_000)
    parser.add_argument("--eval_episodes", type=int, default=20)
    parser.add_argument("--eval_deterministic", type=str2bool, default=True)
    parser.add_argument("--eval_opponent", type=str, default=None)

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

    parser.add_argument("--maca_track_memory_steps", type=int, default=12)
    parser.add_argument("--maca_contact_reward", type=float, default=0.1)
    parser.add_argument("--maca_progress_reward_scale", type=float, default=0.002)
    parser.add_argument("--maca_progress_reward_cap", type=float, default=20.0)
    parser.add_argument("--maca_attack_window_reward", type=float, default=0.1)
    parser.add_argument("--maca_agent_aux_reward_scale", type=float, default=0.0)
    parser.add_argument("--maca_mode_reward_scale", type=float, default=0.5)
    parser.add_argument("--maca_exec_reward_scale", type=float, default=0.2)
    parser.add_argument("--maca_disengage_penalty", type=float, default=0.05)
    parser.add_argument("--maca_bearing_reward_scale", type=float, default=0.05)

    parser.add_argument("--concat_agent_id_onehot", type=str2bool, default=True)

    parser.add_argument("--curriculum_enabled", type=str2bool, default=False)
    parser.add_argument("--curriculum_easy_frac", type=float, default=0.3)
    parser.add_argument("--curriculum_medium_frac", type=float, default=0.7)
    parser.add_argument("--curriculum_easy_opponent", type=str, default="fix_rule_no_att")
    parser.add_argument("--curriculum_medium_opponent", type=str, default="fix_rule")
    parser.add_argument("--curriculum_full_opponent", type=str, default="fix_rule")
    parser.add_argument("--curriculum_easy_max_step", type=int, default=600)
    parser.add_argument("--curriculum_medium_max_step", type=int, default=800)
    parser.add_argument("--curriculum_full_max_step", type=int, default=1000)
    parser.add_argument("--curriculum_easy_random_pos", type=str2bool, default=False)
    parser.add_argument("--curriculum_medium_random_pos", type=str2bool, default=False)
    parser.add_argument("--curriculum_full_random_pos", type=str2bool, default=True)

    args = parser.parse_args(argv)
    if args.chunk_len <= 0:
        raise ValueError("chunk_len must be > 0")
    if args.burn_in < 0:
        raise ValueError("burn_in must be >= 0")
    if args.rollout <= 0:
        raise ValueError("rollout must be > 0")
    if args.mode_interval <= 0:
        raise ValueError("mode_interval must be > 0")
    if not (0.0 <= args.curriculum_easy_frac <= 1.0):
        raise ValueError("curriculum_easy_frac must be in [0,1]")
    if not (0.0 <= args.curriculum_medium_frac <= 1.0):
        raise ValueError("curriculum_medium_frac must be in [0,1]")
    if args.curriculum_easy_frac > args.curriculum_medium_frac:
        raise ValueError("curriculum_easy_frac must be <= curriculum_medium_frac")
    return args


def set_seed(seed: int):
    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def device_from_arg(device_arg: str):
    if device_arg == "gpu" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_teacher_policy(teacher_path: str, model, device):
    if not teacher_path:
        return None
    path = Path(teacher_path)
    if not path.exists():
        raise FileNotFoundError("Teacher checkpoint not found: %s" % teacher_path)
    payload = torch.load(path, map_location="cpu")
    teacher_model = TeamActorCritic(model.cfg).to(device)
    state = payload.get("model", payload)
    load_res = teacher_model.load_state_dict(state, strict=False)
    if load_res.missing_keys or load_res.unexpected_keys:
        print(
            "[teacher] non-strict load missing=%s unexpected=%s"
            % (load_res.missing_keys, load_res.unexpected_keys),
            flush=True,
        )
    teacher_model.eval()
    for p in teacher_model.parameters():
        p.requires_grad_(False)
    print("[teacher] loaded %s" % path, flush=True)
    return teacher_model


def curriculum_profile(args, env_steps: int):
    if not bool(args.curriculum_enabled):
        return {
            "name": "full",
            "id": 3,
            "opponent": args.maca_opponent,
            "max_step": int(args.maca_max_step),
            "random_pos": bool(args.maca_random_pos),
        }

    total = max(1, int(args.train_for_env_steps))
    frac = float(env_steps) / float(total)
    if frac < float(args.curriculum_easy_frac):
        return {
            "name": "easy",
            "id": 1,
            "opponent": args.curriculum_easy_opponent,
            "max_step": int(args.curriculum_easy_max_step),
            "random_pos": bool(args.curriculum_easy_random_pos),
        }
    if frac < float(args.curriculum_medium_frac):
        return {
            "name": "medium",
            "id": 2,
            "opponent": args.curriculum_medium_opponent,
            "max_step": int(args.curriculum_medium_max_step),
            "random_pos": bool(args.curriculum_medium_random_pos),
        }
    return {
        "name": "full",
        "id": 3,
        "opponent": args.curriculum_full_opponent,
        "max_step": int(args.curriculum_full_max_step),
        "random_pos": bool(args.curriculum_full_random_pos),
    }


def apply_curriculum_profile(args, profile):
    args.runtime_maca_opponent = str(profile["opponent"])
    args.runtime_maca_max_step = int(profile["max_step"])
    args.runtime_maca_random_pos = bool(profile["random_pos"])
    args.runtime_curriculum_name = str(profile["name"])
    args.runtime_curriculum_id = int(profile["id"])


def main(argv=None):
    args = parse_args(argv)
    device = device_from_arg(args.device)
    set_seed(args.seed)

    stage_profile = curriculum_profile(args, env_steps=0)
    apply_curriculum_profile(args, stage_profile)

    writer = None
    probe_env = build_env(args, seed_offset=0)
    try:
        initial_obs = probe_env.reset(seed=args.seed)
        base_local_obs_dim = initial_obs["local_obs"].shape[1]
        global_state_dim = initial_obs["global_state"].shape[0]
        num_agents = initial_obs["local_obs"].shape[0]
        attack_dim = initial_obs["attack_masks"].shape[1]
    finally:
        probe_env.close()

    local_obs_dim = int(base_local_obs_dim + (num_agents if args.concat_agent_id_onehot else 0))

    env_spec = {
        "local_obs_dim": int(local_obs_dim),
        "global_state_dim": int(global_state_dim),
        "num_agents": int(num_agents),
        "attack_dim": int(attack_dim),
        "actor_hidden_dim": int(args.hidden_size),
        "use_obs_id_onehot": bool(args.concat_agent_id_onehot),
        "mode_interval": int(args.mode_interval),
    }

    model_cfg = MAPPOModelConfig(
        local_obs_dim=local_obs_dim,
        global_state_dim=global_state_dim,
        num_agents=num_agents,
        hidden_size=args.hidden_size,
        role_embed_dim=args.role_embed_dim,
        course_embed_dim=args.course_embed_dim,
    )
    model = TeamActorCritic(model_cfg).to(device)
    teacher_model = None
    if args.teacher_bc_checkpoint:
        teacher_model = load_teacher_policy(args.teacher_bc_checkpoint, model, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    value_normalizer = ValueNormalizer()

    env_steps = 0
    update_idx = 0
    if args.resume:
        env_steps, update_idx = load_checkpoint(args, model, optimizer, value_normalizer, root_dir=ROOT_DIR)

    shared_actor = allocate_shared_actor_params(model)
    write_shared_actor_params(model, shared_actor)

    collectors = CollectorPool(args, env_spec, shared_actor)
    collectors.reset(seed=args.seed)

    save_config(args, local_obs_dim, global_state_dim, num_agents, collectors.worker_env_counts)
    print(
        "[collector] num_workers=%d num_envs=%d worker_env_counts=%s obs_id_onehot=%s base_obs_dim=%d final_obs_dim=%d curriculum_stage=%s"
        % (
            collectors.num_workers,
            collectors.num_envs,
            collectors.worker_env_counts,
            str(bool(args.concat_agent_id_onehot)),
            int(base_local_obs_dim),
            int(local_obs_dim),
            str(getattr(args, "runtime_curriculum_name", "full")),
        ),
        flush=True,
    )

    writer = build_summary_writer(args, purge_step=env_steps if args.resume else None)

    last_log_time = time.time()
    last_save_time = time.time()
    if args.eval_every_env_steps > 0:
        interval = int(args.eval_every_env_steps)
        next_eval_env_steps = ((env_steps // interval) + 1) * interval
    else:
        next_eval_env_steps = -1
    episode_stats = deque(maxlen=100)

    try:
        while env_steps < args.train_for_env_steps:
            next_profile = curriculum_profile(args, env_steps=env_steps)
            if str(next_profile["name"]) != str(getattr(args, "runtime_curriculum_name", "full")):
                apply_curriculum_profile(args, next_profile)
                collectors.close()
                collectors = CollectorPool(args, env_spec, shared_actor)
                collectors.reset(seed=args.seed + env_steps + update_idx * 17)
                print(
                    "[curriculum] switched stage=%s id=%d opponent=%s max_step=%d random_pos=%s"
                    % (
                        args.runtime_curriculum_name,
                        int(args.runtime_curriculum_id),
                        args.runtime_maca_opponent,
                        int(args.runtime_maca_max_step),
                        str(bool(args.runtime_maca_random_pos)),
                    ),
                    flush=True,
                )

            update_start = time.time()
            print(
                "[stage] update=%d env_steps=%d starting_rollout rollout=%d num_envs=%d workers=%d"
                % (update_idx + 1, env_steps, args.rollout, collectors.num_envs, collectors.num_workers),
                flush=True,
            )

            actor_sync_start = time.time()
            write_shared_actor_params(model, shared_actor)
            actor_sync_time = time.time() - actor_sync_start

            buffer, next_value, rollout_diag, rollout_timing = rollout(
                collectors, model, device, args, episode_stats, value_normalizer
            )
            rollout_wall_time = time.time() - update_start
            print(
                "[stage] update=%d env_steps=%d rollout_finished wall_time_sec=%.2f"
                % (update_idx + 1, env_steps, rollout_wall_time),
                flush=True,
            )

            gae_start = time.time()
            advantages_team, returns = compute_gae(buffer, next_value, args.gamma, args.gae_lambda)
            advantages_aux = compute_aux_advantages(buffer, args.gamma)
            gae_time = time.time() - gae_start

            value_normalizer.update(returns)
            print(
                "[stage] update=%d env_steps=%d starting_ppo_update"
                % (update_idx + 1, env_steps),
                flush=True,
            )

            ppo_start = time.time()
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
                teacher_model=teacher_model,
                update_idx=update_idx,
            )
            ppo_update_time = time.time() - ppo_start
            total_update_time = time.time() - update_start

            timing_stats = {
                "actor_sync_time": float(actor_sync_time),
                "rollout_collect_time": float(rollout_timing.get("rollout_collect_time", 0.0)),
                "rollout_postprocess_time": float(rollout_timing.get("rollout_postprocess_time", 0.0)),
                "value_bootstrap_time": float(rollout_timing.get("value_bootstrap_time", 0.0)),
                "gae_time": float(gae_time),
                "ppo_update_time": float(ppo_update_time),
                "total_update_time": float(total_update_time),
            }

            action_stats = action_distribution_stats(
                course_actions=buffer["course_action"],
                attack_actions=buffer["attack_action"],
                alive_mask=buffer["alive_mask"],
                attack_masks=buffer["attack_masks"].astype(bool, copy=False),
                course_dim=model_cfg.course_dim,
                attack_dim=model_cfg.attack_dim,
            )
            print(
                "[stage] update=%d env_steps=%d ppo_update_finished"
                % (update_idx + 1, env_steps),
                flush=True,
            )

            env_steps += args.rollout * collectors.num_envs
            update_idx += 1
            sample_fps = (args.rollout * collectors.num_envs) / max(total_update_time, 1e-6)

            print(
                "[timing] update=%d env_steps=%d actor_sync=%.4fs collect=%.4fs post=%.4fs bootstrap=%.4fs gae=%.4fs ppo=%.4fs total=%.4fs"
                % (
                    update_idx,
                    env_steps,
                    timing_stats["actor_sync_time"],
                    timing_stats["rollout_collect_time"],
                    timing_stats["rollout_postprocess_time"],
                    timing_stats["value_bootstrap_time"],
                    timing_stats["gae_time"],
                    timing_stats["ppo_update_time"],
                    timing_stats["total_update_time"],
                ),
                flush=True,
            )

            if time.time() - last_log_time >= args.log_every_sec:
                summary = summarize_episode_stats(list(episode_stats))
                reward_mean = summary.get("round_reward_mean", 0.0)
                win_rate = summary.get("win_rate", 0.0)
                log_summary(writer, "train_episode", summary, env_steps)
                log_train_scalars(
                    writer,
                    env_steps=env_steps,
                    stats=stats,
                    rollout_diag=rollout_diag,
                    action_stats=action_stats,
                    sample_fps=sample_fps,
                    timing_stats=timing_stats,
                    concat_agent_id_onehot=bool(args.concat_agent_id_onehot),
                    stage_id=int(getattr(args, "runtime_curriculum_id", 3)),
                )
                print(
                    "[train] env_steps=%d update=%d stage=%s reward_mean=%.2f win_rate=%.3f fps=%.1f policy=%.4f low=%.4f high=%.4f value_loss=%.4f entropy=%.4f imitation=%.4f coef=%.4f ev=%.4f actor_gn=%.4f critic_gn=%.4f active=%.3f skipped_nf=%d repaired_nf=%d hidden_err=%.6f rnn_mismatch=%d grad_steps=%d"
                    % (
                        env_steps,
                        update_idx,
                        str(getattr(args, "runtime_curriculum_name", "full")),
                        reward_mean,
                        win_rate,
                        sample_fps,
                        stats.get("policy_loss", 0.0),
                        stats.get("policy_loss_low", 0.0),
                        stats.get("policy_loss_high", 0.0),
                        stats.get("value_loss", 0.0),
                        stats.get("entropy", 0.0),
                        stats.get("imitation_loss", 0.0),
                        stats.get("imitation_coef", 0.0),
                        stats.get("explained_variance", 0.0),
                        stats.get("actor_grad_norm", 0.0),
                        stats.get("critic_grad_norm", 0.0),
                        stats.get("active_mask_ratio", 0.0),
                        int(stats.get("skipped_non_finite_batches", 0)),
                        int(stats.get("repaired_non_finite_params", 0)),
                        stats.get("hidden_state_init_abs_error", 0.0),
                        int(rollout_diag.get("rnn_hidden_mismatch_count", 0)),
                        int(stats.get("grad_steps", 0)),
                    ),
                    flush=True,
                )
                last_log_time = time.time()

            if time.time() - last_save_time >= args.save_every_sec:
                save_checkpoint(args, model, optimizer, env_steps, update_idx, value_normalizer)
                last_save_time = time.time()

            if args.eval_episodes > 0 and args.eval_every_env_steps > 0 and env_steps >= next_eval_env_steps:
                payload, output_path = run_evaluation(model, device, args, env_steps)
                eval_summary = payload.get("summary", {})
                log_summary(writer, "eval", eval_summary, env_steps)
                if writer is not None:
                    writer.add_scalar("eval/eval_wall_time_sec", float(payload["eval_wall_time_sec"]), env_steps)
                    writer.flush()
                print(
                    "[eval] env_steps=%d win_rate=%.3f destroy_balance=%.3f red_down=%.3f blue_down=%.3f saved=%s"
                    % (
                        env_steps,
                        float(eval_summary.get("win_rate", 0.0)),
                        float(eval_summary.get("fighter_destroy_balance_end_mean", 0.0)),
                        float(eval_summary.get("red_fighter_destroyed_end_mean", 0.0)),
                        float(eval_summary.get("blue_fighter_destroyed_end_mean", 0.0)),
                        output_path,
                    ),
                    flush=True,
                )
                next_eval_env_steps += int(args.eval_every_env_steps)

        save_checkpoint(args, model, optimizer, env_steps, update_idx, value_normalizer)
    finally:
        collectors.close()
        if writer is not None:
            writer.close()


if __name__ == "__main__":
    main()
