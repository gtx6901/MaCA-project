from __future__ import annotations

from typing import Dict, Optional

import numpy as np
from torch.utils.tensorboard import SummaryWriter

from marl_train.checkpoint import experiment_dir


def build_summary_writer(args, purge_step: Optional[int] = None):
    if not args.tensorboard:
        return None
    log_dir = experiment_dir(args) / "tb"
    log_dir.mkdir(parents=True, exist_ok=True)
    if purge_step is not None and int(purge_step) >= 0:
        print("[tensorboard] log_dir=%s purge_step=%d" % (log_dir, int(purge_step)), flush=True)
        return SummaryWriter(log_dir=str(log_dir), purge_step=int(purge_step))
    return SummaryWriter(log_dir=str(log_dir))


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


def log_summary(writer: Optional[SummaryWriter], prefix: str, summary: Dict[str, float], step: int) -> None:
    if writer is None:
        return
    for key, value in summary.items():
        if key == "episodes":
            continue
        try:
            writer.add_scalar("%s/%s" % (prefix, key), float(value), step)
        except Exception:
            continue


def log_train_scalars(
    writer: Optional[SummaryWriter],
    *,
    env_steps: int,
    stats: Dict[str, float],
    rollout_diag: Dict[str, float],
    action_stats: Dict[str, float],
    sample_fps: float,
    timing_stats: Dict[str, float],
    concat_agent_id_onehot: bool,
    stage_id: int,
    current_lr: float,
    reward_mean: float,
    win_rate: float,
    attack_rule_mode: str,
    attack_policy_mode: str,
    disable_high_level_mode: bool,
):
    if writer is None:
        return

    writer.add_scalar("train/policy_loss", float(stats.get("policy_loss", 0.0)), env_steps)
    writer.add_scalar("train/policy_loss_low", float(stats.get("policy_loss_low", 0.0)), env_steps)
    writer.add_scalar("train/policy_loss_high", float(stats.get("policy_loss_high", 0.0)), env_steps)
    writer.add_scalar("train/value_loss", float(stats.get("value_loss", 0.0)), env_steps)
    writer.add_scalar("train/entropy", float(stats.get("entropy", 0.0)), env_steps)
    writer.add_scalar("train/entropy_low", float(stats.get("entropy_low", 0.0)), env_steps)
    writer.add_scalar("train/entropy_high", float(stats.get("entropy_high", 0.0)), env_steps)
    writer.add_scalar("train/imitation_loss", float(stats.get("imitation_loss", 0.0)), env_steps)
    writer.add_scalar("train/imitation_coef", float(stats.get("imitation_coef", 0.0)), env_steps)
    writer.add_scalar("train/explained_variance", float(stats.get("explained_variance", 0.0)), env_steps)
    writer.add_scalar("train/actor_grad_norm", float(stats.get("actor_grad_norm", 0.0)), env_steps)
    writer.add_scalar("train/critic_grad_norm", float(stats.get("critic_grad_norm", 0.0)), env_steps)
    writer.add_scalar("train/active_mask_ratio", float(stats.get("active_mask_ratio", 0.0)), env_steps)
    writer.add_scalar(
        "train/skipped_non_finite_batches",
        float(stats.get("skipped_non_finite_batches", 0.0)),
        env_steps,
    )
    writer.add_scalar(
        "train/repaired_non_finite_params",
        float(stats.get("repaired_non_finite_params", 0.0)),
        env_steps,
    )
    writer.add_scalar(
        "train/hidden_state_init_abs_error",
        float(stats.get("hidden_state_init_abs_error", 0.0)),
        env_steps,
    )
    writer.add_scalar("train/value_target_mean", float(stats.get("value_target_mean", 0.0)), env_steps)
    writer.add_scalar("train/value_target_std", float(stats.get("value_target_std", 0.0)), env_steps)
    writer.add_scalar("train/value_target_norm_mean", float(stats.get("value_target_norm_mean", 0.0)), env_steps)
    writer.add_scalar("train/value_target_norm_std", float(stats.get("value_target_norm_std", 0.0)), env_steps)

    writer.add_scalar("train/damage_reward_mean", float(rollout_diag.get("damage_reward_mean", 0.0)), env_steps)
    writer.add_scalar("train/reward_env_mean", float(rollout_diag.get("reward_env_mean", 0.0)), env_steps)
    writer.add_scalar("train/reward_mode_mean", float(rollout_diag.get("reward_mode_mean", 0.0)), env_steps)
    writer.add_scalar("train/reward_exec_mean", float(rollout_diag.get("reward_exec_mean", 0.0)), env_steps)
    writer.add_scalar("train/kill_reward_mean", float(rollout_diag.get("kill_reward_mean", 0.0)), env_steps)
    writer.add_scalar("train/survival_reward_mean", float(rollout_diag.get("survival_reward_mean", 0.0)), env_steps)
    writer.add_scalar("train/win_indicator_mean", float(rollout_diag.get("win_indicator_mean", 0.0)), env_steps)
    writer.add_scalar("train/value_contact_mean", float(rollout_diag.get("value_contact_mean", 0.0)), env_steps)
    writer.add_scalar("train/value_opportunity_mean", float(rollout_diag.get("value_opportunity_mean", 0.0)), env_steps)
    writer.add_scalar("train/value_survival_mean", float(rollout_diag.get("value_survival_mean", 0.0)), env_steps)
    writer.add_scalar("train/rnn_hidden_mismatch_count", float(rollout_diag.get("rnn_hidden_mismatch_count", 0)), env_steps)
    writer.add_scalar("train/rnn_hidden_max_abs_diff", float(rollout_diag.get("rnn_hidden_max_abs_diff", 0.0)), env_steps)

    writer.add_scalar("train/obs_agent_id_concat_enabled", 1.0 if concat_agent_id_onehot else 0.0, env_steps)
    for key, value in action_stats.items():
        writer.add_scalar("train_action/%s" % key, float(value), env_steps)

    writer.add_scalar("train/sample_fps", float(sample_fps), env_steps)
    writer.add_scalar("timing/actor_sync_time", float(timing_stats.get("actor_sync_time", 0.0)), env_steps)
    writer.add_scalar("timing/rollout_collect_time", float(timing_stats.get("rollout_collect_time", 0.0)), env_steps)
    writer.add_scalar("timing/rollout_postprocess_time", float(timing_stats.get("rollout_postprocess_time", 0.0)), env_steps)
    writer.add_scalar("timing/value_bootstrap_time", float(timing_stats.get("value_bootstrap_time", 0.0)), env_steps)
    writer.add_scalar("timing/gae_time", float(timing_stats.get("gae_time", 0.0)), env_steps)
    writer.add_scalar("timing/ppo_update_time", float(timing_stats.get("ppo_update_time", 0.0)), env_steps)
    writer.add_scalar("timing/total_update_time", float(timing_stats.get("total_update_time", 0.0)), env_steps)

    writer.add_scalar("train/active_samples", float(stats.get("active_samples", 0)), env_steps)
    writer.add_scalar("train/grad_steps", float(stats.get("grad_steps", 0)), env_steps)
    writer.add_scalar("train/current_lr", float(current_lr), env_steps)
    writer.add_scalar("train/reward_mean", float(reward_mean), env_steps)
    writer.add_scalar("train/win_rate", float(win_rate), env_steps)
    writer.add_scalar("train/attack_rule_mode_none", 1.0 if str(attack_rule_mode).lower() == "none" else 0.0, env_steps)
    writer.add_scalar(
        "train/attack_rule_mode_nearest_target",
        1.0 if str(attack_rule_mode).lower() == "nearest_target" else 0.0,
        env_steps,
    )
    writer.add_scalar(
        "train/attack_policy_mode_full_discrete",
        1.0 if str(attack_policy_mode).lower() == "full_discrete" else 0.0,
        env_steps,
    )
    writer.add_scalar(
        "train/attack_policy_mode_fire_or_not",
        1.0 if str(attack_policy_mode).lower() == "fire_or_not" else 0.0,
        env_steps,
    )
    writer.add_scalar("train/disable_high_level_mode", 1.0 if disable_high_level_mode else 0.0, env_steps)
    writer.add_scalar("curriculum/stage_id", float(stage_id), env_steps)
    writer.flush()
