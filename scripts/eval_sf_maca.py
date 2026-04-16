#!/usr/bin/env python
"""Evaluate a Sample Factory MaCA checkpoint against a fixed opponent."""

from __future__ import annotations

import json
import math
import sys
import time
from collections import OrderedDict

import numpy as np
import torch

from fighter_action_utils import ATTACK_IND_NUM
from sample_factory.algorithms.appo import model as appo_model
from sample_factory.algorithms.appo.actor_worker import transform_dict_observations
from sample_factory.algorithms.appo.learner import LearnerWorker
from sample_factory.algorithms.appo.model import create_actor_critic
from sample_factory.algorithms.appo.model_utils import get_hidden_size, normalize_obs
from sample_factory.algorithms.utils.action_distributions import get_action_distribution, sample_actions_log_probs
from sample_factory.algorithms.utils.arguments import parse_args, load_from_checkpoint
from sample_factory.algorithms.utils.multi_agent_wrapper import MultiAgentWrapper, is_multiagent_env
from sample_factory.envs.create_env import create_env
from sample_factory.utils.utils import AttrDict

from marl_env.sample_factory_registration import register_maca_components
from marl_env.runtime_tweaks import (
    format_runtime_tweaks,
    get_attack_prior_strength,
    get_eval_fire_prob_floor,
    load_runtime_tweaks,
)


def _patch_sample_factory_action_masking():
    """Apply MaCA action masks to policy logits during evaluation."""

    invalid_logit = -1e9
    attack_prior_strength = max(0.0, get_attack_prior_strength())
    eval_fire_prob_floor = max(0.0, min(0.95, get_eval_fire_prob_floor()))

    def tuple_action_head_sizes(actor_critic):
        if not hasattr(actor_critic.action_space, "spaces"):
            return None
        spaces = getattr(actor_critic.action_space, "spaces", None)
        if spaces is None:
            return None
        return [space.n for space in spaces]

    def stash_action_context(actor_critic, obs_dict):
        action_mask = obs_dict.get("action_mask")
        actor_critic._last_action_mask = None if action_mask is None else action_mask.bool()
        actor_critic._last_course_prior = obs_dict.get("course_prior")
        actor_critic._last_attack_prior = obs_dict.get("attack_prior")

    def mask_action_logits(actor_critic, action_logits):
        action_mask = getattr(actor_critic, "_last_action_mask", None)
        actor_critic._last_action_mask = None
        course_prior = getattr(actor_critic, "_last_course_prior", None)
        actor_critic._last_course_prior = None
        attack_prior = getattr(actor_critic, "_last_attack_prior", None)
        actor_critic._last_attack_prior = None
        if action_mask is None:
            return action_logits

        action_mask = action_mask.to(device=action_logits.device)
        if action_mask.dtype != torch.bool:
            action_mask = action_mask > 0
        if action_mask.shape != action_logits.shape:
            return action_logits

        has_valid_action = action_mask.any(dim=-1, keepdim=True)
        if not torch.all(has_valid_action):
            action_mask = torch.where(has_valid_action, action_mask, torch.ones_like(action_mask))

        head_sizes = tuple_action_head_sizes(actor_critic)
        if head_sizes is not None:
            split_logits = list(torch.split(action_logits, head_sizes, dim=-1))
            split_masks = list(torch.split(action_mask, head_sizes, dim=-1))
            split_logits = [
                logits.masked_fill(~mask, invalid_logit) for logits, mask in zip(split_logits, split_masks)
            ]
            if len(split_logits) >= 2 and course_prior is not None:
                course_logits = split_logits[0]
                course_mask = split_masks[0]
                course_prior_strength = max(0.0, float(getattr(actor_critic.cfg, "maca_course_prior_strength", 0.0)))
                if course_prior_strength > 0.0:
                    course_prior = course_prior.to(device=course_logits.device, dtype=course_logits.dtype)
                    if course_prior.shape == course_logits.shape:
                        course_logits = course_logits + course_prior_strength * course_prior
                        course_logits = course_logits.masked_fill(~course_mask, invalid_logit)
                        split_logits[0] = course_logits

            attack_logits = split_logits[-1]
            attack_mask = split_masks[-1]
            attack_prior_weight = max(attack_prior_strength, float(getattr(actor_critic.cfg, "maca_attack_prior_strength", 0.0)))
            if attack_prior_weight > 0.0 and attack_prior is not None:
                attack_prior = attack_prior.to(device=attack_logits.device, dtype=attack_logits.dtype)
                if attack_prior.shape == attack_logits.shape:
                    attack_logits = attack_logits + attack_prior_weight * attack_prior
                    attack_logits = attack_logits.masked_fill(~attack_mask, invalid_logit)
            if eval_fire_prob_floor > 0.0 and attack_logits.shape[-1] > 1:
                valid_fire = attack_mask.clone()
                valid_fire[..., 0] = False
                has_attack_opportunity = valid_fire.any(dim=-1, keepdim=True)
                if torch.any(has_attack_opportunity):
                    valid_non_fire = attack_mask.clone()
                    valid_non_fire[..., 1:] = False
                    logsum_fire = torch.logsumexp(attack_logits.masked_fill(~valid_fire, invalid_logit), dim=-1)
                    logsum_non_fire = torch.logsumexp(
                        attack_logits.masked_fill(~valid_non_fire, invalid_logit), dim=-1
                    )
                    target_logit = math.log(eval_fire_prob_floor / max(1e-8, 1.0 - eval_fire_prob_floor))
                    delta = torch.clamp(target_logit + logsum_non_fire - logsum_fire, min=0.0, max=20.0)
                    rows = has_attack_opportunity.squeeze(-1) & torch.isfinite(delta)
                    if torch.any(rows):
                        attack_logits = attack_logits + valid_fire.to(attack_logits.dtype) * delta.unsqueeze(-1)
            split_logits[-1] = attack_logits
            return torch.cat(split_logits, dim=-1)

        action_logits = action_logits.masked_fill(~action_mask, invalid_logit)
        if eval_fire_prob_floor > 0.0 and action_logits.shape[-1] % ATTACK_IND_NUM == 0 and action_logits.shape[-1] > 1:
            action_indices = torch.arange(action_logits.shape[-1], device=action_logits.device)
            fire_positions = (action_indices % ATTACK_IND_NUM) > 0
            valid_fire = action_mask & fire_positions.unsqueeze(0)
            has_attack_opportunity = valid_fire.any(dim=-1, keepdim=True)
            if torch.any(has_attack_opportunity):
                valid_fire_with_opp = valid_fire & has_attack_opportunity
                valid_non_fire = action_mask & (~fire_positions).unsqueeze(0)
                logsum_fire = torch.logsumexp(action_logits.masked_fill(~valid_fire, invalid_logit), dim=-1)
                logsum_non_fire = torch.logsumexp(action_logits.masked_fill(~valid_non_fire, invalid_logit), dim=-1)
                target_logit = math.log(eval_fire_prob_floor / max(1e-8, 1.0 - eval_fire_prob_floor))
                delta = torch.clamp(target_logit + logsum_non_fire - logsum_fire, min=0.0, max=20.0)
                rows = has_attack_opportunity.squeeze(-1) & torch.isfinite(delta)
                if torch.any(rows):
                    action_logits = action_logits + valid_fire_with_opp.to(action_logits.dtype) * delta.unsqueeze(-1)
        return action_logits

    def shared_forward_head(self, obs_dict):
        normalize_obs(obs_dict, self.cfg)
        stash_action_context(self, obs_dict)
        return self.encoder(obs_dict)

    def shared_forward_tail(self, core_output, with_action_distribution=False):
        values = self.critic_linear(core_output)
        action_distribution_params, _ = self.action_parameterization(core_output)
        action_distribution_params = mask_action_logits(self, action_distribution_params)
        action_distribution = get_action_distribution(self.action_space, raw_logits=action_distribution_params)
        actions, log_prob_actions = sample_actions_log_probs(action_distribution)

        result = AttrDict(
            dict(
                actions=actions,
                action_logits=action_distribution_params,
                log_prob_actions=log_prob_actions,
                values=values,
            )
        )

        if with_action_distribution:
            result.action_distribution = action_distribution

        return result

    def separate_forward_head(self, obs_dict):
        normalize_obs(obs_dict, self.cfg)
        stash_action_context(self, obs_dict)
        head_outputs = []
        for encoder in self.encoders:
            head_outputs.append(encoder(obs_dict))
        return torch.cat(head_outputs, dim=1)

    def separate_forward_tail(self, core_output, with_action_distribution=False):
        core_outputs = core_output.chunk(len(self.cores), dim=1)
        action_distribution_params, _ = self.action_parameterization(core_outputs[0])
        action_distribution_params = mask_action_logits(self, action_distribution_params)
        action_distribution = get_action_distribution(self.action_space, raw_logits=action_distribution_params)
        actions, log_prob_actions = sample_actions_log_probs(action_distribution)
        values = self.critic_linear(core_outputs[1])

        result = AttrDict(
            dict(
                actions=actions,
                action_logits=action_distribution_params,
                log_prob_actions=log_prob_actions,
                values=values,
            )
        )

        if with_action_distribution:
            result.action_distribution = action_distribution

        return result

    appo_model._ActorCriticSharedWeights.forward_head = shared_forward_head
    appo_model._ActorCriticSharedWeights.forward_tail = shared_forward_tail
    appo_model._ActorCriticSeparateWeights.forward_head = separate_forward_head
    appo_model._ActorCriticSeparateWeights.forward_tail = separate_forward_tail


def _inject_eval_args(argv):
    result = list(argv)
    if not any(arg == "--algo" or arg.startswith("--algo=") for arg in result):
        result.append("--algo=APPO")
    if not any(arg == "--env" or arg.startswith("--env=") for arg in result):
        result.append("--env=maca_aircombat")
    return result


def _parse_cli(argv):
    episodes = 20
    output_json = None
    progress = True
    passthrough = []

    idx = 0
    while idx < len(argv):
        arg = argv[idx]
        if arg == "--episodes":
            episodes = int(argv[idx + 1])
            idx += 2
            continue
        if arg.startswith("--episodes="):
            episodes = int(arg.split("=", 1)[1])
            idx += 1
            continue
        if arg == "--output_json":
            output_json = argv[idx + 1]
            idx += 2
            continue
        if arg.startswith("--output_json="):
            output_json = arg.split("=", 1)[1]
            idx += 1
            continue
        if arg == "--progress":
            value = argv[idx + 1].strip().lower()
            progress = value not in {"0", "false", "no", "off"}
            idx += 2
            continue
        if arg.startswith("--progress="):
            value = arg.split("=", 1)[1].strip().lower()
            progress = value not in {"0", "false", "no", "off"}
            idx += 1
            continue
        passthrough.append(arg)
        idx += 1

    return episodes, output_json, progress, passthrough


def main(argv=None):
    load_runtime_tweaks()
    print(f"[maca_runtime] {format_runtime_tweaks()}", flush=True)
    register_maca_components()
    _patch_sample_factory_action_masking()

    raw_argv = sys.argv[1:] if argv is None else argv
    episodes, output_json, progress, sf_argv = _parse_cli(raw_argv)
    cfg = parse_args(argv=_inject_eval_args(sf_argv), evaluation=True)
    cfg = load_from_checkpoint(cfg)
    cfg.env_frameskip = 1
    cfg.num_envs = 1
    cfg.no_render = True

    env = create_env(cfg.env, cfg=cfg, env_config=AttrDict({"worker_index": 0, "vector_index": 0}))
    if not is_multiagent_env(env):
        env = MultiAgentWrapper(env)

    device = torch.device("cpu" if cfg.device == "cpu" else "cuda")
    actor_critic = create_actor_critic(cfg, env.observation_space, env.action_space)
    actor_critic.model_to_device(device)
    actor_critic.eval()

    checkpoints = LearnerWorker.get_checkpoints(LearnerWorker.checkpoint_dir(cfg, cfg.policy_index))
    checkpoint_dict = LearnerWorker.load_checkpoint(checkpoints, device)
    actor_critic.load_state_dict(checkpoint_dict["model"])

    obs = env.reset()
    rnn_states = torch.zeros((env.num_agents, get_hidden_size(cfg)), dtype=torch.float32, device=device)

    episode_results = []
    eval_start_time = time.time()

    with torch.no_grad():
        while len(episode_results) < episodes:
            obs_torch = AttrDict(transform_dict_observations(obs))
            for key, value in obs_torch.items():
                obs_torch[key] = torch.from_numpy(value).to(device).float()

            policy_outputs = actor_critic(obs_torch, rnn_states, with_action_distribution=True)
            actions = policy_outputs.actions.cpu().numpy()
            rnn_states = policy_outputs.rnn_states

            obs, rewards, done, infos = env.step(actions)

            if all(done):
                extra_stats = [info.get("episode_extra_stats", {}) for info in infos]
                valid_stats = [stat for stat in extra_stats if stat]
                true_rewards = [float(info.get("true_reward", 0.0)) for info in infos]

                round_reward = float(np.mean([stat.get("round_reward", 0.0) for stat in valid_stats]))
                opponent_round_reward = float(
                    np.mean([stat.get("opponent_round_reward", 0.0) for stat in valid_stats])
                )
                invalid_action_frac = float(
                    np.mean([stat.get("invalid_action_frac", 0.0) for stat in valid_stats])
                )
                fire_action_frac = float(np.mean([stat.get("fire_action_frac", 0.0) for stat in valid_stats]))
                executed_fire_action_frac = float(
                    np.mean([stat.get("executed_fire_action_frac", 0.0) for stat in valid_stats])
                )
                attack_opportunity_frac = float(
                    np.mean([stat.get("attack_opportunity_frac", 0.0) for stat in valid_stats])
                )
                missed_attack_frac = float(np.mean([stat.get("missed_attack_frac", 0.0) for stat in valid_stats]))
                course_change_frac = float(np.mean([stat.get("course_change_frac", 0.0) for stat in valid_stats]))
                course_unique_frac = float(np.mean([stat.get("course_unique_frac", 0.0) for stat in valid_stats]))
                visible_enemy_count_mean = float(
                    np.mean([stat.get("visible_enemy_count_mean", 0.0) for stat in valid_stats])
                )
                contact_frac = float(np.mean([stat.get("contact_frac", 0.0) for stat in valid_stats]))
                attack_window_entry_frac = float(
                    np.mean([stat.get("attack_window_entry_frac", 0.0) for stat in valid_stats])
                )
                nearest_enemy_distance_mean = float(
                    np.mean([stat.get("nearest_enemy_distance_mean", 0.0) for stat in valid_stats])
                )
                nearest_enemy_distance_min = float(
                    np.mean([stat.get("nearest_enemy_distance_min", 0.0) for stat in valid_stats])
                )
                engagement_progress_reward_mean = float(
                    np.mean([stat.get("engagement_progress_reward_mean", 0.0) for stat in valid_stats])
                )
                episode_len = float(np.mean([stat.get("episode_len", 0.0) for stat in valid_stats]))
                win_flag = float(np.mean([stat.get("win_flag", 0.0) for stat in valid_stats]) > 0.5)
                red_fighter_alive_end = float(np.mean([stat.get("red_fighter_alive_end", 0.0) for stat in valid_stats]))
                red_fighter_destroyed_end = float(
                    np.mean([stat.get("red_fighter_destroyed_end", 0.0) for stat in valid_stats])
                )
                blue_fighter_alive_end = float(
                    np.mean([stat.get("blue_fighter_alive_end", 0.0) for stat in valid_stats])
                )
                blue_fighter_destroyed_end = float(
                    np.mean([stat.get("blue_fighter_destroyed_end", 0.0) for stat in valid_stats])
                )
                fighter_destroy_balance_end = float(
                    np.mean([stat.get("fighter_destroy_balance_end", 0.0) for stat in valid_stats])
                )

                episode_results.append(
                    {
                        "episode": len(episode_results) + 1,
                        "win": int(win_flag),
                        "round_reward": round_reward,
                        "opponent_round_reward": opponent_round_reward,
                        "true_reward_mean": float(np.mean(true_rewards)),
                        "invalid_action_frac_mean": invalid_action_frac,
                        "fire_action_frac_mean": fire_action_frac,
                        "executed_fire_action_frac_mean": executed_fire_action_frac,
                        "attack_opportunity_frac_mean": attack_opportunity_frac,
                        "missed_attack_frac_mean": missed_attack_frac,
                        "course_change_frac_mean": course_change_frac,
                        "course_unique_frac_mean": course_unique_frac,
                        "visible_enemy_count_mean": visible_enemy_count_mean,
                        "contact_frac_mean": contact_frac,
                        "attack_window_entry_frac_mean": attack_window_entry_frac,
                        "nearest_enemy_distance_mean": nearest_enemy_distance_mean,
                        "nearest_enemy_distance_min": nearest_enemy_distance_min,
                        "engagement_progress_reward_mean": engagement_progress_reward_mean,
                        "episode_len_mean": episode_len,
                        "red_fighter_alive_end_mean": red_fighter_alive_end,
                        "red_fighter_destroyed_end_mean": red_fighter_destroyed_end,
                        "blue_fighter_alive_end_mean": blue_fighter_alive_end,
                        "blue_fighter_destroyed_end_mean": blue_fighter_destroyed_end,
                        "fighter_destroy_balance_end_mean": fighter_destroy_balance_end,
                    }
                )

                if progress:
                    latest = episode_results[-1]
                    elapsed = time.time() - eval_start_time
                    print(
                        (
                            f"[eval] episode {latest['episode']}/{episodes} "
                            f"win={latest['win']} "
                            f"len={latest['episode_len_mean']:.0f} "
                            f"round={latest['round_reward']:.1f} "
                            f"true={latest['true_reward_mean']:.1f} "
                            f"invalid={latest['invalid_action_frac_mean']:.4f} "
                            f"fire={latest.get('fire_action_frac_mean', 0.0):.4f} "
                            f"fire_exec={latest.get('executed_fire_action_frac_mean', 0.0):.4f} "
                            f"opp={latest.get('attack_opportunity_frac_mean', 0.0):.4f} "
                            f"missed={latest.get('missed_attack_frac_mean', 0.0):.4f} "
                            f"course_change={latest.get('course_change_frac_mean', 0.0):.3f} "
                            f"course_unique={latest.get('course_unique_frac_mean', 0.0):.3f} "
                            f"contact={latest.get('contact_frac_mean', 0.0):.3f} "
                            f"visible_mean={latest.get('visible_enemy_count_mean', 0.0):.3f} "
                            f"nearest_mean={latest.get('nearest_enemy_distance_mean', 0.0):.1f} "
                            f"red_down={latest.get('red_fighter_destroyed_end_mean', 0.0):.1f} "
                            f"blue_down={latest.get('blue_fighter_destroyed_end_mean', 0.0):.1f} "
                            f"engage_reward={latest.get('engagement_progress_reward_mean', 0.0):.2f} "
                            f"elapsed={elapsed:.1f}s"
                        ),
                        flush=True,
                    )

                rnn_states.zero_()

    env.close()

    summary = OrderedDict(
        episodes=len(episode_results),
        win_rate=float(np.mean([row["win"] for row in episode_results])) if episode_results else 0.0,
        round_reward_mean=float(np.mean([row["round_reward"] for row in episode_results])) if episode_results else 0.0,
        opponent_round_reward_mean=float(
            np.mean([row["opponent_round_reward"] for row in episode_results])
        )
        if episode_results
        else 0.0,
        true_reward_mean=float(np.mean([row["true_reward_mean"] for row in episode_results]))
        if episode_results
        else 0.0,
        invalid_action_frac_mean=float(
            np.mean([row["invalid_action_frac_mean"] for row in episode_results])
        )
        if episode_results
        else 0.0,
        fire_action_frac_mean=float(np.mean([row["fire_action_frac_mean"] for row in episode_results]))
        if episode_results
        else 0.0,
        executed_fire_action_frac_mean=float(
            np.mean([row["executed_fire_action_frac_mean"] for row in episode_results])
        )
        if episode_results
        else 0.0,
        attack_opportunity_frac_mean=float(
            np.mean([row["attack_opportunity_frac_mean"] for row in episode_results])
        )
        if episode_results
        else 0.0,
        missed_attack_frac_mean=float(np.mean([row["missed_attack_frac_mean"] for row in episode_results]))
        if episode_results
        else 0.0,
        course_change_frac_mean=float(np.mean([row["course_change_frac_mean"] for row in episode_results]))
        if episode_results
        else 0.0,
        course_unique_frac_mean=float(np.mean([row["course_unique_frac_mean"] for row in episode_results]))
        if episode_results
        else 0.0,
        visible_enemy_count_mean=float(np.mean([row["visible_enemy_count_mean"] for row in episode_results]))
        if episode_results
        else 0.0,
        contact_frac_mean=float(np.mean([row["contact_frac_mean"] for row in episode_results]))
        if episode_results
        else 0.0,
        attack_window_entry_frac_mean=float(
            np.mean([row["attack_window_entry_frac_mean"] for row in episode_results])
        )
        if episode_results
        else 0.0,
        nearest_enemy_distance_mean=float(
            np.mean([row["nearest_enemy_distance_mean"] for row in episode_results])
        )
        if episode_results
        else 0.0,
        nearest_enemy_distance_min=float(np.mean([row["nearest_enemy_distance_min"] for row in episode_results]))
        if episode_results
        else 0.0,
        engagement_progress_reward_mean=float(
            np.mean([row["engagement_progress_reward_mean"] for row in episode_results])
        )
        if episode_results
        else 0.0,
        red_fighter_alive_end_mean=float(np.mean([row["red_fighter_alive_end_mean"] for row in episode_results]))
        if episode_results
        else 0.0,
        red_fighter_destroyed_end_mean=float(
            np.mean([row["red_fighter_destroyed_end_mean"] for row in episode_results])
        )
        if episode_results
        else 0.0,
        blue_fighter_alive_end_mean=float(np.mean([row["blue_fighter_alive_end_mean"] for row in episode_results]))
        if episode_results
        else 0.0,
        blue_fighter_destroyed_end_mean=float(
            np.mean([row["blue_fighter_destroyed_end_mean"] for row in episode_results])
        )
        if episode_results
        else 0.0,
        fighter_destroy_balance_end_mean=float(
            np.mean([row["fighter_destroy_balance_end_mean"] for row in episode_results])
        )
        if episode_results
        else 0.0,
        episode_len_mean=float(np.mean([row["episode_len_mean"] for row in episode_results]))
        if episode_results
        else 0.0,
    )

    if output_json is not None:
        with open(output_json, "w", encoding="utf-8") as handle:
            json.dump({"summary": summary, "episodes": episode_results}, handle, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
