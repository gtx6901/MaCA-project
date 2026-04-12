#!/usr/bin/env python
"""Evaluate a Sample Factory MaCA checkpoint against a fixed opponent."""

from __future__ import annotations

import json
import sys
from collections import OrderedDict

import numpy as np
import torch

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


def _patch_sample_factory_action_masking():
    """Apply MaCA action masks to policy logits during evaluation."""

    invalid_logit = -1e9

    def stash_action_mask(actor_critic, obs_dict):
        action_mask = obs_dict.get("action_mask")
        actor_critic._last_action_mask = None if action_mask is None else action_mask.bool()

    def mask_action_logits(actor_critic, action_logits):
        action_mask = getattr(actor_critic, "_last_action_mask", None)
        actor_critic._last_action_mask = None
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

        return action_logits.masked_fill(~action_mask, invalid_logit)

    def shared_forward_head(self, obs_dict):
        normalize_obs(obs_dict, self.cfg)
        stash_action_mask(self, obs_dict)
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
        stash_action_mask(self, obs_dict)
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
        passthrough.append(arg)
        idx += 1

    return episodes, output_json, passthrough


def main(argv=None):
    register_maca_components()
    _patch_sample_factory_action_masking()

    raw_argv = sys.argv[1:] if argv is None else argv
    episodes, output_json, sf_argv = _parse_cli(raw_argv)
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
                episode_len = float(np.mean([stat.get("episode_len", 0.0) for stat in valid_stats]))
                win_flag = float(np.mean([stat.get("win_flag", 0.0) for stat in valid_stats]) > 0.5)

                episode_results.append(
                    {
                        "episode": len(episode_results) + 1,
                        "win": int(win_flag),
                        "round_reward": round_reward,
                        "opponent_round_reward": opponent_round_reward,
                        "true_reward_mean": float(np.mean(true_rewards)),
                        "invalid_action_frac_mean": invalid_action_frac,
                        "episode_len_mean": episode_len,
                    }
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
