#!/usr/bin/env python
"""Train MaCA with Sample Factory APPO/PPO."""

from __future__ import annotations

import math
import sys
from os.path import join
from collections import OrderedDict

import numpy as np
import torch

from fighter_action_utils import ATTACK_IND_NUM
from sample_factory.algorithms.utils.arguments import parse_args
from sample_factory.algorithms.appo import learner as learner_module
from sample_factory.algorithms.appo import model as appo_model
from sample_factory.algorithms.appo.appo_utils import iterate_recursively, list_of_dicts_to_dict_of_lists
from sample_factory.algorithms.appo.model_utils import normalize_obs
from sample_factory.algorithms.utils.action_distributions import get_action_distribution, sample_actions_log_probs
from sample_factory.run_algorithm import run_algorithm
from sample_factory.utils.utils import AttrDict

from marl_env.sample_factory_registration import register_maca_components
from marl_env.runtime_tweaks import (
    buffer_squeeze_patch_enabled,
    format_runtime_tweaks,
    get_attack_prior_strength,
    get_fire_logit_bias,
    get_fire_prob_floor,
    load_runtime_tweaks,
)


_DEFAULT_FLAGS = {
    "--algo": "APPO",
    "--env": "maca_aircombat",
    "--experiment": "maca_sf_baseline",
}


def _patch_sample_factory_checkpoint_save():
    """Avoid SF 1.x temporary filename incompatibility with newer torch zip writer."""

    def safe_save(self):
        checkpoint = self._get_checkpoint_dict()
        assert checkpoint is not None

        checkpoint_dir = self.checkpoint_dir(self.cfg, self.policy_id)
        tmp_filepath = join(checkpoint_dir, "temp_checkpoint.pth.tmp")
        checkpoint_name = f"checkpoint_{self.train_step:09d}_{self.env_steps}.pth"
        filepath = join(checkpoint_dir, checkpoint_name)

        learner_module.log.info("Saving %s...", tmp_filepath)
        torch.save(checkpoint, tmp_filepath)
        learner_module.log.info("Renaming %s to %s", tmp_filepath, filepath)
        learner_module.os.replace(tmp_filepath, filepath)

        while len(self.get_checkpoints(checkpoint_dir)) > self.cfg.keep_checkpoints:
            oldest_checkpoint = self.get_checkpoints(checkpoint_dir)[0]
            if learner_module.os.path.isfile(oldest_checkpoint):
                learner_module.log.debug("Removing %s", oldest_checkpoint)
                learner_module.os.remove(oldest_checkpoint)

        if self.cfg.save_milestones_sec > 0:
            if learner_module.time.time() - self.last_milestone_time >= self.cfg.save_milestones_sec:
                milestones_dir = learner_module.ensure_dir_exists(join(checkpoint_dir, "milestones"))
                milestone_path = join(milestones_dir, f"{checkpoint_name}.milestone")
                learner_module.log.debug("Saving a milestone %s", milestone_path)
                learner_module.shutil.copy(filepath, milestone_path)
                self.last_milestone_time = learner_module.time.time()

    learner_module.LearnerWorker._save = safe_save


def _patch_sample_factory_buffer_squeeze():
    """Keep env/time dimensions even when a macro-batch has a single trajectory."""

    def safe_prepare_train_buffer(self, rollouts, macro_batch_size, timing):
        trajectories = [learner_module.AttrDict(r["t"]) for r in rollouts]
        max_entropy_coeff = getattr(self.cfg, "max_entropy_coeff", 0.0)

        with timing.add_time("buffers"):
            buffer = learner_module.AttrDict()

            for trajectory in trajectories:
                for key, value in trajectory.items():
                    if key not in buffer:
                        buffer[key] = []
                    buffer[key].append(value)

            for key, value in buffer.items():
                if isinstance(value[0], (dict, OrderedDict)):
                    buffer[key] = list_of_dicts_to_dict_of_lists(value)

        if max_entropy_coeff != 0.0:
            with timing.add_time("max_entropy"), torch.no_grad():
                action_logits = np.concatenate(buffer.action_logits, axis=0)
                action_distr_params = action_logits.reshape((-1, action_logits.shape[-1]))
                entropies = learner_module.get_action_distribution(
                    self.action_space, torch.Tensor(action_distr_params)
                ).entropy().numpy()
                entropies = entropies.reshape((-1, self.cfg.rollout))
                for idx, rewards in enumerate(buffer.rewards):
                    buffer.rewards[idx] = rewards + max_entropy_coeff * entropies[idx]

        if not self.cfg.with_vtrace:
            with timing.add_time("calc_gae"):
                buffer = self._calculate_gae(buffer)

        with timing.add_time("batching"):
            use_pinned_memory = self.cfg.device == "gpu"
            buffer = self.tensor_batcher.cat(buffer, macro_batch_size, use_pinned_memory, timing)

        with timing.add_time("buff_ready"):
            self.shared_buffers.free_trajectory_buffers([r.traj_buffer_idx for r in rollouts])

        with timing.add_time("tensors_gpu_float"):
            device_buffer = self._copy_train_data_to_device(buffer)

        with timing.add_time("squeeze"):
            tensors_to_squeeze = [
                "actions",
                "log_prob_actions",
                "policy_version",
                "policy_id",
                "values",
                "rewards",
                "dones",
                "rewards_cpu",
                "dones_cpu",
            ]
            for tensor_name in tensors_to_squeeze:
                tensor = device_buffer.get(tensor_name, None)
                if tensor is not None and tensor.ndim > 1 and tensor.shape[-1] == 1:
                    device_buffer[tensor_name] = tensor.squeeze(-1)

        self.tensor_batch_pool.put(buffer)
        return device_buffer

    learner_module.LearnerWorker._prepare_train_buffer = safe_prepare_train_buffer


def _patch_sample_factory_action_masking():
    """Apply MaCA action masks to policy logits in both actors and learner updates."""

    invalid_logit = -1e9
    attack_prior_strength = max(0.0, get_attack_prior_strength())
    fire_logit_bias = get_fire_logit_bias()
    fire_prob_floor = max(0.0, min(0.95, get_fire_prob_floor()))

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
        course_prior = obs_dict.get("course_prior")
        actor_critic._last_course_prior = course_prior
        attack_prior = obs_dict.get("attack_prior")
        actor_critic._last_attack_prior = attack_prior

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
            if fire_logit_bias != 0.0 and torch.any(attack_mask[..., 1:]):
                fire_bias_mask = attack_mask.clone()
                fire_bias_mask[..., 0] = False
                attack_logits = attack_logits + fire_bias_mask.to(attack_logits.dtype) * fire_logit_bias

            if fire_prob_floor > 0.0 and attack_logits.shape[-1] > 1:
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
                    target_logit = math.log(fire_prob_floor / max(1e-8, 1.0 - fire_prob_floor))
                    delta = torch.clamp(target_logit + logsum_non_fire - logsum_fire, min=0.0, max=20.0)
                    rows = has_attack_opportunity.squeeze(-1) & torch.isfinite(delta)
                    if torch.any(rows):
                        attack_logits = attack_logits + valid_fire.to(attack_logits.dtype) * delta.unsqueeze(-1)

            split_logits[-1] = attack_logits
            return torch.cat(split_logits, dim=-1)

        action_logits = action_logits.masked_fill(~action_mask, invalid_logit)
        if action_logits.shape[-1] % ATTACK_IND_NUM == 0 and action_logits.shape[-1] > 1:
            action_indices = torch.arange(action_logits.shape[-1], device=action_logits.device)
            fire_positions = (action_indices % ATTACK_IND_NUM) > 0
            valid_fire = action_mask & fire_positions.unsqueeze(0)
            has_attack_opportunity = valid_fire.any(dim=-1, keepdim=True)
        else:
            fire_positions = None
            valid_fire = None
            has_attack_opportunity = None

        if fire_logit_bias != 0.0 and valid_fire is not None:
            if torch.any(has_attack_opportunity):
                fire_bias_mask = valid_fire & has_attack_opportunity
                action_logits = action_logits + fire_bias_mask.to(action_logits.dtype) * fire_logit_bias

        if fire_prob_floor > 0.0 and valid_fire is not None:
            if torch.any(has_attack_opportunity):
                valid_fire_with_opp = valid_fire & has_attack_opportunity
                valid_non_fire = action_mask & (~fire_positions).unsqueeze(0)
                logsum_fire = torch.logsumexp(action_logits.masked_fill(~valid_fire, invalid_logit), dim=-1)
                logsum_non_fire = torch.logsumexp(action_logits.masked_fill(~valid_non_fire, invalid_logit), dim=-1)
                target_logit = math.log(fire_prob_floor / max(1e-8, 1.0 - fire_prob_floor))
                delta = target_logit + logsum_non_fire - logsum_fire
                delta = torch.clamp(delta, min=0.0, max=20.0)
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


def _patch_sample_factory_explained_variance_logging():
    """Add critic explained variance to SF train summaries and learner logs."""

    original_record_summaries = learner_module.LearnerWorker._record_summaries

    def patched_record_summaries(self, train_loop_vars):
        stats = original_record_summaries(self, train_loop_vars)
        var = train_loop_vars

        try:
            targets = var.targets.detach().float()
            values = var.values.detach().float()
            valid_mask = var.valids.to(device=targets.device, dtype=torch.bool)

            if valid_mask.shape != targets.shape:
                valid_mask = valid_mask.reshape_as(targets)

            if torch.any(valid_mask):
                valid_targets = targets[valid_mask]
                valid_values = values[valid_mask]
            else:
                valid_targets = targets.reshape(-1)
                valid_values = values.reshape(-1)

            target_var = torch.var(valid_targets, unbiased=False)
            if torch.isfinite(target_var) and target_var.item() > 1e-8:
                residual_var = torch.var(valid_targets - valid_values, unbiased=False)
                explained_variance = 1.0 - (residual_var / target_var)
            else:
                explained_variance = torch.zeros((), device=valid_targets.device)

            stats["explained_variance"] = float(explained_variance.item())
            stats["value_target_mean"] = float(valid_targets.mean().item())
            stats["value_target_std"] = float(valid_targets.std(unbiased=False).item())

            learner_module.log.info(
                "Train summaries: explained_variance=%.4f value_loss=%.4f policy_loss=%.4f",
                stats["explained_variance"],
                float(stats.get("value_loss", 0.0)),
                float(stats.get("policy_loss", 0.0)),
            )
        except Exception as exc:
            learner_module.log.warning("Failed to compute explained variance: %s", exc)

        return stats

    learner_module.LearnerWorker._record_summaries = patched_record_summaries


def _inject_defaults(argv):
    result = list(argv)
    for flag, value in _DEFAULT_FLAGS.items():
        if not any(arg == flag or arg.startswith(flag + "=") for arg in result):
            result.append(f"{flag}={value}")
    return result


def main(argv=None):
    load_runtime_tweaks()
    torch.multiprocessing.set_sharing_strategy("file_system")
    print(f"[maca_runtime] {format_runtime_tweaks()}", flush=True)
    _patch_sample_factory_checkpoint_save()
    if buffer_squeeze_patch_enabled():
        _patch_sample_factory_buffer_squeeze()
    _patch_sample_factory_action_masking()
    _patch_sample_factory_explained_variance_logging()
    register_maca_components()
    cfg = parse_args(argv=_inject_defaults(sys.argv[1:] if argv is None else argv))
    return run_algorithm(cfg)


if __name__ == "__main__":
    sys.exit(main())
