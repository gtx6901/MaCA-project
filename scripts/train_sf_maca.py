#!/usr/bin/env python
"""Train MaCA with Sample Factory APPO/PPO."""

from __future__ import annotations

import sys
from os.path import join
from collections import OrderedDict

import numpy as np
import torch

from sample_factory.algorithms.utils.arguments import parse_args
from sample_factory.algorithms.appo import learner as learner_module
from sample_factory.run_algorithm import run_algorithm

from marl_env.sample_factory_registration import register_maca_components


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

        with timing.add_time("buffers"):
            buffer = learner_module.AttrDict()

            for trajectory in trajectories:
                for key, value in trajectory.items():
                    if key not in buffer:
                        buffer[key] = []
                    buffer[key].append(value)

            for key, value in buffer.items():
                if isinstance(value[0], (dict, OrderedDict)):
                    buffer[key] = learner_module.list_of_dicts_to_dict_of_lists(value)

        with timing.add_time("buffer_stack_and_squeeze"):
            tensors_to_squeeze = {
                "actions",
                "log_prob_actions",
                "policy_version",
                "policy_id",
                "values",
                "rewards",
                "dones",
            }

            for dct, key, arr in learner_module.iterate_recursively(buffer):
                tensor = np.stack(arr)
                if key in tensors_to_squeeze and tensor.ndim > 2 and tensor.shape[-1] == 1:
                    tensor = np.squeeze(tensor, axis=-1)
                dct[key] = tensor

        if self.cfg.max_entropy_coeff != 0.0:
            with timing.add_time("max_entropy"), torch.no_grad():
                action_distr_params = buffer.action_logits.reshape((-1, buffer.action_logits.shape[-1]))
                entropies = learner_module.get_action_distribution(
                    self.action_space, torch.Tensor(action_distr_params)
                ).entropy().numpy()
                entropies = entropies.reshape((-1, self.cfg.rollout))
                buffer.rewards += self.cfg.max_entropy_coeff * entropies

        if not self.cfg.with_vtrace:
            with timing.add_time("calc_gae"):
                buffer = self._calculate_gae(buffer)

        with timing.add_time("batching"):
            for dct, key, arr in learner_module.iterate_recursively(buffer):
                envs_dim, time_dim = arr.shape[0:2]
                new_shape = (envs_dim * time_dim,) + arr.shape[2:]
                dct[key] = arr.reshape(new_shape)

            use_pinned_memory = self.cfg.device == "gpu"
            buffer = self.tensor_batcher.cat(buffer, macro_batch_size, use_pinned_memory, timing)

        with timing.add_time("buff_ready"):
            self.shared_buffers.free_trajectory_buffers([r.traj_buffer_idx for r in rollouts])

        with timing.add_time("tensors_gpu_float"):
            device_buffer = self._copy_train_data_to_device(buffer)

        self.tensor_batch_pool.put(buffer)
        return device_buffer

    learner_module.LearnerWorker._prepare_train_buffer = safe_prepare_train_buffer


def _inject_defaults(argv):
    result = list(argv)
    for flag, value in _DEFAULT_FLAGS.items():
        if not any(arg == flag or arg.startswith(flag + "=") for arg in result):
            result.append(f"{flag}={value}")
    return result


def main(argv=None):
    torch.multiprocessing.set_sharing_strategy("file_system")
    _patch_sample_factory_checkpoint_save()
    _patch_sample_factory_buffer_squeeze()
    register_maca_components()
    cfg = parse_args(argv=_inject_defaults(sys.argv[1:] if argv is None else argv))
    return run_algorithm(cfg)


if __name__ == "__main__":
    sys.exit(main())
