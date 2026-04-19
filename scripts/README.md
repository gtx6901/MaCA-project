# Script Guide

The repository keeps one active training lane: recurrent MAPPO with a centralized team critic.

Primary entrypoints:

- `train.py`: config-driven training entrypoint
- `evaluate.py`: config-driven evaluation entrypoint
- `hmarl_ctl.sh`: unified HMARL process control (start/stop/status/logs/tb)
- `run_hmarl_train.sh`: HMARL training launcher with resume controls
- `run_hmarl_eval.sh`: HMARL evaluation launcher + acceptance checker
- `run_mappo_fixrule_teacher_pipeline.sh`: one-click `fix_rule -> teacher -> BC -> PPO` launcher
- `train_mappo_maca.py`: low-level recurrent MAPPO trainer
- `eval_mappo_maca.py`: low-level checkpoint evaluator
- `collect_teacher_maca.py`: teacher trajectory collection for BC warm start
- `pretrain_bc_maca.py`: recurrent actor warm start by sequence behavior cloning

Refactor note:

- `scripts/train_mappo_maca.py` keeps the same CLI and remains the training entrypoint.
- Core training logic is split into `marl_train/` (`collector`, `rollout`, `ppo_update`, `checkpoint`, `eval`, `logging_utils`) to reduce single-file complexity.

Engineering rules:

- keep one active launcher path for MAPPO mainline scripts
- keep train and eval entrypoints separate from launcher profiles
- do not reintroduce deprecated non-MAPPO training wrappers into `scripts/`

HMARL usage reference:

- `doc/hmarl_script_usage.md`
