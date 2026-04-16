# Script Guide

The repository keeps one active training lane: recurrent MAPPO with a centralized team critic.

Primary entrypoints:

- `train.py`: config-driven training entrypoint
- `evaluate.py`: config-driven evaluation entrypoint
- `train_mappo_maca.py`: low-level recurrent MAPPO trainer
- `eval_mappo_maca.py`: low-level checkpoint evaluator
- `run_mappo_maca_train.sh`: shared shell launcher for MAPPO profiles
- `run_mappo_maca_4060_library_long.sh`: 4060 long-duration profile
- `run_mappo_maca_4060_overnight.sh`: 4060 overnight profile
- `run_mappo_maca_4080_server_scale.sh`: 4080 larger-scale profile
- `collect_teacher_maca.py`: teacher trajectory collection for BC warm start
- `pretrain_bc_maca.py`: actor warm start by behavior cloning

Engineering rules:

- keep one shared launcher plus one profile per hardware lane
- keep train and eval entrypoints separate from launcher profiles
- do not reintroduce deprecated non-MAPPO training wrappers into `scripts/`
