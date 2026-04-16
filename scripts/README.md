# Script Guide

The repo now has two training lanes:

- legacy / audited-only: the old Sample Factory APPO stack
- active / default: the new lightweight MaCA MAPPO lane with centralized team critic

Active MAPPO entrypoints:

- `train_mappo_maca.py`: custom team-critic PPO trainer for MaCA, now with rollout-level multiprocess collectors, recurrent actor, and centralized learner
- `eval_mappo_maca.py`: checkpoint evaluation entrypoint for the MAPPO lane
- `run_mappo_maca_train.sh`: shared launcher used by all MAPPO profiles

Supported MAPPO launcher profiles:

- `run_mappo_maca_4060_library_long.sh`: 4060 8G low-power long-duration training
- `run_mappo_maca_4060_overnight.sh`: 4060 8G full-power overnight training
- `run_mappo_maca_4080_server_scale.sh`: 4080 32G larger-scale training

Legacy Sample Factory entrypoints are kept for audit / regression comparison:

- `train_sf_maca.py`
- `eval_sf_maca.py`
- `run_maca_4060_library_long.sh`
- `run_maca_4060_overnight.sh`
- `run_maca_4080_server_scale.sh`

Engineering rules for new scripts:

- keep one shared launcher plus one profile per hardware lane
- keep train / eval entrypoints separate from launcher profiles
- do not add one-off recovery wrappers back into `scripts/`
- document launcher changes in `doc/medium_upgrade_plan_2026-04-14.md`
