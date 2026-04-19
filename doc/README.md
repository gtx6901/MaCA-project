# Documentation Guide

This directory now keeps only the current MAPPO lane and the original MaCA environment references.

Recommended reading order:

1. `MaCA_MAPPO_Action_Plan.md`
2. `gpu_env_setup.md`
3. `hmarl_script_usage.md`
4. `MaCA环境说明.pdf`

Current default training lane:

- config-driven launcher: `scripts/train.py`
- recurrent MAPPO trainer: `scripts/train_mappo_maca.py`
- structured env wrapper: `marl_env/mappo_env.py`
- centralized actor-critic model: `marl_env/mappo_model.py`
