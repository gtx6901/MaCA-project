# Documentation Guide

This directory now mixes three kinds of material:

- Canonical references:
  - `tutorial.md`
  - `MaCA环境说明.pdf`
- Active engineering notes:
  - `sample_factory_marl_plan.md`
  - `critical_engineering_notes.md`
  - `gpu_env_setup.md`
  - `rl_training_postrun_playbook.md`
- Historical run notes and retrospectives:
  - `recovery_curriculum_status_2026-04-12.md`
  - `training_retrospective.md`
  - `dqn_upgrade_playbook_2026-04-10.md`
  - `ai_conversation_kickoff.md`
  - `rl_marl_from_scratch_guide.md`

For current active training work, start with:

1. `sample_factory_marl_plan.md`
2. `critical_engineering_notes.md`
3. `rl_training_postrun_playbook.md`
4. `medium_upgrade_plan_2026-04-14.md`

Math/reference scratch files:

- `appo_vtrace_math_fasttrack.md`
- `appo_vtrace_formula_sheet.tex`

Current active execution / audit record:

- `medium_upgrade_plan_2026-04-14.md`

Current default training lane:

- custom lightweight MAPPO trainer under `scripts/train_mappo_maca.py`
- structured env wrapper under `marl_env/mappo_env.py`
- centralized critic model under `marl_env/mappo_model.py`
