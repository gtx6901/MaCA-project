# Script Guide

The `scripts/` directory contains a mix of stable entrypoints and one-off experiment helpers.

Primary Sample Factory entrypoints:

- `train_sf_maca.py`: main SF training launcher/patch point
- `eval_sf_maca.py`: checkpoint evaluation entrypoint
- `run_sf_maca_radar_support_8h_curriculum.sh`: current recommended fresh-start night run
- `run_sf_maca_decoupled_8h_curriculum.sh`: generic 8h decoupled curriculum wrapper

Recovery and audit helpers:

- `eval_sf_maca_recovery_gate.sh`
- `auto_eval_during_training.sh`
- `generate_night_report.py`
- `run_sf_maca_recovery_validation.sh`
- `run_sf_maca_recovery_fresh_gate.sh`

Legacy or hardware-specific scripts:

- `run_sf_maca_4060_baseline.sh`
- `run_sf_maca_4060_8g_curriculum.sh`
- `run_sf_maca_4080_freshstart.sh`
- `run_sf_maca_recovery_curriculum.sh`
- `run_sf_maca_recovery_night.sh`
- `run_sf_maca_takeover_night.sh`

Keep new scripts aligned with these rules:

- prefer one stable wrapper per training lane
- keep eval helpers separate from launchers
- avoid adding generated reports or logs here
