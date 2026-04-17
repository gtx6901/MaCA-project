# HMARL Script Usage

This document is the single operational guide for the HMARL scripts currently in use.

Compatibility note:

- The trainer internals were refactored into `marl_train/` modules for maintainability and performance, but script entrypoints and CLI usage in this document remain unchanged.

## 1. Prerequisites

- Workspace root: `/home/lehan/MaCA-master`
- Python environment: `maca-py37-min`
- Run all commands from repo root:

```bash
cd /home/lehan/MaCA-master
```

## 2. Main Scripts

- `scripts/hmarl_ctl.sh`: unified process control (`start`, `start-fg`, `stop`, `status`, `logs`, `tb`)
- `scripts/run_hmarl_train.sh`: training launcher (supports resume by target env steps or checkpoint path)
- `scripts/run_hmarl_eval.sh`: evaluation launcher + acceptance checker

## 3. Training (Recommended)

Use `hmarl_ctl.sh` for daily operations.

### 3.1 Start training in background

```bash
EXP_NAME="hmarl_recover_from226944_0417_151702"
bash scripts/hmarl_ctl.sh start "$EXP_NAME" 200000
```

Notes:

- The second argument (`200000`) is `resume_from`.
- If no exact checkpoint at `200000` exists, trainer automatically falls back to the latest checkpoint `<= target`.
- If no checkpoint is `<= target`, it falls back to the earliest checkpoint.

### 3.2 Start training in foreground

```bash
EXP_NAME="hmarl_recover_from226944_0417_151702"
bash scripts/hmarl_ctl.sh start-fg "$EXP_NAME" 200000
```

### 3.3 Check status

```bash
EXP_NAME="hmarl_recover_from226944_0417_151702"
bash scripts/hmarl_ctl.sh status "$EXP_NAME"
```

### 3.4 View latest log

```bash
EXP_NAME="hmarl_recover_from226944_0417_151702"
bash scripts/hmarl_ctl.sh logs "$EXP_NAME"
```

### 3.5 Stop training

```bash
EXP_NAME="hmarl_recover_from226944_0417_151702"
bash scripts/hmarl_ctl.sh stop "$EXP_NAME"
```

## 4. TensorBoard

### 4.1 Start TensorBoard

```bash
bash scripts/hmarl_ctl.sh tb 6007
```

Open in browser:

- `http://localhost:6007`

### 4.2 Resume behavior and curve cleanup

- On resume, trainer writes with `purge_step=<resumed_env_steps>`.
- This removes stale future steps in the same TB log directory and prevents mixed old/new curves after restarting from an earlier checkpoint.

## 5. Training Launcher (Direct)

You can call `run_hmarl_train.sh` directly when needed.

Usage:

```bash
bash scripts/run_hmarl_train.sh [experiment_name] [teacher_ckpt] [resume] [resume_from]
```

Examples:

```bash
# Resume from target env steps (auto select checkpoint)
bash scripts/run_hmarl_train.sh hmarl_recover_from226944_0417_151702 "" resume 200000

# Resume from explicit checkpoint file
bash scripts/run_hmarl_train.sh hmarl_recover_from226944_0417_151702 "" resume train_dir/mappo/hmarl_recover_from226944_0417_151702/checkpoint/checkpoint_000000256_226944.pt

# New run (no resume)
bash scripts/run_hmarl_train.sh hmarl_new_run
```

`resume_from` meaning:

- numeric value: target env steps
- existing file path: exact checkpoint file

## 6. Evaluation

Usage:

```bash
bash scripts/run_hmarl_eval.sh <experiment_name> [episodes] [opponent]
```

Example:

```bash
bash scripts/run_hmarl_eval.sh hmarl_recover_from226944_0417_151702 50 fix_rule
```

Outputs:

- JSON report: `train_dir/mappo/<experiment>/eval/eval_manual_<opponent>_<episodes>.json`
- Acceptance check result is printed by `scripts/check_hmarl_acceptance.py`

## 7. Runtime Artifacts

- pid file: `train_dir/mappo/_manual_logs/<experiment>.pid`
- train logs: `train_dir/mappo/_manual_logs/train_<experiment>_*.log`
- tensorboard logs: `train_dir/mappo/<experiment>/tb/`
- checkpoints: `train_dir/mappo/<experiment>/checkpoint/`

## 8. Quick Recovery Playbook

```bash
cd /home/lehan/MaCA-master
EXP_NAME="hmarl_recover_from226944_0417_151702"

# 1) ensure old run stopped
bash scripts/hmarl_ctl.sh stop "$EXP_NAME"

# 2) restart from target step (fallback is automatic)
bash scripts/hmarl_ctl.sh start "$EXP_NAME" 200000

# 3) verify running and watch logs
bash scripts/hmarl_ctl.sh status "$EXP_NAME"
bash scripts/hmarl_ctl.sh logs "$EXP_NAME"

# 4) start TB in another terminal
bash scripts/hmarl_ctl.sh tb 6007
```
