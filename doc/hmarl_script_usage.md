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

Default behavior is fresh start (no resume):

```bash
EXP_NAME="hmarl_fresh_$(date +%m%d_%H%M%S)"
bash scripts/hmarl_ctl.sh start "$EXP_NAME"
```

Start different agent variants with the same unified script:

```bash
# Version A: baseline (full_discrete + no rule attack)
EXP_NAME="hmarl_baseline_$(date +%m%d_%H%M%S)"
HMARL_AGENT_VARIANT=baseline bash scripts/hmarl_ctl.sh start "$EXP_NAME"

# Version B: rule_fire (nearest_target + fire_or_not + disable high-level mode)
EXP_NAME="hmarl_rulefire_$(date +%m%d_%H%M%S)"
HMARL_AGENT_VARIANT=rule_fire bash scripts/hmarl_ctl.sh start "$EXP_NAME"
```

Resume only when explicitly passing `resume_from`:

```bash
EXP_NAME="hmarl_recover_from226944_0417_151702"
bash scripts/hmarl_ctl.sh start "$EXP_NAME" 200000
```

Notes:

- If the second argument is omitted, training starts fresh.
- The second argument (`200000`) is `resume_from` when provided.
- If no exact checkpoint at `200000` exists, trainer automatically falls back to the latest checkpoint `<= target`.
- If no checkpoint is `<= target`, it falls back to the earliest checkpoint.

### 3.2 Start training in foreground

Default fresh start:

```bash
EXP_NAME="hmarl_fresh_$(date +%m%d_%H%M%S)"
bash scripts/hmarl_ctl.sh start-fg "$EXP_NAME"
```

Foreground resume:

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
# Default: aggregate all experiments (each experiment appears as an individual run name)
bash scripts/hmarl_ctl.sh tb 6007

# Single experiment view
bash scripts/hmarl_ctl.sh tb 6007 hmarl_0417_172930
```

Open in browser:

- `http://localhost:6007`

Notes:

- `tb 6007` uses multi-run mode via `--logdir_spec` and shows one run per experiment.
- `tb 6007 <exp_name>` uses single-run mode; TensorBoard may display run name as `.` (this is expected because logdir points directly to one `tb/` folder).

### 4.2 Resume behavior and curve cleanup

- On resume, trainer writes with `purge_step=<resumed_env_steps>`.
- This removes stale future steps in the same TB log directory and prevents mixed old/new curves after restarting from an earlier checkpoint.

## 5. Training Launcher (Direct)

You can call `run_hmarl_train.sh` directly when needed.

Usage:

```bash
bash scripts/run_hmarl_train.sh [experiment_name] [teacher_ckpt] [resume] [resume_from] [agent_variant] [extra_train_args...]
```

Examples:

```bash
# Resume from target env steps (auto select checkpoint)
bash scripts/run_hmarl_train.sh hmarl_recover_from226944_0417_151702 "" resume 200000

# Resume from explicit checkpoint file
bash scripts/run_hmarl_train.sh hmarl_recover_from226944_0417_151702 "" resume train_dir/mappo/hmarl_recover_from226944_0417_151702/checkpoint/checkpoint_000000256_226944.pt

# New run (no resume)
bash scripts/run_hmarl_train.sh hmarl_new_run

# New run with explicit variant
bash scripts/run_hmarl_train.sh hmarl_rulefire_run "" "" "" rule_fire

# Pass extra trainer args (example: disable GUI during periodic eval)
bash scripts/run_hmarl_train.sh hmarl_rulefire_run "" "" "" rule_fire --eval_maca_render false
```

`resume_from` meaning:

- numeric value: target env steps
- existing file path: exact checkpoint file

## 6. Evaluation

Usage:

```bash
bash scripts/run_hmarl_eval.sh <experiment_name> [episodes] [opponent] [extra_eval_args...]
```

Example:

```bash
bash scripts/run_hmarl_eval.sh hmarl_recover_from226944_0417_151702 50 fix_rule
```

Pass-through behavior:

- The first 3 positional arguments are fixed: `experiment_name`, `episodes`, `opponent`.
- Any arguments after that are forwarded directly to `scripts/eval_mappo_maca.py`.
- This means you can pass `--maca_render`, `--deterministic`, `--checkpoint`, `--output_json`, etc. through the unified launcher.

### 6.1 GUI Evaluation (Render On)

GUI is now enabled by default for manual evaluation (`run_hmarl_eval.sh`).

Default GUI evaluation:

```bash
EXP_NAME="hmarl_recover_from226944_0417_151702"
EPISODES=3
OPPONENT="fix_rule"

bash scripts/run_hmarl_eval.sh "${EXP_NAME}" "${EPISODES}" "${OPPONENT}"
```

Explicit GUI evaluation (equivalent):

```bash
EXP_NAME="hmarl_recover_from226944_0417_151702"
EPISODES=3
OPPONENT="fix_rule"

bash scripts/run_hmarl_eval.sh "${EXP_NAME}" "${EPISODES}" "${OPPONENT}" \
  --maca_render True
```

Headless override (disable GUI):

```bash
bash scripts/run_hmarl_eval.sh "${EXP_NAME}" "${EPISODES}" "${OPPONENT}" \
  --maca_render False
```

If you want GUI + JSON output at the same time:

```bash
EXP_NAME="hmarl_recover_from226944_0417_151702"
EPISODES=3
OPPONENT="fix_rule"
OUT_JSON="train_dir/mappo/${EXP_NAME}/eval/eval_manual_gui_${OPPONENT}_${EPISODES}.json"

bash scripts/run_hmarl_eval.sh "${EXP_NAME}" "${EPISODES}" "${OPPONENT}" \
  --maca_render True \
  --output_json "${OUT_JSON}"
```

Remote server note:

- GUI rendering requires a valid display.
- For SSH sessions, use X11 forwarding (for example `ssh -X`) or run under a virtual display (for example `xvfb-run`).

### 6.2 Training-Loop Evaluation GUI Default

Periodic evaluation during training (controlled by `--eval_every_env_steps`) now also defaults to GUI on.

If you need headless training-time evaluation:

```bash
bash scripts/run_hmarl_train.sh hmarl_headless_eval "" "" "" baseline \
  --eval_maca_render false
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

我在做一个多智能体空战强化学习项目，项目代码就是我给你attach上的github仓库。我已在当前仓库上实现最小判别实验开关与全链路接入，支持 nearest_target + fire_or_not、支持关闭 high-level mode 的最小 ablation，并把策略原始攻击决策与最终执行攻击动作分离记录，同时新增你要求的诊断指标到训练日志与 TensorBoard；所有修改文件已通过静态错误检查。 现在开始跑“ 规则目标 + fire/not 二值策略版”。我开启了固定开火策略模式，现在开始训练，但似乎遇到了瓶颈，在跑了500k轮后，发现有的AI在开局之后会很积极的接敌和开火，但有的会直接向左上角冲过去。由于每台小飞机的导弹数量是有限制的，
在本来的DQN版本里，有一个全局图像通过CNN输入，但我感觉这个东西信息量很大，反而不利于训练，所以删了，具体的接口可以看maca环境的输出。在现在的情况下，有可能重新接入，提高Agent的全局态势感知能力嘛？