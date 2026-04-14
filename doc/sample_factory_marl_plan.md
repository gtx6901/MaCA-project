# Sample Factory MARL Plan (Code-Aligned, 2026-04-13)

## 1. Scope

This file is the **single source of truth for current training workflow** in this repo.
If any older document conflicts with this file, trust code + this file.
For quick handoff reminders, also read `doc/critical_engineering_notes.md`.

Mainline training path:

- Environment: `marl_env/maca_parallel_env.py` + `marl_env/sample_factory_env.py`
- Framework: `Sample Factory 1.x`
- Algo: `APPO` (`with_vtrace=True`)
- Policy: shared red fighter policy (`10` fixed fighter slots)
- Recurrent core: `LSTM`

Legacy DQN scripts are kept for history/reproduction only.

## 2. Code Truth Snapshot

### Environment / Action / Observation

- Red fighter agents are fixed to 10 (`red_fighter_0..9`), dead agents remain in set and only no-op is valid.
- Action space is `336` (`COURSE_NUM=16`, `ATTACK_IND_NUM=21`, `ACTION_NUM=16*21`).
- Action legality comes from `fighter_action_utils.py`:
  - Long missile range threshold: `120.0`
  - Short missile range threshold: `50.0`
- Measurement vector is `7`-dim:
  - heading encoded as `sin/cos` + 5 linear fields
- Env-side fallback sanitization is active; policy-side action-mask logit patch is also active.

### Reward Baseline (raw env defaults)

Source: `configuration/reward.py`

- `reward_strike_fighter_success = 900`
- `reward_strike_act_valid = 2`
- `reward_strike_act_invalid = -4`
- `reward_keep_alive_step = -1`
- `reward_draw = -1500`
- `reward_totally_win = 8000`
- `reward_totally_lose = -2000`

Runtime overrides are supported via env vars in `marl_env/runtime_tweaks.py`.

### Eval Metrics Already Implemented

`scripts/eval_sf_maca.py` summary includes:

- `win_rate`
- `round_reward_mean`
- `opponent_round_reward_mean`
- `true_reward_mean`
- `invalid_action_frac_mean`
- `fire_action_frac_mean`
- `attack_opportunity_frac_mean`
- `missed_attack_frac_mean`
- `episode_len_mean`

## 3. Current Default Hyperparameters

### Registration defaults (`marl_env/sample_factory_registration.py`)

- `num_workers=8`
- `rollout=64`
- `recurrence=64`
- `batch_size=5120`
- `ppo_epochs=4`
- `hidden_size=256`
- `learning_rate=1e-4`
- `gamma=0.999`
- `reward_scale=0.005`
- `reward_clip=50.0`
- `max_policy_lag=15`
- `exploration_loss_coeff=0.02`

### 4060 baseline launcher (`scripts/run_sf_maca_4060_baseline.sh`)

Default effective values match the same core profile:

- `num_workers=8`
- `rollout=64`
- `recurrence=64`
- `batch_size=5120`
- `train_in_background_thread=False`
- opponent `fix_rule`

### Takeover / validation tuned defaults (`2026-04-13` hotfix)

For `scripts/run_sf_maca_takeover_night.sh` -> `scripts/run_sf_maca_recovery_validation.sh`:

- `reward_scale=0.002`
- `exploration_loss_coeff=0.06`
- `MACA_MISSED_ATTACK_PENALTY=25`
- `MACA_FIRE_LOGIT_BIAS=1.6` (takeover) / `1.2` (validation fallback)
- monitor thresholds:
  - `MACA_AUTO_EVAL_LOW_FIRE_THRESHOLD=0.001`
  - `MACA_AUTO_EVAL_MIN_FIRE_OPP_RATIO=0.002`
  - `MACA_AUTO_EVAL_ANOMALY_FIRE_THRESHOLD=0.001`

## 4. Resume vs Fresh-Start Rule

### Use `resume` when:

- same observation/action/model definitions
- same experiment goal (`fix_rule` track continuation)
- you want incremental gain from existing checkpoint

### Use `fresh_start` when:

- model architecture changed
- observation schema changed
- action semantics changed
- reward shaping changed substantially (distribution shift for critic/policy)

Current recovery/takeover scripts are explicitly **resume-oriented**:

- `scripts/run_sf_maca_recovery_validation.sh`
- `scripts/run_sf_maca_takeover_night.sh`

## 5. Stability Guards in Current Resume Workflow

Implemented checks in recovery validation flow:

- resume experiment path existence check
- latest checkpoint existence + parsable env-step check
- env-step headroom check before run
- minimum env-step delta check after run
- `done` file removal before continuation
- online auto-eval monitor process
- low-fire and low fire-to-opportunity ratio guardrails

## 6. Recommended Commands

### Smoke

```bash
RUN_ID="$(date +%Y%m%d_%H%M%S)"
EXP_NAME="sf_maca_gpu_smoke_${RUN_ID}" \
bash scripts/run_sf_maca_gpu_smoke.sh
```

### Baseline Training

```bash
RUN_ID="$(date +%Y%m%d_%H%M%S)"
EXP_NAME="sf_maca_4060_fixrule_${RUN_ID}" \
TRAIN_SECONDS=7200 \
bash scripts/run_sf_maca_4060_baseline.sh
```

### Resume Validation / Takeover

```bash
bash scripts/run_sf_maca_takeover_night.sh
```

### Evaluation

```bash
conda run --no-capture-output -n maca-py37-min \
  python scripts/eval_sf_maca.py \
  --experiment="$EXP_NAME" \
  --train_dir=train_dir/sample_factory \
  --episodes=30 \
  --maca_opponent=fix_rule \
  --output_json="log/${EXP_NAME}.eval30.fix_rule.json"
```

## 7. Practical Rule

For this repo, “baseline” means:

1. can launch
2. can resume
3. can checkpoint safely
4. can be independently evaluated with fixed settings

It does **not** mean “already optimal policy.”
