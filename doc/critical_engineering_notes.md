# Critical Engineering Notes (Keep Updated)

Last updated: `2026-04-13` (night takeover rescue patch)

## 1. First Principles

- Trust code + logs + eval JSON, not memory.
- Don’t judge by training reward alone.
- `fix_rule` performance is the primary target; `fix_rule_no_att` is auxiliary.

## 2. Resume vs Fresh-Start

- Resume only if obs/action/model/reward semantics are still compatible.
- If reward shaping or model/obs definition changed materially, do fresh-start.
- For current takeover scripts, default assumption is **resume existing experiment**.

## 3. 2-4 Hour Fast Decision Gate

After `2-4h`, run fixed eval and decide continue/adjust:

- Must check:
  - `win_rate` (vs `fix_rule`)
  - `fire_action_frac_mean`
  - `executed_fire_action_frac_mean`
  - `attack_opportunity_frac_mean`
  - `fire_action_frac_mean / attack_opportunity_frac_mean`
  - `missed_attack_frac_mean`
  - `course_change_frac_mean`
  - `course_unique_frac_mean`
  - `episode_len_mean`
- If fire/opportunity conversion remains persistently low, continuing same config usually has low marginal value.

## 3.1 No-Fire Root-Cause Quick Test

- If `attack_opportunity_frac_mean` is high (e.g. `>0.4`) but fire metrics are still near zero:
  - this is usually **decision collapse to no-fire**, not pure radar blindness.
- If `course_change_frac_mean`/`course_unique_frac_mean` are non-trivial:
  - it is usually **not** “agent fully static”.
- Confirm by policy-level fire probability probe:
  - if fire probability mass when opportunity exists is near zero, treat as policy collapse.

Important code truth (2026-04-13 fix):

- `attack_opportunity` must only count actions with `attack_index > 0`.
- Do **not** use `mask[1:]` directly on flattened action space; that incorrectly includes
  no-fire actions from other course bins and inflates opportunity/missed stats.

## 4. High-Risk Toggles

- `train_in_background_thread=True` is risky in this patch stack; default keep `False`.
- Buffer-squeeze patch is runtime-toggle only; do not enable by default without reason.
- Keep `train_for_env_steps` headroom large enough before long resume runs.
- Keep no-fire rescue knobs controlled:
  - `MACA_FIRE_PROB_FLOOR` should be small (`0.02-0.05`) to break collapse, not force random spam.
  - `MACA_EVAL_FIRE_PROB_FLOOR` should stay `0` for unbiased eval unless running ablation.
- `MACA_MISSED_ATTACK_PENALTY` can be raised moderately when stuck in no-fire collapse, but avoid extreme values.

## 5. Must-Have Guards for Overnight Runs

- verify resume checkpoint exists
- remove `done` before continue
- verify env-step headroom before launch
- enforce minimum env-step delta after run
- run auto-eval monitor during training

## 6. Operational Commands

Quick status:

```bash
ls -lh train_dir/sample_factory/<exp_name>/checkpoint_p0 | tail
tail -n 80 log/<run_tag>.master.log
```

Quick eval:

```bash
conda run --no-capture-output -n maca-py37-min \
  python scripts/eval_sf_maca.py \
  --experiment="<exp_name>" \
  --train_dir=train_dir/sample_factory \
  --episodes=20 \
  --maca_opponent=fix_rule \
  --output_json="log/<exp_name>.eval20.fix_rule.json"
```

Quick monitor thresholds (current takeover defaults):

- `MACA_AUTO_EVAL_LOW_FIRE_THRESHOLD=0.003`
- `MACA_AUTO_EVAL_MIN_FIRE_OPP_RATIO=0.01`
- `MACA_AUTO_EVAL_ANOMALY_FIRE_THRESHOLD=0.003`

No-fire rescue defaults:

- `MACA_FIRE_LOGIT_BIAS=0.0`
- `MACA_FIRE_PROB_FLOOR=0.03`
- `MACA_EVAL_FIRE_PROB_FLOOR=0.0`
