# Medium Upgrade Plan — 2026-04-14

## Goal

Under current time/GPU constraints, push the existing Sample Factory APPO lane toward a **medium-scope upgrade** that directly targets the validated `fix_rule` bottlenecks:

- sparse / unstable contact
- weak pursuit continuity
- high-frequency heading switching
- low conversion from intermittent contact into stable attack windows

This plan explicitly avoids large architectural rewrites such as full hierarchy, GNN command layers, MAPPO replacement, or self-play infrastructure.

## Evidence Base

Current audited evidence before this plan:

- `fix_rule` remains the hard target:
  - `log/sf_maca_local_low_gpu_.eval5.fix_rule.audit_20260414_1837.json`
  - `log/sf_maca_decoupled_8h_20260414_0040.eval20.fix_rule.audit_20260414_1303.json`
- `fix_rule_no_att` is easier but non-transfer is obvious:
  - `log/sf_maca_local_low_gpu_.eval5.fix_rule_no_att.audit_20260414_1837.json`
  - `log/sf_maca_decoupled_8h_20260414_0040.eval20.fix_rule_no_att.audit_20260414_1307.json`

Main audit takeaway:

1. the agent is **not fully blind** (`contact` > 0)
2. the agent is **not mainly failing by refusing to fire when opportunities exist** (`executed_fire ≈ opportunity`)
3. the main bottleneck is **contact formation + pursuit/control continuity**, with perception insufficiency still present but not dominant alone

## Medium-Scope Upgrade Checklist

### 1. Tactical-mode observation

Add a cheap pseudo-hierarchical phase indicator to the fighter observation:

- `search`
- `reacquire`
- `pursue`
- `attack`

Rationale:

- Borrow the useful inductive bias from hierarchical air-combat papers without rebuilding the training stack.
- Reduce task entanglement inside one shared fighter policy.

### 2. Course prior observation + policy bias

Add a heuristic pursuit prior over course bins and expose it as an auxiliary observation channel used only at action-logit time.

Rationale:

- Early exploration in `fix_rule` is too inefficient.
- A soft prior is cheaper than behavior cloning or guided policy training.
- The prior must remain soft; the learned policy still controls the final action.

### 3. Course smoothing / persistence

Post-process course decisions before env execution:

- hold course for a short number of steps
- cap per-step course-bin jumps

Rationale:

- Current `course_change_frac` is too high.
- The policy behaves more like a noisy classifier than a pursuit controller.

### 4. Relative-motion features

Keep the already-added track memory and extend it with low-cost motion cues:

- closure rate
- bearing-rate
- track streak length

Rationale:

- Gives the policy “state + first derivative” instead of only instantaneous geometry.

### 5. Script consolidation

Keep only three launcher lanes:

- 4060 8G library low-power long-duration
- 4060 8G overnight training
- 4080 32G server-scale training

Everything else should be removed if it is a legacy launcher or one-off helper no longer needed for the active training lane.

### 6. Artifact cleanup

Delete stale Sample Factory experiment directories that no longer represent a recommended training path.

## Implemented on 2026-04-14

### Code changes

1. Tactical-mode observation
   - Added `search / reacquire / pursue / attack` pseudo-mode one-hot into fighter measurements.
   - Implemented in `marl_env/sample_factory_env.py`.

2. Course prior observation + soft course-logit bias
   - Added `course_prior` observation vector over course bins.
   - Added `maca_course_prior_strength` config so the policy uses the prior as a soft bias rather than a hard override.
   - Implemented in:
     - `marl_env/sample_factory_env.py`
     - `scripts/train_sf_maca.py`
     - `scripts/eval_sf_maca.py`

3. Course smoothing / persistence
   - Added executed-course hold and per-step course-bin jump cap.
   - Implemented in `marl_env/sample_factory_env.py`.

4. Medium-upgrade config surface
   - Added CLI/config args in `marl_env/sample_factory_registration.py`:
     - tactical mode observation
     - course prior observation
     - course prior strength
     - course hold steps
     - max course change bins

5. Launcher defaults updated
   - The feature-on training lane now defaults to:
     - relative-motion observation on
     - delta-course action on
     - tactical-mode observation on
     - course prior on
     - course smoothing on

### Script consolidation result

Kept core stack:

- `scripts/train_sf_maca.py`
- `scripts/eval_sf_maca.py`
- `scripts/run_sf_maca_decoupled_8h_curriculum.sh`
- `scripts/run_sf_maca_radar_support_8h_curriculum.sh`

Kept user-facing launcher profiles:

- `scripts/run_maca_4060_library_long.sh`
- `scripts/run_maca_4060_overnight.sh`
- `scripts/run_maca_4080_server_scale.sh`

Deleted stale launchers / one-off helpers from `scripts/`.

### Model / artifact cleanup result

Deleted stale Sample Factory experiment directories under `train_dir/sample_factory/`:

- `sf_maca_decoupled_8h_20260414_0040`
- `sf_maca_fs_probe_20260413_181441`
- `sf_maca_local_low_gpu_`
- `sf_maca_local_low_gpu_2026年 04月 14日 星期二 16:01:00 CST`
- `sf_maca_recovery_20260412_163435`

## Why this is still medium-scope, not large-scope

The current round keeps the existing:

- APPO + V-trace training stack
- shared fighter policy
- Sample Factory integration
- existing env/reward structure

It only injects stronger inductive bias into:

- phase awareness
- pursuit continuity
- soft action guidance
- launcher hygiene

This keeps the project understandable enough to explain on a resume while still reflecting serious engineering judgment.

## References / Idea Sources

These references are not copied mechanically into the codebase. They are used as design inspiration and reduced to medium-scope changes compatible with the current project.

1. Hierarchical air combat decision making:
   - `https://www.nature.com/articles/s41598-024-54938-5`
2. Continuous-control / curriculum insight for air combat maneuvering:
   - `https://www.mdpi.com/2076-3417/13/16/9421`
3. Guided-policy / structured-prior inspiration:
   - `https://www.nature.com/articles/s41598-024-54268-6`
4. Multi-tiered / graph-structured command inspiration:
   - `https://www.nature.com/articles/s41598-025-00463-y`

## Explicit Non-Goals

Not in scope for this round:

- full hierarchical multi-policy training
- GNN commander
- MAPPO / Transformer migration
- self-play league / PFSP infra
- full continuous control rewrite

If this medium upgrade still fails the `fix_rule` gate, the next serious step should be a control-paradigm rewrite rather than more reward tweaking.

## Run Log — 2026-04-14 Library Gate

### Launch profile

- Launcher: `scripts/run_maca_4060_library_long.sh`
- Experiment: `sf_maca_4060_library_gate_20260414`
- Gate override:
  - `PHASE1_SECONDS=300`
  - `PHASE2_CYCLES=1`
  - `PHASE2_PULSE_SECONDS=180`
  - `PHASE2_MAIN_SECONDS=1200`
  - `PHASE3_SECONDS=900`
  - `TOTAL_ENV_STEPS=12000000`
  - `MIN_PHASE_ENV_STEP_HEADROOM=800000`

### First runtime failure

- Initial launch failed before training started because sandboxed execution blocked PyTorch shared-memory setup:
  - `torch_shm_manager ... Operation not permitted`
- This was an execution-environment issue, not a MaCA code-path issue.

### Second runtime failure (real code bug)

- After rerunning outside the sandbox, rollout workers failed immediately with:
  - `KeyError: 'course'`
- Root cause:
  - `marl_env/maca_parallel_env.py` used `self._last_red_obs["fighter"][idx]` inside `_decode_red_actions`.
  - `self._last_red_obs` stores processed observations (`info/screen/alive`), not raw fighter dictionaries.
  - Delta-course control and adaptive support logic therefore read the wrong structure.

### Fix applied

- Added raw-observation caches:
  - `self._last_red_raw_obs`
  - `self._last_blue_raw_obs`
- `_collect_step_output()` now stores raw obs separately.
- `_decode_red_actions()` now reads raw fighter state from:
  - `self._last_red_raw_obs["fighter_obs_list"][idx]`
- Post-fix smoke test passed with:
  - `adaptive_support_policy=True`
  - `delta_course_action=True`
  - two environment steps without exception

### Current status

- Relaunched the same library gate after the fix.
- Training is now progressing normally through the curriculum and writing checkpoints under:
  - `train_dir/sample_factory/sf_maca_4060_library_gate_20260414/`
- Logs:
  - `log/sf_maca_4060_library_gate_20260414.phase1_noatt_warmup.log`
  - `log/sf_maca_4060_library_gate_20260414.phase2_pulse_noatt_c1.log`
  - `log/sf_maca_4060_library_gate_20260414.phase2_fixrule_c1.log`
  - `log/sf_maca_4060_library_gate_20260414.phase3_fixrule_consolidate.log`

### Mid-run evaluation snapshots

- Pre-`fix_rule`-main checkpoint (`checkpoint_000006471_828672.pth`):
  - `fix_rule`: `win_rate=0.0`, `contact=0.0186`, `nearest_enemy_distance_mean=110.8`
  - `fix_rule_no_att`: `win_rate=0.8`, `contact=0.0348`, `nearest_enemy_distance_mean=51.4`
  - files:
    - `log/sf_maca_4060_library_gate_20260414.eval5.fix_rule.midrun_cpu.json`
    - `log/sf_maca_4060_library_gate_20260414.eval5.fix_rule_no_att.midrun_cpu.json`

- After substantial `fix_rule` training (`checkpoint_000011003_1377190.pth`):
  - `fix_rule`: `win_rate=0.0`, `contact=0.0246`, `nearest_enemy_distance_mean=112.9`
  - `fix_rule_no_att`: `win_rate=1.0`, `contact=0.0278`, `nearest_enemy_distance_mean=56.0`
  - files:
    - `log/sf_maca_4060_library_gate_20260414.eval5.fix_rule.fixrule_phase_midrun_cpu.json`
    - `log/sf_maca_4060_library_gate_20260414.eval5.fix_rule_no_att.fixrule_phase_midrun_cpu.json`

Audit interpretation at this point:

- `fix_rule` still had **zero wins** after meaningful `fix_rule` exposure.
- `executed_fire` remained close to `attack_opportunity`; the model still did not look like “has windows but refuses to shoot”.
- `contact` rose modestly versus the older audited baselines, but `nearest_enemy_distance_mean` stayed bad and did not convert into wins.
- `fix_rule_no_att` remained strong, so the transfer gap persisted.

### Final saved checkpoint and stop decision

- Latest saved checkpoint after stopping the run:
  - `train_dir/sample_factory/sf_maca_4060_library_gate_20260414/checkpoint_p0/checkpoint_000021302_2624574.pth`
- Final 10-episode eval:
  - `fix_rule`:
    - `win_rate=0.0`
    - `contact=0.0147`
    - `attack_opportunity=0.00098`
    - `executed_fire=0.00081`
    - `missed_attack=0.00017`
    - `episode_len_mean=849.6`
    - `opponent_round_reward_mean=4600.0`
    - `nearest_enemy_distance_mean=130.4`
    - `course_change=0.179`
    - file: `log/sf_maca_4060_library_gate_20260414.eval10.fix_rule.final_cpu.json`
  - `fix_rule_no_att`:
    - `win_rate=1.0`
    - `contact=0.0305`
    - `attack_opportunity=0.00132`
    - `executed_fire=0.00119`
    - `missed_attack=0.00013`
    - `episode_len_mean=1000.0`
    - `opponent_round_reward_mean=-1000.0`
    - `nearest_enemy_distance_mean=48.2`
    - `course_change=0.299`
    - file: `log/sf_maca_4060_library_gate_20260414.eval10.fix_rule_no_att.final_cpu.json`

Why the run was stopped manually:

- The final checkpoint still provided **no evidence of effectiveness on the main target `fix_rule`**.
- The `fix_rule` metrics regressed from the mid-run checkpoint:
  - `contact`: `0.0246 -> 0.0147`
  - `nearest_enemy_distance_mean`: `112.9 -> 130.4`
- The run therefore looked like another case of “auxiliary scenario remains good, main target still not solved”.
- Continuing to burn the library profile after this point was not justified.

### Visual observation note

User-side visual inspection of a rendered `fix_rule` episode:

- The red fighters no longer showed the previous tight high-frequency spin.
- The turning radius became visibly larger, which is consistent with the added course hold / smoothing bias.
- There were occasional successful kills, so the policy is not “completely blind” or “never fires”.
- However, the user did **not** observe stable proactive intercept / closing behavior.

Audit interpretation:

- This visual evidence is consistent with the metrics:
  - smoothing reduced pathological jitter
  - opportunistic kills can still happen
  - but stable pursuit /主动接敌 still does not emerge
- Therefore the current medium upgrade appears to improve **local motion smoothness** more than **global intercept behavior**.

## Intercept-Focused Follow-up — 2026-04-14

Based on the failed library gate plus the visual inspection, the next medium-scope change is **not** more generic smoothing. It is an intercept-focused upgrade targeted at the still-missing “主动接敌” behavior.

### Why this follow-up is necessary

- The previous medium round reduced tight spinning, but `fix_rule` still failed with:
  - weak contact
  - poor closure
  - no stable intercept chain
- Visual inspection confirmed the policy can sometimes kill opportunistically, but still does not deliberately close distance.

### What changed in code

1. Added lock-state measurements in `marl_env/sample_factory_env.py`
   - fresh-track flag
   - lost-track age
   - pursuit-failure streak
   - closure-health flag

2. Added intercept guidance logic in `marl_env/sample_factory_env.py`
   - pursue mode: stronger guidance toward target bearing, with extra lead when bearing keeps moving and closure is poor
   - reacquire mode: stronger pull toward the last remembered bearing after contact loss
   - search-with-receive mode: mild heading guidance from jammer receive direction when there is no direct contact

3. Added optional course-assist postprocessing in `marl_env/sample_factory_env.py`
   - blend the policy course command toward the heuristic intercept bin during pursue / reacquire / receive-guided search
   - allow strong intercept disagreement to break short course-hold smoothing

4. Added new config surface in `marl_env/sample_factory_registration.py`
   - `maca_lock_state_observation`
   - `maca_intercept_course_assist`
   - `maca_intercept_course_blend`
   - `maca_intercept_break_hold_bins`
   - `maca_intercept_lead_deg`

5. Enabled the new intercept-focused defaults in:
   - `scripts/run_sf_maca_decoupled_8h_curriculum.sh`
   - `scripts/run_sf_maca_radar_support_8h_curriculum.sh`

### Validation

- Syntax:
  - `python -m py_compile marl_env/sample_factory_registration.py marl_env/sample_factory_env.py`
- Shell:
  - `bash -n scripts/run_sf_maca_decoupled_8h_curriculum.sh scripts/run_sf_maca_radar_support_8h_curriculum.sh`
- Env smoke:
  - stepped the env for 3 steps with
    - decoupled heads
    - relative motion
    - lock-state observation
    - intercept course assist
  - result: `SMOKE_OK`

### Intended effect

- keep the already-achieved reduction in local jitter
- add stronger structure for:
  - closing on a visible target
  - turning back toward a recently lost target
  - moving toward receive cues before direct contact exists

### Risk

- This is more assertive than the previous “soft prior only” round.
- If it helps `contact` but still does not improve `nearest_enemy_distance_mean` or `win_rate` on `fix_rule`, then the remaining blocker is likely the control paradigm itself rather than observation shaping alone.

## 4060 Full-Performance Run — 2026-04-14

- Launcher:
  - `scripts/run_maca_4060_overnight.sh`
- Experiment:
  - `sf_maca_4060_fullperf_20260414_intercept`
- Start mode:
  - fresh start
  - 4060 full-performance / overnight profile
  - intercept-focused defaults enabled

Observed startup checks:

- shared-memory allocation succeeded
- learner initialized normally
- policy worker initialized normally
- 4 actor workers reset successfully
- first checkpoint written:
  - `train_dir/sample_factory/sf_maca_4060_fullperf_20260414_intercept/checkpoint_p0/checkpoint_000000001_1024.pth`

Primary live log:

- `log/sf_maca_4060_fullperf_20260414_intercept.phase1_noatt_warmup.log`

## 4060 Full-Performance Run Relaunch — 2026-04-14

- Previous conservative 4060 overnight run was stopped after confirming it did not match the expected “yesterday” throughput / VRAM profile.
- `scripts/run_maca_4060_overnight.sh` was tightened to a saturated 4060 profile:
  - `NUM_WORKERS=10`
  - `BATCH_SIZE=6400`
  - `PPO_EPOCHS=4`
  - `HIDDEN_SIZE=256`
  - `LEARNER_MAIN_LOOP_NUM_CORES=3`
  - `TRAJ_BUFFERS_EXCESS_RATIO=4.0`
  - `MAX_POLICY_LAG=12`

### Relaunch

- Experiment:
  - `sf_maca_4060_fullperf_20260414_intercept_max`
- Mode:
  - fresh start
  - 4060 overnight saturated profile
  - intercept-focused medium upgrade enabled

### Startup evidence

- phase1 log:
  - `log/sf_maca_4060_fullperf_20260414_intercept_max.phase1_noatt_warmup.log`
- first checkpoint:
  - `train_dir/sample_factory/sf_maca_4060_fullperf_20260414_intercept_max/checkpoint_p0/checkpoint_000000001_6400.pth`
- trajectory buffers:
  - log shows `Using a total of 840 trajectory buffers`

### Throughput evidence

- early warmup:
  - `10 sec FPS = 2559.4` at 22:58:52
- after startup settles:
  - `10 sec FPS = 3794.1` at 22:59:58
  - `10 sec FPS = 3828.0` at 23:00:08
  - `10 sec FPS = 3817.3` at 23:00:13
  - `10 sec FPS = 3770.5` at 23:00:23
  - `10 sec FPS = 3785.7` at 23:00:48
  - `10 sec FPS = 3806.2` at 23:00:58
- smoothed throughput:
  - `60 sec FPS = 3692.3` by 23:00:58

### GPU evidence

- `nvidia-smi` outside sandbox:
  - memory used `7612 MiB / 8188 MiB`
  - GPU utilization `100%`

### Current audit read

- This relaunch does recover the expected hardware operating point much better than the conservative overnight profile.
- It does **not** yet say anything about `fix_rule` effectiveness; only that the 4060 is now being used close to saturation.

## Current Mid-Run Audit — 2026-04-15

### Training progress

- `phase1_noatt_warmup` finished normally at `2026-04-14 23:28:39` after collecting `6,566,400` frames.
- `phase2_pulse_noatt_c1` finished normally at `2026-04-15 00:08:53` after collecting `15,046,400` frames total.
- `phase2_fixrule_c1` is the current meaningful phase. Latest visible log line at `2026-04-15 00:13:38` shows:
  - `Total num frames: 15,649,726`
  - `10 sec FPS: 2209.1`
  - `60 sec FPS: 2283.9`
  - `Avg episode reward: -2324.398`

### Current evaluation checkpoint

- Eval loaded:
  - `train_dir/sample_factory/sf_maca_4060_fullperf_20260414_intercept_max/checkpoint_p0/checkpoint_000009364_15059200.pth`

### Current `fix_rule` eval (5 episodes, CPU)

- Output:
  - `log/sf_maca_4060_fullperf_20260414_intercept_max.eval5.fix_rule.current_20260415_0015.json`
- Summary:
  - `win_rate = 0.0`
  - `round_reward_mean = -1800.0`
  - `opponent_round_reward_mean = 6640.0`
  - `fire_action_frac_mean = 0.001296`
  - `executed_fire_action_frac_mean = 0.001296`
  - `attack_opportunity_frac_mean = 0.001439`
  - `missed_attack_frac_mean = 0.000143`
  - `contact_frac_mean = 0.02177`
  - `visible_enemy_count_mean = 0.02429`
  - `nearest_enemy_distance_mean = 111.54`
  - `course_change_frac_mean = 0.18237`
  - `course_unique_frac_mean = 0.90125`
  - `engagement_progress_reward_mean = 0.0`
  - `episode_len_mean = 700.6`

### Current `fix_rule_no_att` eval (5 episodes, CPU)

- Output:
  - `log/sf_maca_4060_fullperf_20260414_intercept_max.eval5.fix_rule_no_att.current_20260415_0015.json`
- Summary:
  - `win_rate = 1.0`

## Takeover Audit + Medium Upgrade v2 — 2026-04-15

### Workload

- audited code path:
  - `marl_env/maca_parallel_env.py`
  - `marl_env/sample_factory_env.py`
  - `marl_env/sample_factory_registration.py`
  - `marl_env/runtime_tweaks.py`
  - `scripts/train_sf_maca.py`
  - `scripts/eval_sf_maca.py`
  - `scripts/run_sf_maca_decoupled_8h_curriculum.sh`
  - `scripts/run_sf_maca_radar_support_8h_curriculum.sh`
  - `scripts/run_maca_4060_library_long.sh`
- audited runtime artifacts:
  - `train_dir/sample_factory/sf_maca_4060_commit_fire_20260415/cfg.json`
  - `log/sf_maca_4060_commit_fire_20260415.phase2_fixrule_c1.library_resume_20260415_1554.log`
  - `log/sf_maca_4060_commit_fire_20260415.eval5.fix_rule.poststop_20260415_1927.json`
  - `log/sf_maca_4060_commit_fire_20260415.eval5.fix_rule_no_att.poststop_20260415_1927.json`

### Audit Conclusion

Main conclusion after full takeover audit:

1. `fix_rule_no_att` is good because the current policy can already form contact, close distance, and survive when the opponent does not force a defensive geometry problem.
2. `fix_rule` stays at `win_rate=0.0` because the policy still lacks armed-opponent survival structure:
   - contact is lower: `0.1078 -> 0.0330`
   - nearest distance is much worse: `54.21 -> 116.36`
   - end-state attrition collapses: `fighter_destroy_balance_end_mean = +6.2 -> -5.0`
3. `executed_fire ≈ attack_opportunity` is real, but it does **not** mean the combat policy is good.
   - It only says that once a legal fire window exists, the policy usually takes it.
   - It says nothing about whether the policy can:
     - survive long enough to keep the geometry
     - maintain closure
     - avoid losing 6-10 fighters before the few fire windows appear

### Failure Breakdown From First Principles

#### 1. Perception is not the primary blocker, but still incomplete

- The agent is not blind:
  - `contact_frac_mean = 0.0330` on `fix_rule`
  - occasional kills still happen
- But the repo had only injected team alive/destroyed counts into env outputs; before this round those counts were **not** appended into policy measurements, so the policy could not condition on attrition state even though the env already computed it.

Assessment:

- perception issue exists, but is secondary
- relative importance: `感知 < 追击控制 < 威胁处理/生存`

#### 2. Pursuit / closure continuity is still weak

- `fix_rule` post-stop:
  - `nearest_enemy_distance_mean = 116.36`
  - `engagement_progress_reward_mean = 0.0`
  - `course_change_frac_mean = 0.204`
  - `course_unique_frac_mean = 0.9125`
- `fix_rule_no_att` post-stop:
  - `nearest_enemy_distance_mean = 54.21`
  - `contact_frac_mean = 0.1078`

Interpretation:

- smoothing did reduce pathological spin compared with older runs
- but a high `course_unique_frac` together with zero progress reward means the policy still explores many headings without turning those choices into stable closure
- current course assist is mostly intercept-pursue biased; under armed opposition it often commits without enough survival logic

Assessment:

- pursuit/control is a major blocker
- relative importance: high

#### 3. Threat handling / survival is the dominant blocker on `fix_rule`

- The sharpest split between `fix_rule_no_att` and `fix_rule` is not fire behavior:
  - `executed_fire ≈ attack_opportunity` in both
- The split is:
  - contact collapses
  - distance stays large
  - red losses explode
  - blue survives in larger numbers
- `fix_rule` end state:
  - `red_fighter_destroyed_end_mean = 9.0`
  - `blue_fighter_destroyed_end_mean = 4.0`
- `fix_rule_no_att` end state:
  - `red_fighter_destroyed_end_mean = 0.0`
  - `blue_fighter_destroyed_end_mean = 6.2`

Interpretation:

- armed opponent pressure is breaking the red side before pursuit can mature into repeated legal attack windows
- the current policy has weak state about:
  - current attrition disadvantage
  - received threat level
  - when to beam/disengage instead of blindly recommitting

Assessment:

- threat handling / survival is the dominant blocker
- relative importance: highest

### Which Reward Improvements Were Likely Fake

- high or less-bad shaped reward during training was partly fake because:
  - `contact_reward`
  - `progress_reward`
  - `attack_window_reward`
  can improve without producing winning attrition
- this is directly visible because `engagement_progress_reward_mean` stayed `0.0` in final evals while total reward sometimes looked less bad in training logs
- `fix_rule_no_att` can also inflate confidence:
  - it rewards cleaner intercept behavior in a much easier regime
  - it does not validate armed-opponent survival

Therefore the reliable acceptance metrics remain:

- `win_rate`
- `fighter_destroy_balance_end`
- `red_fighter_destroyed_end`
- `blue_fighter_destroyed_end`
- `contact`
- `nearest_enemy_distance`

with `fix_rule` as the only main gate.

### Modifications Applied In This Round

1. Team status is now actually consumed by the policy
   - Added `maca_team_status_observation`
   - Appends:
     - red alive
     - blue alive
     - red destroyed
     - blue destroyed
     - attrition balance

2. Threat / defensive state is now observed explicitly
   - Added `maca_threat_state_observation`
   - Appends:
     - receive intensity
     - threat-active flag
     - team disadvantage
     - defensive-mode flag
     - low-ammo-under-threat flag

3. Defensive course assistance was added
   - Added `maca_defensive_course_assist`
   - Added beam/disengage guidance from receive direction when:
     - threat exists
     - no attack opportunity exists
     - closure is poor or attrition is losing
   - Defensive guidance can break short course holds, because old hold logic was too eager to preserve a bad heading under threat.

4. Attrition shaping was added and aligned to the main goal
   - Added runtime tweaks:
     - `MACA_FRIENDLY_ATTRITION_PENALTY`
     - `MACA_ENEMY_ATTRITION_REWARD`
   - This directly rewards enemy kills and penalizes friendly losses during the episode instead of over-trusting contact-like proxies.

5. Low-power curriculum defaults were tightened toward `fix_rule`
   - reduced proxy-heavy shaping defaults:
     - `MACA_CONTACT_REWARD: 20 -> 8`
     - `MACA_PROGRESS_REWARD_SCALE: 0.5 -> 0.2`
     - `MACA_ATTACK_WINDOW_REWARD: 30 -> 12`
   - enabled:
     - team-status observation
     - threat-state observation
     - defensive course assist
   - enabled attrition shaping defaults:
     - `MACA_FRIENDLY_ATTRITION_PENALTY=220`
     - `MACA_ENEMY_ATTRITION_REWARD=160`

### Failure Reasons Addressed By The Changes

- previous code exposed alive/destroyed counts only in env outputs and episode stats, not in policy measurements
- previous guidance logic had pursue / reacquire / search-emit, but no explicit defensive branch for armed-opponent pressure
- previous shaping overpaid proxy progress and underpaid actual attrition

### Checkpoint Compatibility

- This round increases fighter measurement dim from `30` to `40`
- Existing checkpoints for old observation shape are therefore not trusted as compatible
- Decision:
  - `fresh start` for the new low-power training lane

### Validation

- `python -m py_compile marl_env/runtime_tweaks.py marl_env/sample_factory_registration.py marl_env/sample_factory_env.py scripts/train_sf_maca.py scripts/eval_sf_maca.py`
- `bash -n scripts/run_sf_maca_decoupled_8h_curriculum.sh scripts/run_sf_maca_radar_support_8h_curriculum.sh scripts/run_maca_4060_library_long.sh scripts/run_maca_4060_overnight.sh`
- smoke in training conda env:
  - `SMOKE_OK 10 40 1`

### References / Evidence Sources

- repo code audited above
- `cfg.json` for `sf_maca_4060_commit_fire_20260415`
- final post-stop `fix_rule` and `fix_rule_no_att` eval JSONs
- resume log tail proving the run stopped at the low-power env-step cap rather than converging early

## Observation Slimdown Audit — 2026-04-15

### Conclusion

- yes, lowering the current observation dimension is necessary
- not because “40 dims is always too big”, but because the current 40-dim fighter measurement contained multiple layers of **derived duplicate state**
- the main issue was not raw dimensionality alone; it was that several states were encoded two or three times in different wrappers:
  - direct geometry / contact quantities
  - tactical one-hot wrappers
  - lock-state wrappers
  - threat-state wrappers

This increases optimization burden without adding equivalent new information.

### Why dim reduction is justified from the code

Current measurement before slimdown was:

- base = `7`
- extended = `3`
- radar tracking = `9`
- relative motion = `3`
- tactical mode = `4`
- lock state = `4`
- team status = `5`
- threat state = `5`
- total = `40`

After reading the code, the strongest redundancy points were:

1. `tactical_mode_observation`
   - pure one-hot derived from:
     - contact
     - attack opportunity
     - track freshness
   - almost no new information

2. `extended_observation`
   - `visible_enemy_count/contact/attack_opportunity`
   - overlaps with:
     - radar/track features
     - action mask / attack prior

3. `radar_tracking_observation`
   - previously mixed:
     - track bearing
     - track distance
     - freshness
     - recv count
     - recv flag
     - recv direction
     - dominant frequency
   - several of these were repeated again in lock/threat features

4. `team_status_observation`
   - alive counts were redundant because team size is fixed, so alive can be recovered from destroyed counts

5. `threat_state_observation`
   - binary threat-active and low-ammo-under-threat were secondary wrappers around raw receive and ammo state

### Targeted slimdown applied

Principle:

- keep states that are hard to infer and directly useful for `fix_rule`
- remove wrapper features that re-encode already-visible state

Kept:

- base fighter state
- track bearing / distance
- receive intensity + receive direction
- relative motion
- lock-loss / pursuit-failure / closure-health
- attrition status
- threat/disadvantage/defensive-mode

Removed or slimmed:

1. default-off:
   - `MACA_EXTENDED_OBSERVATION=False`
   - `MACA_TACTICAL_MODE_OBSERVATION=False`

2. `radar_tracking_observation`
   - `9 -> 6`
   - removed duplicated wrapper terms, kept only:
     - track bearing sin/cos
     - track distance
     - recv intensity
     - recv direction sin/cos

3. `lock_state_observation`
   - `4 -> 3`
   - removed fresh-track flag
   - kept:
     - lost-track age
     - pursuit-fail streak
     - closure-health flag

4. `team_status_observation`
   - `5 -> 3`
   - removed alive counts
   - kept:
     - red destroyed
     - blue destroyed
     - attrition balance

5. `threat_state_observation`
   - `5 -> 3`
   - kept:
     - recv intensity
     - team disadvantage
     - defensive-mode flag

### Resulting dimension

- new default fighter measurement:
  - `25`
- plus `is_alive` is still passed separately in the observation dict, so the model input path remains valid
- validated by smoke:
  - `SMOKE_OK 10 25 1`

### Why this slimdown is considered safe

- it does **not** remove the new `fix_rule`-critical information introduced in the previous round:
  - team attrition
  - threat pressure
  - defensive mode
- it removes mostly repeated encodings and lightweight heuristic wrappers
- this should reduce learning burden more safely than deleting geometry, closure, or attrition state directly

### Training switch caused by slimdown

- old run stopped:
  - `sf_maca_4060_fixrule_attrition_20260415`
- reason:
  - it still used the pre-slim 40-dim observation
- new fresh-start low-power run launched:
  - `sf_maca_4060_fixrule_slimobs_20260415`
- startup evidence:
  - runtime now shows:
    - `maca_extended_observation=False`
    - `maca_tactical_mode_observation=False`
  - run is actively training with FPS around `1.14k~1.20k`

### Validation

- `python -m py_compile marl_env/sample_factory_env.py marl_env/sample_factory_registration.py marl_env/runtime_tweaks.py`
- `bash -n scripts/run_sf_maca_radar_support_8h_curriculum.sh scripts/run_sf_maca_decoupled_8h_curriculum.sh scripts/run_maca_4060_library_long.sh`
- conda smoke:
  - `SMOKE_OK 10 25 1`

### Acceptance focus after slimdown

This slimdown is only justified if `fix_rule` later improves on:

- `red_fighter_destroyed_end`
- `blue_fighter_destroyed_end`
- `fighter_destroy_balance_end`
- `contact`
- `nearest_enemy_distance`
- `win_rate`

If reward becomes cleaner but the attrition metrics stay bad, then the slimdown alone is not enough and the next bottleneck remains policy/control rather than observation load.
  - `round_reward_mean = 1200.0`
  - `opponent_round_reward_mean = -1000.0`
  - `fire_action_frac_mean = 0.00144`
  - `executed_fire_action_frac_mean = 0.00144`
  - `attack_opportunity_frac_mean = 0.00148`
  - `missed_attack_frac_mean = 0.00004`
  - `contact_frac_mean = 0.04564`
  - `visible_enemy_count_mean = 0.05234`
  - `nearest_enemy_distance_mean = 46.44`
  - `course_change_frac_mean = 0.31145`
  - `course_unique_frac_mean = 0.90250`
  - `engagement_progress_reward_mean = 0.0`
  - `episode_len_mean = 1000.0`

### Audit read

- This run is clearly better than the earlier library gate on intermediate behavior:
  - `fix_rule` contact increased from about `0.0147` to `0.0218`
  - `fix_rule` nearest-enemy distance improved from about `130.4` to `111.5`
  - `fix_rule` course-change fraction dropped from about `0.75` to `0.18`
- However, the main target is still not met:
  - `fix_rule` win rate remains `0.0`
  - `engagement_progress_reward_mean` is still `0.0`
  - `opponent_round_reward_mean` is still extremely bad at `6640.0`
- Current read:
  - the medium upgrade is reducing wasteful turning and slightly improving contact / proximity
  - but it has **not yet produced proactive winning behavior against `fix_rule`**

### Overnight decision

- Decision at `2026-04-15` before sleep:
  - allow this run to continue overnight
- Reason:
  - the current `fix_rule` phase is still producing non-zero intermediate improvement
  - there is still no win evidence, but it is not yet a dead line like the earlier recovery branch
- Next required check:
  - run a broader `fix_rule` audit after the overnight segment completes before making any “effective on main target” claim

### Overnight stop-risk note

- There is no immediate sign of the old “almost no training happened and the script exited” failure mode.
- Why:
  - the curriculum script now checks phase headroom before each phase and refuses to enter a phase only when remaining env-step budget is below `MIN_PHASE_ENV_STEP_HEADROOM`
  - this run uses `TOTAL_ENV_STEPS=90,000,000` with `MIN_PHASE_ENV_STEP_HEADROOM=2,500,000`
  - current progress is only about `16.3M` frames, so env-step headroom is still very large
- Current likely stop conditions are therefore:
  - normal scheduled completion of the overnight curriculum
  - an unexpected runtime failure outside the known cumulative-seconds / env-step trap
- Expected scheduled end time for the current launcher is roughly `2026-04-15 06:48` Asia/Shanghai if it keeps running without crashes.

## Final Overnight Audit — 2026-04-15

### Completion evidence

- The overnight run is finished. There are no live training processes for `sf_maca_4060_fullperf_20260414_intercept_max`.
- Final saved checkpoint:
  - `train_dir/sample_factory/sf_maca_4060_fullperf_20260414_intercept_max/checkpoint_p0/checkpoint_000058743_90008272.pth`
- Final completed log segment:
  - `log/sf_maca_4060_fullperf_20260414_intercept_max.phase2_pulse_noatt_c3.log`
  - collected `90,001,872` frames and exited cleanly

### Important schedule finding

- The run **did not** finish on a `fix_rule` segment.
- The last completed `fix_rule` block was `phase2_fixrule_c2`, which ended at checkpoint:
  - `train_dir/sample_factory/sf_maca_4060_fullperf_20260414_intercept_max/checkpoint_p0/checkpoint_000051201_77918672.pth`
- After that, the run entered `phase2_pulse_noatt_c3` and used up the remaining env-step budget.
- There is no `phase2_fixrule_c3` log and no `phase3_fixrule_consolidate` log.
- So this overnight run again has a curriculum-design issue:
  - the experiment terminated on a helper opponent phase instead of a final `fix_rule` consolidation phase

### Final eval of latest checkpoint (`checkpoint_000058743_90008272.pth`)

- `fix_rule` 20 episodes:
  - output: `log/sf_maca_4060_fullperf_20260414_intercept_max.eval20.fix_rule.final_20260415.json`
  - `win_rate = 0.0`
  - `round_reward_mean = -1750.0`
  - `opponent_round_reward_mean = 6300.0`
  - `fire_action_frac_mean = 0.001872`
  - `executed_fire_action_frac_mean = 0.001872`
  - `attack_opportunity_frac_mean = 0.001954`
  - `missed_attack_frac_mean = 0.000081`
  - `contact_frac_mean = 0.03143`
  - `visible_enemy_count_mean = 0.04319`
  - `nearest_enemy_distance_mean = 110.09`
  - `course_change_frac_mean = 0.18527`
  - `course_unique_frac_mean = 0.81187`
  - `engagement_progress_reward_mean = 0.0`
  - `episode_len_mean = 516.3`

- `fix_rule_no_att` 20 episodes:
  - output: `log/sf_maca_4060_fullperf_20260414_intercept_max.eval20.fix_rule_no_att.final_20260415.json`
  - `win_rate = 1.0`
  - `round_reward_mean = 4600.0`
  - `opponent_round_reward_mean = -1500.0`
  - `fire_action_frac_mean = 0.003664`
  - `executed_fire_action_frac_mean = 0.003664`
  - `attack_opportunity_frac_mean = 0.003907`
  - `missed_attack_frac_mean = 0.000243`
  - `contact_frac_mean = 0.12175`
  - `visible_enemy_count_mean = 0.14046`
  - `nearest_enemy_distance_mean = 85.40`
  - `course_change_frac_mean = 0.33405`
  - `course_unique_frac_mean = 0.88656`
  - `engagement_progress_reward_mean = 0.0`
  - `episode_len_mean = 878.75`

### Final read

- Relative to the earlier short library run, the model has improved intermediate `fix_rule` behavior:
  - more contact
  - slightly closer nearest-enemy distance
  - much lower course-change rate
- But the main target is still unmet:
  - `fix_rule` win rate is still `0.0 / 20`
  - `engagement_progress_reward_mean` is still exactly `0.0`
  - `opponent_round_reward_mean` is still heavily in the opponent’s favor
- So the medium upgrade produced a cleaner and somewhat more stable policy, but **still did not produce a winning `fix_rule` policy**.

## GUI Visual Audit Follow-up — 2026-04-15

### Visual findings

- In rendered `fix_rule` episodes, the policy now sometimes rotates while translating toward the enemy side instead of only orbiting in place.
- In some first-contact sequences, at least part of the formation can perform a short follow-up chase after the initial merge and occasionally convert that into a kill.
- The main failure remains:
  - most episodes still do not secure the first effective shot before the opponent
  - pursuit exists in small bursts but is not stable enough to dominate the engagement

### Interpretation

- This is consistent with the quantitative audit:
  - intercept / heading behavior is improved versus the older library run
  - but the model still does not create enough early, stable attack geometry against `fix_rule`
- Therefore the current blocker is no longer “pure random spinning”.
- The more precise remaining issue is:
  - partial intercept has emerged
  - first-wave closure / shot timing / pursuit continuity are still insufficient

### Next adjustment direction

- Do **not** pivot back to generic smoothing-only work.
- The next medium changes should focus on:
  - stronger pre-merge commit toward armed opponents
  - better persistence after first contact loss or first merge crossing
  - curriculum repair so training ends on `fix_rule`, not `fix_rule_no_att`

## Commit-and-Fire Follow-up — 2026-04-15

### Why this change

- Visual audit shows the policy is no longer purely spinning; partial intercept and short post-merge chase have emerged.
- At this stage, “fire earlier when the first valid shot opens” does matter more than before.
- But the evidence still says early fire is **not** the only issue:
  - `executed_fire` remains very close to `attack_opportunity`
  - so the remaining bottleneck is still mostly “get to the right geometry more often, then commit to the shot immediately”

### Code changes

1. Stronger hard-commit pursuit in `marl_env/sample_factory_env.py`
   - added `commit_distance`
   - added `commit_course_blend`
   - visible pursue states inside the commit distance now get more aggressive course pull
   - hard-commit guidance can break course hold immediately when needed

2. New fire-prior observation in `marl_env/sample_factory_env.py`
   - added `attack_prior`
   - when an attack opportunity exists, fire logits now receive a heuristic urgency prior
   - urgency is stronger for closer targets, fresh opportunities, and weak-closure situations

3. Action-logit patch updated in:
   - `scripts/train_sf_maca.py`
   - `scripts/eval_sf_maca.py`
   - both now consume `attack_prior` in addition to `course_prior`

4. New config surface in `marl_env/sample_factory_registration.py`
   - `maca_commit_distance`
   - `maca_commit_course_blend`
   - `maca_attack_prior_strength`

5. Training defaults adjusted in wrappers
   - `scripts/run_sf_maca_radar_support_8h_curriculum.sh`
   - `scripts/run_maca_4060_overnight.sh`
   - the 4060 overnight profile is now more `fix_rule`-heavy and is intended to finish on `fix_rule`, not on a helper phase

### Validation

- `python -m py_compile marl_env/sample_factory_registration.py marl_env/runtime_tweaks.py marl_env/sample_factory_env.py scripts/train_sf_maca.py scripts/eval_sf_maca.py`
- `bash -n scripts/run_sf_maca_decoupled_8h_curriculum.sh scripts/run_sf_maca_radar_support_8h_curriculum.sh scripts/run_maca_4060_overnight.sh`
- one-episode eval smoke on an old checkpoint loaded successfully after the observation-surface change

### New training run started

- Experiment:
  - `sf_maca_4060_commit_fire_20260415`
- Launcher:
  - `scripts/run_maca_4060_overnight.sh`
- Mode:
  - fresh start
  - 4060 saturated profile
  - fix-rule-heavy repaired curriculum
  - commit-and-fire medium upgrade enabled

### Startup evidence

- runtime tweaks now show:
  - `attack_prior_strength = 1.1`
  - `attack_window_reward = 30.0`
  - `contact_reward = 20.0`
  - `progress_reward_scale = 0.5`
- first checkpoint written:
  - `train_dir/sample_factory/sf_maca_4060_commit_fire_20260415/checkpoint_p0/checkpoint_000000001_6400.pth`

### Mid-run quick audit — 2026-04-15 13:34

- checkpoint under audit:
  - `train_dir/sample_factory/sf_maca_4060_commit_fire_20260415/checkpoint_p0/checkpoint_000004641_7449600.pth`
- training stage context:
  - this checkpoint was still before any `fix_rule` training phase completed
  - it came right after `phase2_pulse_noatt_c1`, so any `fix_rule` improvement claim must stay provisional

#### `fix_rule` 5-episode quick eval

- source:
  - `log/sf_maca_4060_commit_fire_20260415.eval5.fix_rule.current_20260415_1333.json`
- summary:
  - `win_rate = 0.0`
  - `round_reward_mean = -1300`
  - `opponent_round_reward_mean = 2020`
  - `contact_frac_mean = 0.0320`
  - `visible_enemy_count_mean = 0.0363`
  - `fire_action_frac_mean = 0.001605`
  - `executed_fire_action_frac_mean = 0.001605`
  - `attack_opportunity_frac_mean = 0.001645`
  - `missed_attack_frac_mean = 0.00004`
  - `nearest_enemy_distance_mean = 112.95`
  - `nearest_enemy_distance_min = 89.59`
  - `course_change_frac_mean = 0.191`
  - `course_unique_frac_mean = 0.905`
  - `engagement_progress_reward_mean = 0.0`
  - `episode_len_mean = 996.6`

#### `fix_rule_no_att` 5-episode quick eval

- source:
  - `log/sf_maca_4060_commit_fire_20260415.eval5.fix_rule_no_att.current_20260415_1333.json`
- summary:
  - `win_rate = 1.0`
  - `round_reward_mean = 1200`
  - `contact_frac_mean = 0.0327`
  - `nearest_enemy_distance_mean = 35.01`
  - `course_change_frac_mean = 0.317`

#### Audit interpretation

- there is a mild positive sign on `fix_rule` relative to earlier weak checkpoints:
  - `opponent_round_reward_mean` is much lower than the previous `6640` quick audit
  - first-pass `contact` is not collapsing
  - `executed_fire` remains very close to `attack_opportunity`, so the main failure is still not "refusing to shoot"
- however, the main target is still not met:
  - `win_rate` remains `0.0`
  - `engagement_progress_reward_mean` remains `0.0`
  - `nearest_enemy_distance_mean` is still around `113`, which is not evidence of stable active closure
- current judgment:
  - `yes, there are preliminary signs of improvement`
  - `no, there is still no valid evidence that this run has solved or materially beaten fix_rule`

### On training length vs. direction — 2026-04-15

- the question "is it simply too few training iterations?" has two different answers depending on which run is being discussed.

#### For the current `commit_fire` run

- yes, it is still early:
  - the latest audited checkpoint is only at about `7.45M` env steps
  - current live training is only around `9.4M` env steps and has just entered `phase2_fixrule_c1`
  - source:
    - `train_dir/sample_factory/sf_maca_4060_commit_fire_20260415/checkpoint_p0/checkpoint_000004641_7449600.pth`
    - `log/sf_maca_4060_commit_fire_20260415.phase2_fixrule_c1.log`
- therefore this run should not be judged final yet.

#### For the overall project direction so far

- no, "too few iterations" is not a sufficient explanation by itself.
- evidence:
  - one prior line reached `90,008,272` env steps and still ended with:
    - `fix_rule win_rate = 0.0`
    - `contact_frac_mean = 0.0314`
    - `nearest_enemy_distance_mean = 110.09`
    - `engagement_progress_reward_mean = 0.0`
    - source:
      - `log/sf_maca_4060_fullperf_20260414_intercept_max.eval20.fix_rule.final_20260415.json`
      - `log/sf_maca_4060_fullperf_20260414_intercept_max.phase2_pulse_noatt_c3.log`
  - another prior line restored from `141,041,377` env steps and its 20-episode `fix_rule` audit still showed:
    - `win_rate = 0.0`
    - `contact_frac_mean = 0.0127`
    - `nearest_enemy_distance_mean = 105.15`
    - `engagement_progress_reward_mean = 0.0`
    - source:
      - `log/sf_maca_decoupled_8h_20260414_0040.phase2_fixrule_c3.log`
      - `log/sf_maca_decoupled_8h_20260414_0040.eval20.fix_rule.audit_20260414_1303.json`

#### Practical conclusion

- long training is necessary for this task, but our evidence so far says:
  - `more steps can refine a viable direction`
  - `more steps do not rescue a direction that still fails to produce stable closure`
- so the correct stance is:
  - `for the current run: keep training`
  - `for model selection: do not excuse failure only by saying training time was short`

### Note on explained variance — 2026-04-15

- `explained variance` usually refers to the critic/value-head fit quality, not combat performance.
- common definition:
  - `EV = 1 - Var(target - value_pred) / Var(target)`
- interpretation:
  - near `1`: critic predicts return targets well
  - near `0`: critic is no better than predicting the mean
  - below `0`: critic is worse than the mean baseline
- this project currently does not log `explained variance` in its console or eval JSON outputs.
- local code search on `explained_variance` returned no hits, so it is not part of the present audit surface.
- even if added later, it should be treated only as a training-diagnostics metric; it cannot replace:
  - `fix_rule win_rate`
  - `contact`
  - `attack_opportunity / executed_fire`
  - `nearest_enemy_distance`
  - `engagement_progress_reward`

### Explained-variance patch and resume — 2026-04-15 14:02

- the stopped run was not resumed through the full curriculum wrapper.
- reason:
  - relaunching `scripts/run_maca_4060_overnight.sh` with `FRESH_START=0` restarted `phase1_noatt_warmup`, which is not the correct continuation target for this experiment.
  - I stopped that relaunch and resumed directly into `fix_rule`.

#### Code change

- file:
  - `scripts/train_sf_maca.py`
- patch:
  - monkey-patched `LearnerWorker._record_summaries`
  - added:
    - `explained_variance`
    - `value_target_mean`
    - `value_target_std`
  - learner now logs:
    - `Train summaries: explained_variance=... value_loss=... policy_loss=...`

#### Resume command choice

- resumed directly with:
  - `maca_opponent=fix_rule`
  - `train_for_seconds=6900`
  - `exploration_loss_coeff=0.045`
- log:
  - `log/sf_maca_4060_commit_fire_20260415.phase2_fixrule_c1.resume_20260415_1401.log`

#### Resume evidence

- resumed from checkpoint:
  - `train_dir/sample_factory/sf_maca_4060_commit_fire_20260415/checkpoint_p0/checkpoint_000005838_9382400.pth`
- new checkpoint after resume:
  - `train_dir/sample_factory/sf_maca_4060_commit_fire_20260415/checkpoint_p0/checkpoint_000005839_9388800.pth`
- first logged explained-variance summaries:
  - `explained_variance=0.5391`, `value_loss=0.2080`, `policy_loss=0.0000`
  - `explained_variance=0.7569`, `value_loss=0.3987`, `policy_loss=-0.0043`

#### Audit note

- explained variance is now available as a critic-health indicator.
- it should be interpreted only as:
  - `critic fit quality`
- it still does not answer the main question:
  - `whether fix_rule win rate and active closure are improving`

### Current status check — 2026-04-15 14:55

- live training stage:
  - `fix_rule` resume line is still running
  - latest live frames seen: about `16.99M`
  - source:
    - `log/sf_maca_4060_commit_fire_20260415.phase2_fixrule_c1.resume_20260415_1401.log`

#### Live train indicators

- recent learner summaries:
  - `explained_variance=0.6638`, `value_loss=0.0526`
  - `explained_variance=0.4671`, `value_loss=0.3110`
  - `explained_variance=0.4574`, `value_loss=0.1420`
- recent training log state:
  - fps roughly `2.2k~2.5k`
  - average training reward still negative and unstable, typically around `-800` to `-2100`

#### `fix_rule` 5-episode quick eval on latest checkpoint

- checkpoint:
  - `train_dir/sample_factory/sf_maca_4060_commit_fire_20260415/checkpoint_p0/checkpoint_000010307_15963191.pth`
- source:
  - `log/sf_maca_4060_commit_fire_20260415.eval5.fix_rule.current_20260415_1404.json`
- summary:
  - `win_rate = 0.0`
  - `opponent_round_reward_mean = 1200.0`
  - `contact_frac_mean = 0.01732`
  - `executed_fire_action_frac_mean = 0.00124`
  - `attack_opportunity_frac_mean = 0.00126`
  - `nearest_enemy_distance_mean = 110.52`
  - `course_change_frac_mean = 0.1851`
  - `engagement_progress_reward_mean = 0.0`

#### `fix_rule_no_att` 5-episode quick eval on latest checkpoint

- source:
  - `log/sf_maca_4060_commit_fire_20260415.eval5.fix_rule_no_att.current_20260415_1404.json`
- summary:
  - `win_rate = 1.0`
  - `contact_frac_mean = 0.04458`
  - `nearest_enemy_distance_mean = 29.84`

#### Audit conclusion

- there is no main-target breakthrough yet.
- relative to the earlier 13:33 quick eval, the latest checkpoint is actually mixed-to-worse on `fix_rule`:
  - `contact` dropped from about `0.0320` to `0.0173`
  - `nearest_enemy_distance_mean` improved slightly from `112.95` to `110.52`
  - `engagement_progress_reward_mean` is still `0.0`
- current interpretation:
  - `critic training is numerically healthy enough to continue`
  - `policy has not yet turned that into stable active closure or wins against fix_rule`

### Low-power quiet resume — 2026-04-15 15:54

- objective:
  - continue the same experiment `sf_maca_4060_commit_fire_20260415`
  - keep the same `fix_rule` phase semantics
  - reduce runtime load only

#### Constraint check

- `hidden_size` must remain `256` when resuming from the existing checkpoint; changing it would break checkpoint compatibility.
- therefore the low-power resume changes only runtime/load knobs, not model shape.

#### Low-power overrides actually used

- `num_workers=2`
- `rollout=32`
- `recurrence=32`
- `batch_size=512`
- `ppo_epochs=2`
- `learner_main_loop_num_cores=1`
- `traj_buffers_excess_ratio=2.0`
- `max_policy_lag=16`
- `train_for_seconds=10500`
- `train_for_env_steps=30000000`
- opponent remains:
  - `fix_rule`

#### Resume evidence

- log:
  - `log/sf_maca_4060_commit_fire_20260415.phase2_fixrule_c1.library_resume_20260415_1554.log`
- resumed from checkpoint:
  - `train_dir/sample_factory/sf_maca_4060_commit_fire_20260415/checkpoint_p0/checkpoint_000013294_20367710.pth`
- first new checkpoint:
  - `train_dir/sample_factory/sf_maca_4060_commit_fire_20260415/checkpoint_p0/checkpoint_000013295_20368222.pth`
- low-power runtime evidence:
  - `Using a total of 92 trajectory buffers`
  - early fps around `694.6`
- first resumed learner summary:
  - `explained_variance=0.2542`, `value_loss=0.0027`, `policy_loss=-0.0000`

### Added kill/loss end-state metrics — 2026-04-15

- rationale:
  - yes, this is more useful for audit than only looking at `round_reward`
  - it distinguishes:
    - `clean loss with zero damage`
    - `trade-heavy loss`
    - `narrow tactical advantage that still fails to convert into a win`

#### Metrics added

- end-of-episode fighter counts:
  - `red_fighter_alive_end`
  - `red_fighter_destroyed_end`
  - `blue_fighter_alive_end`
  - `blue_fighter_destroyed_end`
  - `fighter_destroy_balance_end = blue_destroyed - red_destroyed`

#### Code surface

- raw env info:
  - `marl_env/maca_parallel_env.py`
- episode stats:
  - `marl_env/sample_factory_env.py`
- eval JSON + progress print:
  - `scripts/eval_sf_maca.py`

#### Important limitation

- these fields are available for new evals and any training process restarted after the patch.
- the currently running low-power process was started before this patch was loaded, so its already-running Python process will not emit the new fields until the next restart.

### Kill/loss metric restart and bug fix — 2026-04-15 18:12

- action taken:
  - stopped the pre-patch low-power process
  - restarted low-power `fix_rule` continuation so the new metrics are part of runtime episode stats

#### Bug found during first restart

- first restart failed during reset because blue-side alive counts were read from:
  - `blue_obs["fighter"]`
- that key is not present for the current opponent-side observation layout.
- fix:
  - switched the new count logic to raw obs:
    - `red_raw_obs["fighter_obs_list"]`
    - `blue_raw_obs["fighter_obs_list"]`
- file:
  - `marl_env/maca_parallel_env.py`

#### Successful restart evidence

- resumed from:
  - `train_dir/sample_factory/sf_maca_4060_commit_fire_20260415/checkpoint_p0/checkpoint_000035126_25705258.pth`
- new checkpoint:
  - `train_dir/sample_factory/sf_maca_4060_commit_fire_20260415/checkpoint_p0/checkpoint_000035127_25705770.pth`
- log:
  - `log/sf_maca_4060_commit_fire_20260415.phase2_fixrule_c1.library_resume_20260415_1554.log`
- learner resumed normally with low-power settings and no reset crash after the fix.

### Assessment of `fix_rule` as of 2026-04-15

- code-audit limitation:
  - `agent/fix_rule/agent_core.py` is PyArmor-obfuscated, so internal rule quality cannot be audited line by line.
  - assessment therefore has to be black-box and evidence-based.

#### Black-box strength

- `fix_rule` is clearly much stronger than `fix_rule_no_att` for the current project target.
- evidence:
  - our current lane repeatedly gets `win_rate = 1.0` on `fix_rule_no_att` while remaining at `win_rate = 0.0` on `fix_rule`.
  - `fix_rule_no_att` evals typically show `nearest_enemy_distance_mean` around `30~35`.
  - `fix_rule` evals still sit around `110`.

#### Practical judgment

- as a benchmark opponent for this project, `fix_rule` is good:
  - it is hard enough that weak “looks okay” policies do not pass by accident
  - it exposes whether a learner can really survive first merge, sustain closure, and trade damage
- as an engineering artifact, `fix_rule` is weakly auditable:
  - the wrapper is readable
  - the actual logic is hidden by obfuscation
  - this makes targeted counter-analysis and controlled ablation much harder

#### Bottom line

- yes, the current evidence supports the user intuition that `fix_rule` is genuinely hard to beat.
- but the strongest defensible statement is:
  - `it is a strong project-specific baseline`
  - `not necessarily a universally strong air-combat AI, because we cannot inspect or benchmark its internal policy quality directly`

### Post-stop audit at 30M env-step cap — 2026-04-15 19:29

- stop reason:
  - normal stop at the low-power env-step cap
  - final log shows:
    - `Collected {0: 30000314}`
    - `Done!`
  - final checkpoint:
    - `train_dir/sample_factory/sf_maca_4060_commit_fire_20260415/checkpoint_p0/checkpoint_000052731_30000823.pth`

#### `fix_rule` 5-episode post-stop eval

- source:
  - `log/sf_maca_4060_commit_fire_20260415.eval5.fix_rule.poststop_20260415_1927.json`
- summary:
  - `win_rate = 0.0`
  - `opponent_round_reward_mean = 5280.0`
  - `contact_frac_mean = 0.0330`
  - `nearest_enemy_distance_mean = 116.36`
  - `red_fighter_destroyed_end_mean = 9.0`
  - `blue_fighter_destroyed_end_mean = 4.0`
  - `fighter_destroy_balance_end_mean = -5.0`
  - `engagement_progress_reward_mean = 0.0`

#### `fix_rule_no_att` 5-episode post-stop eval

- source:
  - `log/sf_maca_4060_commit_fire_20260415.eval5.fix_rule_no_att.poststop_20260415_1927.json`
- summary:
  - `win_rate = 1.0`
  - `contact_frac_mean = 0.10776`
  - `nearest_enemy_distance_mean = 54.21`
  - `red_fighter_destroyed_end_mean = 0.0`
  - `blue_fighter_destroyed_end_mean = 6.2`
  - `fighter_destroy_balance_end_mean = 6.2`

#### Audit conclusion

- yes, the run stopped because it hit the configured low-power cap, not because the main target was solved.
- yes, this is exactly the point where evaluation and iteration should happen.
- no, the latest 30M checkpoint does not justify further blind continuation:
  - it still loses every `fix_rule` eval
  - end-state damage exchange is negative (`blue_down 4.0` vs `red_down 9.0`)
  - `engagement_progress_reward_mean` remains `0.0`

### First-principles cleanup and refactor — 2026-04-15 21:15

#### Workload

- re-audited:
  - env adapter logic
  - runtime reward overrides
  - curriculum scripts
  - latest `fix_rule` / `fix_rule_no_att` eval outputs
- removed one full layer of over-detailed tactical bias from the training stack
- revalidated compile, shell syntax, and env smoke before the next full-power run

#### Audit conclusion

- the main failure is still `fix_rule`, not `fix_rule_no_att`.
- `fix_rule_no_att` being good does **not** mean the agent is close to solving the real target:
  - `fix_rule_no_att` removes the adversarial pressure that punishes naive closure.
  - this lets a “close fast and fire when allowed” policy look good.
- `fix_rule` staying at `0` wins is most consistent with:
  - `perception is not the dominant bottleneck`
  - `fire execution is not the dominant bottleneck`
  - `the dominant bottleneck is the policy being trained toward the wrong combat problem`
- evidence chain:
  - `fix_rule` post-stop at `30.0M`:
    - `win_rate=0.0`
    - `contact_frac_mean=0.0330`
    - `nearest_enemy_distance_mean=116.36`
    - `red_fighter_destroyed_end_mean=9.0`
    - `blue_fighter_destroyed_end_mean=4.0`
  - slim-obs stop-check at `4.8M`:
    - `win_rate=0.0`
    - `contact_frac_mean=0.0267`
    - `nearest_enemy_distance_mean=113.86`
    - `red_fighter_destroyed_end_mean=9.0`
    - `blue_fighter_destroyed_end_mean=4.6`
  - both runs:
    - `executed_fire_action_frac_mean ≈ attack_opportunity_frac_mean`
    - `engagement_progress_reward_mean = 0.0`
- interpretation:
  - when a legal fire window exists, the policy usually takes it.
  - the problem is that against armed opposition it does **not** create enough good windows before losing too many fighters.

#### Relative priority of failure modes

- `1. threat handling / survival under pressure`
  - strongest evidence:
    - `fix_rule_no_att` wins easily
    - `fix_rule` loses heavily on attrition
  - this points to failure under continuous armed pressure, not generic blindness.
- `2. pursuit / closure control`
  - `contact` is low and `nearest_enemy_distance` stays large in `fix_rule`.
  - this says the policy is not maintaining productive geometry after first detection.
- `3. perception`
  - not solved, but lower priority.
  - if perception were the main blocker, `fix_rule_no_att` would not be this strong and `executed_fire` would not track `attack_opportunity` this closely.

#### Why some reward improvements were false positives

- reward components that can improve without solving `fix_rule`:
  - `reward_radar_fighter_detector`
  - `reward_radar_fighter_fighter`
  - `reward_strike_act_valid`
  - `contact_reward`
  - `progress_reward_scale`
  - `attack_window_reward`
  - `fire_prob_floor` / `attack_prior_strength` induced firing statistics
- why they are risky:
  - they reward “seeing something”, “closing somewhat”, or “taking a valid shot”
  - they do **not** require the policy to survive and win the armed exchange
- this explains how training reward can look better while:
  - `win_rate` stays `0`
  - `red_fighter_destroyed_end` stays near `9`
  - `fighter_destroy_balance_end` stays negative

#### Failure reason of the previous iteration

- the previous stack combined three misaligned biases:
  - long `fix_rule_no_att` warmup, which over-trained threat-free closure
  - extra proxy rewards, which over-valued contact, detection, and valid fire actions
  - detailed course guidance, which crossed the line from “useful inductive bias” into “hand-written tactic”
- that combination made the learner good at a proxy task:
  - get to contact
  - take shots when legal
- but not good at the real task:
  - win armed engagements against `fix_rule`

#### Refactor decision

- keep:
  - state features that expose the real decision context:
    - radar track
    - relative motion
    - team attrition
    - threat pressure
  - action smoothing that reduces control noise:
    - delta course action
    - course hold
    - max course change bins
- remove or default-off:
  - detailed tactical guidance:
    - `offset_pursuit`
    - `recommit cooldown`
    - `defensive beam/disengage` scripting
  - logit shaping that directly nudges behavior:
    - `course_prior`
    - `attack_prior`
    - `fire_prob_floor`
  - proxy rewards:
    - radar detection reward overrides
    - valid-fire bonus override
    - extra contact/progress/window shaping

#### Code changes

- `marl_env/sample_factory_env.py`
  - removed detailed `offset_pursuit` / `recommit` / scripted defensive guidance paths
  - simplified threat-state observation from `3` dims to `2` dims:
    - receive pressure
    - attrition disadvantage
  - simplified engagement shaping so only the configured result/proxy knobs remain, with no threat-conditioned reward gating
- `marl_env/sample_factory_registration.py`
  - removed CLI args for the deleted detailed tactical controls
  - updated threat-state help text to match the simplified semantics
- `scripts/run_sf_maca_decoupled_8h_curriculum.sh`
  - reset default reward overrides toward result-level training:
    - radar rewards `-> 0`
    - valid strike reward `-> 0`
    - keep-alive step `-> -1`
    - draw `-> -1500`
    - missed-attack penalty `-> 0`
    - contact/progress/window shaping `-> 0`
    - extra attrition shaping `-> 0`
  - removed deleted tactical CLI flags from the launch command
- `scripts/run_sf_maca_radar_support_8h_curriculum.sh`
  - defaulted `course_prior`, `intercept_course_assist`, and `attack_prior` to `off`
  - kept semantic screen, radar tracking, relative motion, lock/team/threat observation `on`
  - kept action smoothing `on`
- `scripts/run_maca_4060_overnight.sh`
  - changed the full-power default curriculum to:
    - short `fix_rule_no_att` warmup
    - no no-att pulse cycles
    - long direct `fix_rule` training
- `scripts/run_maca_4060_library_long.sh`
  - aligned the low-power long script to the same curriculum principle

#### Validation

- `python -m py_compile marl_env/sample_factory_env.py marl_env/sample_factory_registration.py marl_env/runtime_tweaks.py scripts/train_sf_maca.py scripts/eval_sf_maca.py`
  - passed
- `bash -n scripts/run_sf_maca_decoupled_8h_curriculum.sh scripts/run_sf_maca_radar_support_8h_curriculum.sh scripts/run_maca_4060_library_long.sh scripts/run_maca_4060_overnight.sh`
  - passed
- smoke test in `conda` env:
  - `SMOKE_OK 10 24 1`
  - meaning:
    - 10 red fighters present
    - fighter measurement dim is now `24`
    - env reset/step path is healthy

#### Reference sources

- eval JSON:
  - `log/sf_maca_4060_commit_fire_20260415.eval5.fix_rule.poststop_20260415_1927.json`
  - `log/sf_maca_4060_commit_fire_20260415.eval5.fix_rule_no_att.poststop_20260415_1927.json`
  - `log/sf_maca_4060_fixrule_slimobs_20260415.eval5.fix_rule.stopcheck_20260415_2032.json`
  - `log/sf_maca_4060_fixrule_slimobs_20260415.eval5.fix_rule_no_att.stopcheck_20260415_2032.json`
- training logs:
  - `log/sf_maca_4060_commit_fire_20260415.phase2_fixrule_c1.library_resume_20260415_1554.log`
  - `log/sf_maca_4060_fixrule_slimobs_20260415.phase1_noatt_warmup.log`

## Scale Refactor — 2026-04-16

### Why the previous training lane was paused

The previous full-power Sample Factory run was paused because the failure mode had stopped looking like a tunable reward or observation issue and started looking like a structural training-lane ceiling.

The strongest audit conclusions were:

1. `fix_rule_no_att` can be solved while `fix_rule` stays at `0` win because the old lane mainly learns "close and fire when legal", not "survive pressure and still produce useful attack windows".
2. `executed_fire ≈ attack_opportunity` shows the policy is not mainly failing by refusing to fire.
3. The old lane still had two near-hard blockers:
   - attack legality collapsed to the nearest visible target only
   - value learning used local observations only while reward was team-level and strongly coupled
4. More tactical reward / course guidance would further blur the line between RL and hand-scripted doctrine, which is explicitly not the direction we want.

### First-principles diagnosis

#### Why `fix_rule_no_att` is good but `fix_rule` is still `0` win

- `fix_rule_no_att` removes the main adversarial pressure. That lets a weak "approach, detect, shoot when legal" policy appear competent.
- `fix_rule` punishes exactly that policy class, because the agent must survive continuous hostile detection / attack pressure before it can create enough windows to trade positively.
- Therefore the transfer gap is not surprising; it is evidence that the old curriculum learned the wrong control problem.

#### Why `executed_fire ≈ attack_opportunity` still loses

- This metric only covers behavior after a legal firing window already exists.
- It says almost nothing about:
  - whether windows appear early enough
  - whether enough fighters survive to exploit them
  - whether the policy can choose the right target when several are visible
- In the audited runs, contact stayed low, distance stayed high, and red losses stayed very high. So the policy fires when it can, but it rarely reaches a sustainable geometry where firing matters.

#### Relative priority of the three problem classes

1. `threat handling / survival`
2. `pursuit / closure control`
3. `perception`

Perception is not perfect, but it is not the dominant blocker anymore. The more severe issue is that the old lane never represented team-level combat pressure and target freedom well enough for PPO/APPO to learn robust behavior.

#### Reward improvements that can be fake

The following can improve while the main objective still fails:

- detector / radar rewards
- valid-fire bonuses
- contact / progress / attack-window shaping
- fire-probability floors or attack priors

Those are proxy metrics. They can move upward while:

- `win_rate` remains `0`
- `red_fighter_destroyed_end` remains near `8~10`
- `fighter_destroy_balance_end` remains strongly negative

### Refactor plan

The project now switches from "keep patching Sample Factory APPO" to a narrower and more controllable training core:

1. Stop the old run and freeze the old lane as audit-only.
   - Purpose: avoid mixing new conclusions with old proxy-driven training.
   - Risk: none.
   - Validation: no active `sf_maca_4060_fixrule_fpclean_20260415` process remains.

2. Replace nearest-target-only attack legality with raw-target-based legality.
   - Purpose: restore real target-selection freedom inside the policy class.
   - Change points:
     - `fighter_action_utils.py`
   - Risk: a larger attack action space can slow early learning.
   - Validation: raw-target attack masks now open every visible in-range target, not just the nearest one.

3. Add a structured local-observation wrapper independent from Sample Factory.
   - Purpose: stop binding the policy to the old `simple` 6-field info bottleneck.
   - Change points:
     - `marl_env/mappo_env.py`
     - `marl_env/maca_parallel_env.py`
   - Risk: this is a new training path, so it needs fresh checkpoints.
   - Validation: env reset/step returns local obs, global state, alive masks, and per-target attack masks.

4. Add a lightweight team-critic PPO trainer.
   - Purpose: fix the local-critic / team-reward mismatch without introducing a large framework dependency change.
   - Change points:
     - `marl_env/mappo_model.py`
     - `scripts/train_mappo_maca.py`
   - Risk: this first version is intentionally simple and not yet recurrent; very long-horizon memory limits may remain.
   - Validation: tiny smoke run can save checkpoints and print episode summaries.

5. Add a matching evaluation + launcher surface.
   - Purpose: make the new lane operational on 4060 / 4080 without another round of ad hoc wrappers.
   - Change points:
     - `scripts/eval_mappo_maca.py`
     - `scripts/run_mappo_maca_train.sh`
     - `scripts/run_mappo_maca_4060_library_long.sh`
     - `scripts/run_mappo_maca_4060_overnight.sh`
     - `scripts/run_mappo_maca_4080_server_scale.sh`
   - Risk: none beyond normal script hygiene.
   - Validation: `bash -n` and one tiny train/eval smoke.

### Implemented code changes

#### 1. Target-selection hard blocker removed

- `fighter_action_utils.py`
  - added `get_valid_attack_indices_from_raw()`
  - added `build_attack_mask_from_raw()`
  - added `build_decoupled_action_mask_from_raw()`
- The new legality path is built from the raw `r_visible_list`, so all visible in-range fighter targets can be selected.

#### 2. New structured environment wrapper

- `marl_env/maca_parallel_env.py`
  - added `get_raw_snapshot()`
  - added `get_map_size()`
- `marl_env/mappo_env.py`
  - new independent wrapper around `MaCAParallelEnv`
  - structured per-fighter local obs:
    - self state
    - missile state
    - receive summary
    - team attrition summary
    - top-K visible enemies
  - centralized `global_state` passthrough for the critic
  - raw-target attack masks
  - episode audit stats aligned with the previous evaluation vocabulary

#### 3. New team-critic PPO core

- `marl_env/mappo_model.py`
  - shared actor
  - small slot / role embedding
  - centralized team critic on `global_state`
- `scripts/train_mappo_maca.py`
  - custom parameter-sharing PPO trainer
  - team reward = mean per-agent reward from the base env
  - actor loss masked by alive agents only
  - checkpoint save / resume support
  - train config persisted to `cfg.json`

This is intentionally a `MAPPO-lite` lane rather than a large, feature-heavy framework replacement.

#### 4. New operational scripts

- `scripts/eval_mappo_maca.py`
  - deterministic or stochastic checkpoint evaluation
  - same main audit indicators as the old eval lane
- `scripts/run_mappo_maca_train.sh`
  - shared launcher
- `scripts/run_mappo_maca_4060_library_long.sh`
  - low-power 4060 profile
- `scripts/run_mappo_maca_4060_overnight.sh`
  - full-power 4060 profile
- `scripts/run_mappo_maca_4080_server_scale.sh`
  - server profile

### Workload summary

- paused one legacy training process
- added 5 new code files
- modified 5 existing code / doc files to wire the new training lane in
- kept MaCA env, dependencies, map assets, and opponent interface unchanged

### Failure reasons of the old lane, stated plainly

- too much effort went into shaping / guidance around a policy class that still lacked key freedoms
- the old actor could not really choose among multiple visible targets
- the old critic did not see the team-level state that dominated reward outcomes
- more detailed tactical shaping would likely make the codebase more scripted, not more learnable

### Validation checklist

- `python -m py_compile fighter_action_utils.py marl_env/maca_parallel_env.py marl_env/mappo_env.py marl_env/mappo_model.py scripts/train_mappo_maca.py scripts/eval_mappo_maca.py`
- `bash -n scripts/run_mappo_maca_train.sh scripts/run_mappo_maca_4060_library_long.sh scripts/run_mappo_maca_4060_overnight.sh scripts/run_mappo_maca_4080_server_scale.sh`
- `conda run --no-capture-output -n maca-py37-min python scripts/train_mappo_maca.py --experiment=mappo_smoke_20260416 --train_dir=/tmp/mappo_smoke --device=cpu --num_envs=1 --rollout=8 --train_for_env_steps=16 --save_every_sec=1 --log_every_sec=1`
- `conda run --no-capture-output -n maca-py37-min python scripts/eval_mappo_maca.py --experiment=mappo_smoke_20260416 --train_dir=/tmp/mappo_smoke --episodes=1 --device=cpu --maca_opponent=fix_rule_no_att --progress=False`

Observed results:

- env smoke:
  - `ENV_OK 10 44 161 (10, 21)`
  - `STEP_OK -1.0 False ['agent_ids', 'alive_mask', 'attack_masks', 'global_state', 'local_obs']`
- train smoke:
  - checkpoint saved at `/tmp/mappo_smoke/mappo_smoke_20260416/checkpoint/checkpoint_000000002_16.pt`
- eval smoke:
  - `fix_rule_no_att`, 1 episode, deterministic
  - `win_rate=1.0`
  - `red_fighter_destroyed_end_mean=0.0`
  - `blue_fighter_destroyed_end_mean=2.0`
  - `fighter_destroy_balance_end_mean=2.0`

### Default recommendation after this refactor

Use the new MAPPO lane as the active training path.

Keep the Sample Factory lane only for:

- historical comparison
- regression audit
- evidence that the refactor is solving the right structural problems

### Active run after refactor

- active experiment:
  - `mappo_maca_4060_overnight_20260416_013337`
- launcher:
  - `scripts/run_mappo_maca_4060_overnight.sh`
- early live log:
  - `env_steps=5632 update=11 reward_mean=-2000.00 win_rate=0.000`
  - `env_steps=10752 update=21 reward_mean=-1769.23 win_rate=0.000`
- log file:
  - `log/mappo_maca_4060_overnight_20260416_013337.train.log`

## Throughput Refactor — 2026-04-16

### Why another training-core change was necessary

The first MAPPO-lite refactor fixed representation and credit-assignment issues, but it still left the machine badly underused:

- one learner process was consuming about one CPU core
- GPU utilization stayed around `1%`
- env collection was still sequential inside one Python process

That meant the training direction was better, but the implementation was still prototype-grade and too slow for real overnight use on a 4060.

### Concrete change

`scripts/train_mappo_maca.py` now uses:

- `multiprocess collectors`
- `centralized learner / centralized policy inference`
- `synchronous rollout exchange over pipes`

This keeps the design simple:

- workers only own env stepping and reset
- the main process still owns model inference, PPO update, checkpoints, and logs

So the project gets multi-core sampling without jumping to a much heavier distributed RL framework.

### Code changes

- `scripts/train_mappo_maca.py`
  - added `--num_workers`
  - added spawned collector worker processes
  - added `CollectorPool`
  - changed rollout collection from in-process env list to worker-backed env batches
  - added `fps` logging
- `scripts/run_mappo_maca_train.sh`
  - forwards `NUM_WORKERS`
- launcher defaults updated:
  - `scripts/run_mappo_maca_4060_library_long.sh`
  - `scripts/run_mappo_maca_4060_overnight.sh`
  - `scripts/run_mappo_maca_4080_server_scale.sh`

### Validation

- `python -m py_compile scripts/train_mappo_maca.py scripts/eval_mappo_maca.py`
  - passed
- `bash -n scripts/run_mappo_maca_train.sh scripts/run_mappo_maca_4060_library_long.sh scripts/run_mappo_maca_4060_overnight.sh scripts/run_mappo_maca_4080_server_scale.sh`
  - passed
- multiprocess smoke train:
  - `--num_envs=2 --num_workers=2`
  - collector log:
    - `[collector] num_workers=2 num_envs=2 worker_env_counts=[1, 1]`
  - checkpoint saved:
    - `/tmp/mappo_mp_smoke/mappo_mp_smoke_20260416/checkpoint/checkpoint_000000001_16.pt`
- multiprocess smoke eval:
  - `scripts/eval_mappo_maca.py`
  - completed successfully against `fix_rule_no_att`

### Active multiprocess run

- experiment:
  - `mappo_maca_4060_overnight_20260416_021101`
- key runtime evidence:
  - collector processes:
    - `worker_env_counts=[2, 2, 2, 2]`
  - worker CPU usage observed:
    - roughly `72% ~ 76%` on four collector processes
  - learner CPU usage observed:
    - about `18%`
  - GPU utilization snapshot:
    - still low (`1%`), which is expected for the current small MLP actor/critic
  - throughput:
    - around `fps=468`

### Audit conclusion after throughput refactor

- the main throughput blocker was fixed:
  - env stepping is no longer single-core
- the 4060 is now using multiple CPU cores as intended
- GPU is still not saturated because the policy network is intentionally small; that is not the current bottleneck
- this is enough to continue the main experiment without reverting to the old training lane

## Throughput Refactor v2 — 2026-04-16

### Why v1 was still not enough

The first multiprocess version still used step-level IPC:

- main process sampled actions each step
- workers stepped envs each step
- `obs / masks / rewards / dones` crossed process boundaries every single step

That fixed the single-core issue, but it still left too much Python IPC overhead. Measured throughput was only around `fps=468`.

### v2 change

`train_mappo_maca.py` now uses rollout-level collection:

- each worker keeps a local actor copy
- each worker collects a full rollout locally
- only the completed rollout tensors are sent back to the learner
- the learner still owns:
  - PPO updates
  - centralized critic
  - checkpoints
  - logging

This keeps the architecture simple while removing the heaviest per-step pipe traffic.

### Validation

- `python -m py_compile scripts/train_mappo_maca.py`
  - passed
- rollout-level smoke train:
  - `--num_envs=2 --num_workers=2 --rollout=8`
  - checkpoint saved:
    - `/tmp/mappo_rollout_smoke/mappo_rollout_smoke_20260416/checkpoint/checkpoint_000000001_16.pt`
- rollout-level smoke eval:
  - `scripts/eval_mappo_maca.py`
  - completed successfully against `fix_rule_no_att`

### Active rollout-level run

- experiment:
  - `mappo_maca_4060_overnight_20260416_022315`
- first observed throughput:
  - `fps=569.7`
  - `fps=593.3`
- compared with the previous step-level collector version:
  - previous:
    - about `fps=442 ~ 488`
  - current:
    - about `fps=570 ~ 593`
- observed CPU usage:
  - four collector workers each around `92% ~ 93%`
- observed learner CPU usage:
  - around `6%`
- observed GPU snapshot:
  - still near idle
  - because the actor / critic are still small MLPs and the bottleneck remains environment sampling rather than model compute

### Current conclusion

- rollout-level collection is materially better than step-level IPC
- the dominant waste moved out of Python process chatter
- the machine is now meaningfully using multiple CPU cores
- GPU saturation is still low, but that is no longer the first problem to solve

## Recurrent + Attrition Refactor — 2026-04-16

### Why this change was necessary

After the rollout-level collector was in place, the main `fix_rule` evaluation still showed the same structural failure:

- `fix_rule_no_att` remained strong
- `fix_rule` remained `0` win
- contact improved somewhat, but the policy was still dying too early and trading badly

That pointed to a remaining control gap:

- the actor still had no temporal memory
- reward still did not explicitly reflect attrition exchange as a dense result-level signal

### Change scope

Only two changes were added:

1. recurrent actor memory
2. light attrition-delta shaping

No tactical-script reward was added.

### Code changes

- `marl_env/mappo_model.py`
  - actor now includes `GRUCell`
  - added `actor_hidden_dim`
  - added `actor_step()`
- `marl_env/mappo_env.py`
  - added config knobs:
    - `friendly_attrition_penalty`
    - `enemy_attrition_reward`
  - team reward now includes light destroy-count delta shaping
- `scripts/train_mappo_maca.py`
  - rollout buffer now stores actor hidden state
  - workers sample recurrently with local hidden state
  - learner computes PPO log-probs with the stored hidden state
  - launcher args now include:
    - `--maca_friendly_attrition_penalty`
    - `--maca_enemy_attrition_reward`
- `scripts/eval_mappo_maca.py`
  - evaluation now runs the recurrent actor with hidden-state carry
- `scripts/run_mappo_maca_train.sh`
  - forwards the new attrition shaping knobs

### Validation

- `python -m py_compile marl_env/mappo_model.py marl_env/mappo_env.py scripts/train_mappo_maca.py scripts/eval_mappo_maca.py`
  - passed
- `bash -n scripts/run_mappo_maca_train.sh scripts/run_mappo_maca_4060_overnight.sh`
  - passed
- recurrent smoke train:
  - experiment:
    - `mappo_rnn_smoke_20260416`
  - checkpoint saved:
    - `/tmp/mappo_rnn_smoke/mappo_rnn_smoke_20260416/checkpoint/checkpoint_000000001_16.pt`
- recurrent smoke eval:
  - `fix_rule_no_att`
  - `win_rate=1.0`

### Compatibility note

This change is checkpoint-incompatible in practice for the main experiment path:

- actor architecture changed from feed-forward to recurrent
- reward function changed

So the new run intentionally starts from scratch.

### Active run after recurrent refactor

- experiment:
  - `mappo_maca_4060_overnight_20260416_091844`
- launch characteristics:
  - 4060 full power
  - rollout-level multiprocess collectors
  - recurrent actor
  - attrition shaping:
    - `friendly_attrition_penalty=200`
    - `enemy_attrition_reward=100`
- early log:
  - `env_steps=12288 update=12 fps=366.2`
  - `env_steps=80896 update=79 fps=591.1`
  - `env_steps=97280 update=95 fps=545.7`
