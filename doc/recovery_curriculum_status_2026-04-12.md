# Recovery 课表与当前进展（2026-04-12）

## 1) 当前课表（以代码为准）

主脚本：`scripts/run_sf_maca_recovery_curriculum.sh`

- Phase 1（打破不开火局部最优）
  - 对手：`fix_rule_no_att`
  - 时长：`PHASE1_SECONDS=5400`（1.5h）
  - 探索系数：`PHASE1_EXPLORATION=0.06`
- Phase 2（迁移到真实目标对手）
  - 对手：`fix_rule`
  - 时长：默认 `PHASE2_SECONDS=16200`（4.5h）
  - 探索系数：`PHASE2_EXPLORATION=0.045`（较旧版本 0.03 更高）
  - 新增脉冲课表：每个 block 先短时 `fix_rule_no_att` 再切回 `fix_rule`
    - `PHASE2_BLOCK_SECONDS=2700`
    - `PHASE2_PULSE_SECONDS=900`
    - `PHASE2_MAIN_SECONDS=1800`
    - `PHASE2_CYCLES=6`
    - 作用：在 phase2 持续维持主动攻击行为，减少“会规避但不开火”的回退

辅助守护（当前代码）：

- `scripts/run_sf_maca_recovery_validation.sh`
- `scripts/auto_eval_during_training.sh`

当前实验：`sf_maca_recovery_20260412_163435`

## 2) 到目前为止，哪个环节缓解了问题

### 已被缓解

- “几乎不发射导弹”在 Phase 1 明显缓解。
- 对 `fix_rule_no_att` 的评估已经达到稳定高表现：
  - `win_rate = 1.0`（20局）
  - `attack_opportunity_frac_mean = 1.0`
  - `fire_action_frac_mean ≈ 0.0017`

### 仍待攻克

- 对 `fix_rule` 仍然整体偏弱，但最新暂停复评已偶发破零：
  - 10 局：`win_rate = 0.10`
  - 20 局复评：`win_rate = 0.05`
- 训练趋势仍是正向：生存能力明显提升（早亡局下降），但攻击机会转化仍低。
- 说明接下来应优先“训练策略微调”，而不是只延长同参训练时长。

## 3) 关键结果文件

- `log/sf_maca_recovery_20260412_163435.eval20.no_att.json`
- `log/sf_maca_recovery_20260412_163435.eval20.fix_rule.json`
- `log/sf_maca_recovery_20260412_163435.eval20.fix_rule.after1h.json`
- `log/sf_maca_recovery_20260412_163435.phase2_resume_1h.log`
- `log/eval_pause_check_latest.json`
- `log/eval_pause_check_rerun20_20260412_213204.json`

## 4) 操作建议（当前）

- 继续训练，但改为“phase2 脉冲课表 + 更高探索系数”的新默认。
- 固定每 1~2 小时做一次 20 局评估，避免被 5~10 局小样本波动误导。
- 重点盯两个指标：
  - `fire_action_frac_mean`
  - `fire_action_frac_mean / attack_opportunity_frac_mean`
- 若 2~3 个评估窗口后攻击转化仍不升，再进一步提高攻击相关奖励项权重。
