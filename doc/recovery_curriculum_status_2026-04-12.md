# Recovery 课表与当前进展（2026-04-12）

## 1) 当前课表（以代码为准）

主脚本：`scripts/run_sf_maca_recovery_curriculum.sh`

- Phase 1（打破不开火局部最优）
  - 对手：`fix_rule_no_att`
  - 时长：`PHASE1_SECONDS=5400`（1.5h）
  - 探索系数：`PHASE1_EXPLORATION=0.06`
- Phase 2（迁移到真实目标对手）
  - 对手：`fix_rule`
  - 时长：累计秒数控制，默认 `PHASE2_TRAIN_SECONDS_TOTAL=21600`
  - 探索系数：`PHASE2_EXPLORATION=0.03`

辅助守护：`scripts/watch_resume_until_2200.sh`

- 作用：在 `22:00` 前若训练中断，自动清理 `done` 并续训。
- 当前实验：`sf_maca_recovery_20260412_163435`

## 2) 到目前为止，哪个环节缓解了问题

### 已被缓解

- “几乎不发射导弹”在 Phase 1 明显缓解。
- 对 `fix_rule_no_att` 的评估已经达到稳定高表现：
  - `win_rate = 1.0`（20局）
  - `attack_opportunity_frac_mean = 1.0`
  - `fire_action_frac_mean ≈ 0.0017`

### 仍待攻克

- 对 `fix_rule` 仍未破零胜率（当前 `win_rate = 0.0`），但续训后
  `round_reward_mean` 与 `true_reward_mean` 已较前一轮明显改善。
- 说明 Phase 2 的迁移方向是对的，但还需要更长时间窗口和更密集评估。

## 3) 关键结果文件

- `log/sf_maca_recovery_20260412_163435.eval20.no_att.json`
- `log/sf_maca_recovery_20260412_163435.eval20.fix_rule.json`
- `log/sf_maca_recovery_20260412_163435.eval20.fix_rule.after1h.json`
- `log/sf_maca_recovery_20260412_163435.phase2_resume_1h.log`

## 4) 操作建议（当前）

- 继续维持 Phase 2 训练到 `22:00`，再做一次 20 局评估。
- 若 `win_rate` 仍为 0，但 `true_reward_mean` 持续改善，优先再给 2~4 小时训练窗口。
- 若出现回报退化且 fire 频率再次塌缩，再回看奖励项与探索系数。
