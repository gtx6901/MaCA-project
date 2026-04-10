# MaCA DQN 项目对话开场白（给下一位 AI）

你现在在项目：`/home/lehan/MaCA-master`。  
这是一个 MaCA（多智能体空战）上的 DQN 强化学习训练项目，我正在用它学习 RL。

## 1. 目标与当前问题

- 目标：训练出在 `fix_rule`（带攻击敌人）下可稳定获胜的模型。
- 当前主要问题：
  - 模型在 `fix_rule` 上长期 `win_rate=0`。
  - 在 `fix_rule_no_att` 场景中，常出现 episode 跑满 `max_step`（策略不够“高效终结”）。
  - 训练瓶颈偏 CPU，且磁盘空间有限（checkpoint 体积大）。

## 2. 我当前环境（已可用）

- Conda 环境：`maca-py37-min`
- GPU 可用（训练日志会打印 `GPU Available!!`）
- 我可以手动输入训练命令；AI 负责：
  - 读日志、分析结果
  - 调参与改脚本
  - 给我可直接执行的命令

参考文档：
- `doc/gpu_env_setup.md`
- `doc/rl_training_postrun_playbook.md`
- `doc/dqn_upgrade_playbook_2026-04-10.md`

## 3. 关键代码入口（先读这些）

- 模型与学习器：`dqn.py`
- 单次训练主脚本：`scripts/train_dqn_pipeline.py`
- 课程式自动训练：`scripts/run_curriculum_autotrain.sh`
- 2-3 小时基础优先训练封装：`scripts/run_foundation_first_2to3h.sh`
- 1-2 小时 fix 主导混合训练封装：`scripts/run_fix_mixed_1to2h.sh`

## 4. 当前已完成的核心改造

- DQN 升级到：Dueling + Double DQN + Huber loss + grad clip + reward clip。
- replay 增加 `done`，目标值使用 `(1-done)` mask。
- 训练流程增加 `--min_replay_size` 预热门槛。
- 自动训练脚本支持 checkpoint 清理与保留上限。
- 新增 `--noatt-epochs-after-first`，可在首轮后跳过 `no_att` block。
- 支持动作改为训练/推理一致的确定性策略，避免 hidden-policy mismatch。
- DQN 增加基于当前 `info` 的有效攻击动作掩码，减少无效探索。
- 对 fighter 单体死亡样本使用 terminal 截断，避免死亡后继续 bootstrap。

## 5. 最近一轮训练结果（重点）

最近需要优先参考两轮结果：

- `run_id=20260410_094147`（旧 fix-heavy 课表）
  - `ALL noatt`: `n=36, win=0.417, avg_steps=500.0, timeout_rate=1.000`
  - `ALL fix`: `n=1200, win=0.000, avg_steps=417.9, reward_last20≈-18994.0`

- `run_id=20260410_122530`（引入 action mask / terminal 修正后的半程验证）
  - `ALL noatt`: `n=2, win=0.500, avg_steps=300.0, timeout_rate=1.000`
  - `ALL fix`: `n=268, win=0.000, avg_steps=373.8, timeout_rate=0.291, reward_last20≈-12098.7`
  - 结论：`fix_rule` 仍未赢，但总奖励显著改善，`-2000` 级惨败占比从 `0.833` 降到 `0.712`，说明结构改动方向正确；下一步应改为 `no_att` 基础优先训练，而不是继续 fix-heavy。

- `run_id=20260410_142256`（超长 foundation-first，GPU headless）
  - `ALL noatt`: `n=1770, win=0.766, avg_steps=321.0, timeout_rate=0.999, reward_last20≈85812.3`
  - `ALL fix`: `n=320, win=0.000, avg_steps=252.7, timeout_rate=0.256, reward_last20≈-16778.8`
  - 结论：长时间纯 `no_att` 预训练会明显强化“生存/拿分”，但没有学到“对真实攻击敌人取胜”；一进入 `fix_rule` 仍会快速崩掉。后续默认方向应改为：
    - 只保留较短的 `no_att` foundation
    - 尽快转入 `fix_rule` 主导的交替训练
    - 不再把 `foundation-first` 作为首选主线

日志位置：
- `log/overnight_2to3h_20260410_094147.log`
- `log/auto_curriculum_20260410_094147_phaseA.log`
- `log/auto_curriculum_20260410_094147_phaseB.log`
- `log/auto_curriculum_20260410_094147_phaseC.log`

## 6. 我希望你的工作方式

1. 先读“最新一轮”日志与 metrics，再给结论。  
2. 结论必须量化：`win_rate`、`avg_steps`、`reward_last10/20`、是否过拟合/退化。  
3. 每轮优先只改 `1-3` 个关键参数，避免一次改太多。  
4. 给我可直接运行的命令（含 `nohup`、`tail -f`、`stop`）。  
5. 如果要大改（网络结构/奖励函数），先说明理由和风险。  

当前默认建议入口：

- 优先用 `scripts/run_fix_mixed_1to2h.sh` 做新策略验证
- `scripts/run_foundation_first_1h.sh` / `scripts/run_foundation_first_2to3h.sh` 保留作历史基线，不再作为首选

## 7. 常用命令模板

后台启动（from scratch，推荐当前默认入口）：

```bash
RUN_ID="$(date +%Y%m%d_%H%M%S)"
nohup bash scripts/run_foundation_first_2to3h.sh \
  --run-id "$RUN_ID" \
  --headless 1 \
  --fresh-start \
  --clean-model \
  > "log/foundation_first_2to3h_${RUN_ID}_launcher.log" 2>&1 &
echo $! > "log/foundation_first_2to3h_${RUN_ID}.pid"
```

实时看日志：

```bash
tail -f "log/foundation_first_2to3h_${RUN_ID}.log"
```

停止训练：

```bash
kill "$(cat log/foundation_first_2to3h_${RUN_ID}.pid)"
```

---

请先从“读取最新日志并给出参数调整建议”开始，不要先讲泛泛理论。
