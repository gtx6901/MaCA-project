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
- 2-3 小时训练封装：`scripts/run_overnight_2to3h.sh`

## 4. 当前已完成的核心改造

- DQN 升级到：Dueling + Double DQN + Huber loss + grad clip + reward clip。
- replay 增加 `done`，目标值使用 `(1-done)` mask。
- 训练流程增加 `--min_replay_size` 预热门槛。
- 自动训练脚本支持 checkpoint 清理与保留上限。
- 新增 `--noatt-epochs-after-first`，可在首轮后跳过 `no_att` block。

## 5. 最近一轮训练结果（重点）

最近 `run_id=20260410_094147` 的汇总：
- `ALL noatt`: `n=36, win=0.417, avg_steps=500.0, timeout_rate=1.000`
- `ALL fix`: `n=1200, win=0.000, avg_reward=-23936.4, avg_steps=417.9`

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

## 7. 常用命令模板

后台启动（from scratch）：

```bash
RUN_ID="$(date +%Y%m%d_%H%M%S)"
nohup bash scripts/run_overnight_2to3h.sh \
  --run-id "$RUN_ID" \
  --headless 1 \
  --fresh-start \
  --clean-model \
  > "log/overnight_2to3h_${RUN_ID}_launcher.log" 2>&1 &
echo $! > "log/overnight_2to3h_${RUN_ID}.pid"
```

实时看日志：

```bash
tail -f "log/overnight_2to3h_${RUN_ID}.log"
```

停止训练：

```bash
kill "$(cat log/overnight_2to3h_${RUN_ID}.pid)"
```

---

请先从“读取最新日志并给出参数调整建议”开始，不要先讲泛泛理论。
