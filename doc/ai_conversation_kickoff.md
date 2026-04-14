# MaCA 项目对话开场白（2026-04-11）

你现在在项目：`/home/lehan/MaCA-master`（或你当前本地仓库根目录）。

这是一个 MaCA（多智能体空战）强化学习项目。当前主线已经从旧版 DQN 迁移到 `Sample Factory + APPO/PPO + LSTM`，DQN 相关脚本仅作为历史参考保留。

## 1. 当前目标

- 用当前 `Sample Factory` 训练链路，训练红方 10 架战斗机共享策略。
- 在 `fix_rule` 对手上建立稳定、可复现实验闭环。
- 用统一的评估脚本比较不同 checkpoint，而不是只盯训练日志。

## 2. 当前主线入口

- 多智能体并行环境：`marl_env/maca_parallel_env.py`
- Sample Factory 适配层：`marl_env/sample_factory_env.py`
- 自定义 encoder：`marl_env/sample_factory_model.py`
- 环境与默认参数注册：`marl_env/sample_factory_registration.py`
- 训练入口：`scripts/train_sf_maca.py`
- GPU 冒烟脚本：`scripts/run_sf_maca_gpu_smoke.sh`
- 基线训练脚本：`scripts/run_sf_maca_4060_baseline.sh`
- 低显存训练脚本：`scripts/run_sf_maca_4060_8g_curriculum.sh`
- 恢复课表脚本：`scripts/run_sf_maca_recovery_curriculum.sh`
- 4080 fresh-start 脚本：`scripts/run_sf_maca_4080_freshstart.sh`
- 评估脚本：`scripts/eval_sf_maca.py`

旧 DQN 路线仍在仓库中，但不是当前默认建议：

- `scripts/train_dqn_pipeline.py`
- `scripts/run_curriculum_autotrain.sh`
- `scripts/run_fix_mixed_1to2h.sh`
- `scripts/run_foundation_first_1h.sh`
- `scripts/run_foundation_first_2to3h.sh`

## 3. 当前代码已经落实的关键设计

- 只把红方战斗机暴露为学习 agent，固定 10 个槽位。
- 飞机死亡后不移除 agent，而是保留槽位、清零观测，并只允许 no-op。
- 动作空间固定为 `336`，合法动作由 `fighter_action_utils.py` 动态生成 mask。
- 训练和评估都会把 `action_mask` 真正作用到 policy logits。
- 环境层仍保留 `_sanitize_action()` 兜底，避免非法动作导致崩溃。
- `measurements` 已从旧的线性 6 维改为当前 7 维：`sin(course) + cos(course) + 其余 5 个线性量`。
- `scripts/train_sf_maca.py` 里包含关键运行时补丁链：
  - checkpoint 临时文件保存补丁
  - 单轨迹 buffer squeeze 补丁（可开关）
  - 动作掩码补丁（训练/评估一致）
  - 可选 fire-logit bias（仅训练侧）

## 4. 当前代码中的默认训练参数

以 `marl_env/sample_factory_registration.py`、`scripts/run_sf_maca_4060_baseline.sh`、`scripts/run_sf_maca_recovery_curriculum.sh` 为准。

### 框架注册默认值

| 参数 | 当前值 |
|---|---|
| `hidden_size` | `256` |
| `rollout` | `64` |
| `recurrence` | `64` |
| `num_workers` | `8` |
| `batch_size` | `5120` |
| `learning_rate` | `1e-4` |
| `gamma` | `0.999` |
| `reward_scale` | `0.005` |
| `reward_clip` | `50.0` |
| `ppo_epochs` | `4` |
| `max_policy_lag` | `15` |
| `exploration_loss_coeff` | `0.02` |

### 4060 基线脚本默认值

`scripts/run_sf_maca_4060_baseline.sh` 当前默认也是：

- `OPPONENT=fix_rule`
- `MAX_STEP=1000`
- `TRAIN_SECONDS=7200`
- `TRAIN_ENV_STEPS=50000000`
- `NUM_WORKERS=8`
- `ROLLOUT=64`
- `RECURRENCE=64`
- `BATCH_SIZE=5120`

注意：

- 这代表“当前代码默认启动配置”，不等于“已经证明最优”。
- 旧文档里出现过的 `4 workers / 256 batch_size` 是更早一轮调参假设，不再代表现在脚本默认值。

## 5. 当前奖励与动作约束事实

以 `configuration/reward.py` 和 `fighter_action_utils.py` 为准：

- `reward_strike_fighter_success = 900`
- `reward_strike_act_valid = 2`
- `reward_strike_act_invalid = -4`
- `reward_keep_alive_step = -1`
- `reward_totally_win = 8000`
- `reward_totally_lose = -2000`
- 长导弹距离阈值：`120.0`
- 短导弹距离阈值：`50.0`

如果文档里出现“合法开火奖励已经改成 0”之类描述，以代码为准，当前并没有改成 0。

## 6. 当前建议工作方式

1. 先看 `scripts/run_sf_maca_gpu_smoke.sh` 是否能在本机闭环。
2. 正式训练优先走 `scripts/run_sf_maca_recovery_curriculum.sh`（通用）或 `scripts/run_sf_maca_4060_8g_curriculum.sh`（8GB 显存）。
3. 训练结束后必须跑 `scripts/eval_sf_maca.py` 做独立评估。
4. 调参时优先改 1 到 3 个关键参数，不要一次同时改整组。
5. 如果结论与文档不一致，优先相信当前代码和脚本。

## 7. 常用命令模板

### 冒烟

```bash
RUN_ID="$(date +%Y%m%d_%H%M%S)"
EXP_NAME="sf_maca_gpu_smoke_${RUN_ID}" \
bash scripts/run_sf_maca_gpu_smoke.sh
```

### 正式训练

```bash
RUN_ID="$(date +%Y%m%d_%H%M%S)"
EXP_NAME="sf_maca_4060_fixrule_${RUN_ID}"
nohup bash scripts/run_sf_maca_4060_baseline.sh \
  > "log/${EXP_NAME}.launcher.log" 2>&1 &
echo $! > "log/${EXP_NAME}.pid"
echo "$EXP_NAME"
```

### 看日志

```bash
tail -f "log/${EXP_NAME}.launcher.log"
```

### 停止训练

```bash
kill "$(cat "log/${EXP_NAME}.pid")"
```

### 评估最新 checkpoint

```bash
conda run --no-capture-output -n maca-py37-min \
  python scripts/eval_sf_maca.py \
  --experiment="$EXP_NAME" \
  --train_dir=train_dir/sample_factory \
  --episodes=30
```

## 8. 结论优先级

当你需要给下一步建议时，优先级按这个顺序：

1. 当前代码真实实现
2. 当前脚本默认值
3. 独立评估结果
4. 训练日志中的吞吐、policy lag、checkpoint 行为
5. 历史文档中的经验总结

不要先讲泛泛理论，先对照现代码和最新日志给结论。
