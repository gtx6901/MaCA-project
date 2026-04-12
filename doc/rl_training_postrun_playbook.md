# MaCA 训练后标准流程（当前主线：Sample Factory）

目的：每次一轮 `Sample Factory` 训练结束后，都用同一套流程检查 experiment、checkpoint、日志和独立评估结果，然后再决定下一轮怎么改。

## 0. 当前适用范围

本文档适用于当前主线：

- 训练入口：`scripts/train_sf_maca.py`
- 冒烟脚本：`scripts/run_sf_maca_gpu_smoke.sh`
- 基线脚本：`scripts/run_sf_maca_4060_baseline.sh`
- 评估脚本：`scripts/eval_sf_maca.py`

旧 DQN 训练产物格式和这里不同，不再是本文默认对象。

## 1. 一轮训练最少要拿到什么

至少确认下面几类产物存在：

- launcher 日志：`log/<exp_name>.launcher.log`
- experiment 目录：`train_dir/sample_factory/<exp_name>/`
- checkpoint 目录：`train_dir/sample_factory/<exp_name>/checkpoint_p0/`
- 独立评估结果：建议保存成 `log/<exp_name>.eval.json`

## 2. 固定读取顺序

### 2.1 先确认 experiment 是否真的完成过训练

看 launcher 日志里是否出现这些事实：

- learner 成功启动
- policy worker 成功启动
- 至少发生过一次 optimizer update
- 至少保存过一个 checkpoint

建议先读日志尾部：

```bash
tail -n 80 "log/<exp_name>.launcher.log"
```

### 2.2 再看 checkpoint

```bash
ls -lh "train_dir/sample_factory/<exp_name>/checkpoint_p0"
```

重点确认：

- 是否有 `checkpoint_*.pth`
- 最新 checkpoint 时间是否合理
- 是否出现 `done` 文件

### 2.3 再跑独立评估

```bash
conda run --no-capture-output -n maca-py37-min \
  python scripts/eval_sf_maca.py \
  --experiment="<exp_name>" \
  --train_dir=train_dir/sample_factory \
  --episodes=30 \
  --output_json="log/<exp_name>.eval.json"
```

### 2.4 最后读评估摘要

```bash
cat "log/<exp_name>.eval.json"
```

重点字段：

- `summary.win_rate`
- `summary.round_reward_mean`
- `summary.opponent_round_reward_mean`
- `summary.true_reward_mean`
- `summary.invalid_action_frac_mean`
- `summary.episode_len_mean`

## 3. 当前决策规则

### A. 可以继续当前配置

满足以下特征时，优先继续当前配置而不是立刻大改：

- `win_rate` 在最近几轮评估里持续抬升
- `invalid_action_frac_mean` 很低
- 训练日志没有长期 learner backlog
- checkpoint 保存稳定，没有异常退出

### B. 应该先查系统问题

优先处理系统层问题，而不是急着调 reward 或网络：

- launcher 日志反复出现 learner backlog / accumulated too much experience
- checkpoint 保存失败
- 训练一恢复就立刻退出
- 评估脚本无法加载最新 checkpoint

### C. 应该调整超参数

在系统链路正常、但效果不理想时，再考虑调参：

- `win_rate` 长期接近 0
- `episode_len_mean` 长期接近 `max_step`
- `true_reward_mean` 长期很差
- reward 在训练日志中剧烈震荡但评估无改善

## 4. 先查哪里，再改哪里

### 4.1 启动与框架参数

优先入口：

- `scripts/run_sf_maca_4060_baseline.sh`
- `marl_env/sample_factory_registration.py`

常用项：

- `num_workers`
- `rollout`
- `recurrence`
- `batch_size`
- `ppo_epochs`
- `learning_rate`
- `gamma`
- `reward_scale`
- `reward_clip`
- `max_policy_lag`
- `exploration_loss_coeff`

### 4.2 环境与观测

文件：

- `marl_env/maca_parallel_env.py`
- `marl_env/sample_factory_env.py`
- `fighter_action_utils.py`

常看点：

- 动作 mask 是否合理
- `measurements` 是否和模型输入一致
- 死亡飞机槽位处理
- 非法动作兜底比例

### 4.3 奖励

文件：

- `configuration/reward.py`

当前代码中关键事实：

- `reward_strike_act_valid = 5`
- `reward_strike_act_invalid = -8`
- `reward_totally_win = 8000`
- `reward_totally_lose = -2000`

改奖励前先确认：训练信号问题是否真的来自 reward，而不是来自吞吐、动作掩码或评估方式。

## 5. Resume / 继续训练检查清单

恢复训练前固定检查：

1. `EXP_NAME` 是否和旧 experiment 完全一致。
2. experiment 目录下是否有 `done` 文件。
3. 新设定的 `train_for_env_steps` 是否大于旧 checkpoint 已达到的 env steps。
4. 观测维度是否发生变化。
5. 当前代码 patch 是否仍与旧 checkpoint 兼容。

如果 observation space、模型结构或动作定义已经改动，优先 fresh start。

## 6. 每轮结束后的标准动作

1. 记录 experiment 名称、训练脚本、关键超参数。
2. 归档 launcher 日志路径。
3. 确认最新 checkpoint 路径。
4. 跑固定 episode 数的独立评估。
5. 记录 `win_rate`、`true_reward_mean`、`invalid_action_frac_mean`、`episode_len_mean`。
6. 只改 1 到 3 个关键参数再开下一轮。
7. 把结论写回 `doc/` 或 `chat_summary.md`。

## 7. 推荐命令模板

### 查看 experiment checkpoint

```bash
ls -lh "train_dir/sample_factory/<exp_name>/checkpoint_p0"
```

### 查看 done 文件

```bash
find "train_dir/sample_factory/<exp_name>" -maxdepth 2 -name done -o -name "*.pth"
```

### 跑评估并落盘

```bash
conda run --no-capture-output -n maca-py37-min \
  python scripts/eval_sf_maca.py \
  --experiment="<exp_name>" \
  --train_dir=train_dir/sample_factory \
  --episodes=30 \
  --output_json="log/<exp_name>.eval.json"
```

## 8. 当前最重要的实践规则

不要只依据训练 reward 做结论。对于当前主线，独立评估结果的优先级高于训练过程里的局部 reward 波动。
