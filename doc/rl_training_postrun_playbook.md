# MaCA 训练后标准流程（Post-Run Playbook）

目的：每次你说“本轮训练完毕”后，用同一套流程评估结果、决定下一轮配置，尽量自动化。

## 0. 输入与产物约定

每轮训练至少产出两个文件：

- `log/<metrics>.csv`（逐 epoch 指标）
- `log/<summary>.json`（本轮概览）

示例：

- `log/train_dqn_metrics_stage1_e100.csv`
- `log/train_dqn_summary_stage1_e100.json`

## 0.1 续训规则（重要）

当前训练脚本 `scripts/train_dqn_pipeline.py` 的行为：

- 默认自动续训：若存在 `model/simple/model.pkl`，会自动加载继续训练
- 显式指定续训：`--resume <checkpoint_path>`
- 强制从头训练：`--fresh_start`

建议：
- 阶段切换（如 `fix_rule_no_att -> fix_rule`）默认用续训
- 只有做对照实验时才用 `--fresh_start`

## 1. 固定读取顺序

1. 先读 summary：

```bash
cat log/<summary>.json
```

重点字段：
- `epochs`
- `opponent`
- `latest_checkpoint`
- `elapsed_sec`
- 是否续训成功（看训练启动日志是否包含 `Resume training from:`）

2. 再读 metrics 首尾：

```bash
sed -n '1,15p' log/<metrics>.csv
echo '---'
tail -n 15 log/<metrics>.csv
```

3. 再做统计（建议固定脚本）：

```bash
python - <<'PY'
import csv, statistics as st
p='log/<metrics>.csv'
rows=list(csv.DictReader(open(p)))
tr=[float(r['total_reward']) for r in rows]
win=[int(r['red_win']) for r in rows]
el=[float(r['elapsed_sec']) for r in rows]
print('epochs',len(rows))
print('win_rate_all',sum(win)/len(win))
for k in [10,20,30]:
    if len(rows)>=k:
        print(f'win_rate_last{k}',sum(win[-k:])/k)
        print(f'total_reward_last{k}',sum(tr[-k:])/k)
print('reward_min/max/mean',min(tr),max(tr),sum(tr)/len(tr))
if len(tr)>=20:
    print('reward_std_last20',st.pstdev(tr[-20:]))
print('epoch_time_mean',sum(el)/len(el))
PY
```

## 2. 决策规则（建议）

### A. 是否进入下一阶段（对手升级）

满足以下任意一条可升级：

- `win_rate_last20 >= 0.95`
- 且最近 20 局无明显崩溃（例如连续多局 `red_win=0`）

升级方式：
- 从 `--opponent fix_rule_no_att` 进入 `--opponent fix_rule`
- 先小规模 20~30 epoch 观察，再决定是否继续长训

### B. 是否继续当前阶段

出现以下情况建议继续当前阶段：

- `win_rate_last20 < 0.9`
- 或 `total_reward` 波动仍非常大且存在明显回退

### C. 是否回退参数

- 若胜率下降且波动放大：减小学习率（如 `0.001 -> 0.0005`）
- 若学习太慢且胜率长期不涨：增加训练强度（减小 `learn_interval`）

## 3. 可调项总览（含文件位置）

## 3.1 训练命令参数（首选调参入口）

文件：`scripts/train_dqn_pipeline.py`

- `--epochs`
- `--max_step`
- `--learn_interval`
- `--batch_size`
- `--lr`
- `--gamma`
- `--epsilon`
- `--epsilon_increment`
- `--target_replace_iter`
- `--memory_size`
- `--opponent`
- `--seed`

## 3.2 模型结构（中等风险）

文件：`dqn.py`

- `NetFighter.conv1/conv2`（卷积通道、核大小）
- `info_fc`（信息向量维度）
- `feature_fc`（全连接层宽度、dropout）
- 优化器（RMSprop/Adam）

说明：改模型结构后，历史 checkpoint 常常不能直接复用。

## 3.3 观测构造（高影响）

文件：`obs_construct/simple/construct.py`

可调：
- `img_obs_reduce_ratio`
- fighter `info` 维度与定义
- 图像通道编码方式

注意：改观测后要同步检查 `dqn.py` 输入维度（例如 `in_channels` 与 `info_fc` 输入）。

## 3.4 奖励（高影响）

接口读取位置：训练中通过 `env.get_reward()` 使用（`scripts/train_dqn_pipeline.py`）。

底层奖励逻辑在环境内核模块中（`environment/world/*`）。

注意：奖励设计变动会直接改变学习目标，需单独记录实验编号。

## 4. 每轮结束后的标准动作清单

1. 归档结果文件（metrics + summary + checkpoint 名）
2. 计算固定统计（全局、last10/20/30）
3. 根据第 2 节规则决定：
- 继续当前阶段
- 升级对手
- 微调超参数
4. 确认下一轮启动模式：
- 续训：默认即可，或显式 `--resume model/simple/model.pkl`
- 从头：显式加 `--fresh_start`
5. 只改 1~2 个关键参数再开下一轮（避免变量过多）
6. 记录本轮配置与结论（建议追加到 `chat_summary.md`）

## 5. 自动化建议（最小可落地）

建议新增一个脚本（例如 `scripts/analyze_training_metrics.py`），固定输出：

- win_rate_all / last10 / last20 / last30
- reward min/max/mean/std
- epoch_time mean/min/max
- 推荐动作（继续/升级/回退）

这样每轮只需：

```bash
python scripts/analyze_training_metrics.py --metrics log/<metrics>.csv --summary log/<summary>.json
```

## 6. 当前项目的阶段建议（基于 stage1_e100）

已观察到：
- `win_rate_all = 0.95`
- `win_rate_last20 = 1.0`

建议：
1. 进入阶段 2（攻击型对手 `fix_rule`）
2. 先跑 30 epoch 观察稳定性
3. 阶段 2 使用续训模式（不要 `--fresh_start`）
4. 若胜率掉到 <0.7，再回到 no_att 混训（例如 2:1 轮换）
