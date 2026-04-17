# HMARL TensorBoard 指标说明（数学意义 / 训练作用 / 评价标准）

本文面向当前项目 `scripts/train_mappo_maca.py` 的真实 TensorBoard 输出，按以下四类组织：

1. `train/*`：优化与数值稳定指标
2. `train_episode/*`：按回合聚合的战术/结果指标
3. `train_action/*`：动作分布与塌缩检测指标
4. `eval/*` 与 `curriculum/*`：评估与课表阶段指标

## 1. 指标命名与来源

- 训练优化指标：`writer.add_scalar("train/..."...)`
- 回合聚合指标：`log_summary(writer, "train_episode", summary, env_steps)`
- 评估聚合指标：`log_summary(writer, "eval", eval_summary, env_steps)`
- 动作分布指标：`writer.add_scalar("train_action/..."...)`
- 课表指标：`writer.add_scalar("curriculum/stage_id"...)`

说明：
- `train_episode/*` 与 `eval/*` 绝大多数键来自环境 `marl_env/mappo_env.py` 的 `episode_extra_stats`。
- 本文中的“评价标准”给的是工程可操作区间，不是严格理论上界；需结合阶段（easy/medium/full）解读。

---

## 2. train/*：优化与稳定性指标

### 2.1 PPO/层级策略相关

| 指标 | 数学意义 | 训练作用 | 评价标准 |
|---|---|---|---|
| `train/policy_loss` | 总策略损失：$L_{\pi}=L_{\pi}^{low}+\lambda_h L_{\pi}^{high}$ | 反映整体策略更新方向与强度 | 绝对值无统一目标；长期精确等于 0 往往可疑（可能优势退化/数值问题） |
| `train/policy_loss_low` | 低层动作 PPO 损失：$-\mathbb{E}[\min(r_t\hat A_t,\text{clip}(r_t)\hat A_t)]$ | 驱动 course/attack 低层策略改进 | 应有波动并随收敛趋稳；长期 0 需排查 |
| `train/policy_loss_high` | 高层 mode PPO 损失（同式，换高层 ratio 和优势） | 驱动 mode 选择策略 | 长期 0 可能是高层决策掩码过少或优势退化 |
| `train/entropy` | 总熵：$H_{low}+H_{high}$ | 控制探索-利用平衡 | 初期偏高，后期逐步下降；异常上升可能策略发散 |
| `train/entropy_low` | 低层策略熵 | 低层探索强度 | 与任务复杂度匹配，过低会早收敛，过高会学不动 |
| `train/entropy_high` | 高层策略熵 | 高层模式探索强度 | 应缓慢下降，不宜骤降为 0 |

### 2.2 模仿与价值函数

| 指标 | 数学意义 | 训练作用 | 评价标准 |
|---|---|---|---|
| `train/imitation_loss` | 学生与教师策略 KL（course+attack） | 早期稳定 warm start | 有 imitation 时应先降后稳；长期升高说明偏离教师或分布漂移 |
| `train/imitation_coef` | 模仿系数（含 warmup 衰减） | 控制 RL 与 imitation 权重 | 训练前期可 >0，后期通常衰减到 0 |
| `train/value_loss` | 多头价值损失加权和 | 训练 critic 拟合回报与辅助目标 | 不应持续爆涨；尖峰可接受但需回落 |
| `train/explained_variance` | $EV=1-\frac{\mathrm{Var}(R-V)}{\mathrm{Var}(R)}$ | 评估价值函数解释度 | 接近 1 较好；长期负值或 NaN 是风险信号 |
| `train/value_target_mean` | critic 目标均值 | 监控目标尺度漂移 | 应平滑变化，突变常对应奖励尺度变化 |
| `train/value_target_std` | critic 目标标准差 | 监控目标方差 | 过大易致训练不稳 |
| `train/value_target_norm_mean` | 归一化后目标均值 | 检查归一化效果 | 期望接近 0 |
| `train/value_target_norm_std` | 归一化后目标方差 | 检查归一化效果 | 期望接近 1 |

### 2.3 梯度、RNN 与吞吐

| 指标 | 数学意义 | 训练作用 | 评价标准 |
|---|---|---|---|
| `train/actor_grad_norm` | actor 参数梯度 $\ell_2$ 范数 | 监控策略更新强度 | 持续尖峰（特别是双位数/百位）需警惕爆炸 |
| `train/critic_grad_norm` | critic 参数梯度 $\ell_2$ 范数 | 监控值函数更新强度 | 通常比 actor 大，但持续爆涨要降 LR/clip |
| `train/active_mask_ratio` | 有效样本比率 | 判断 batch 中“可学习样本”密度 | 接近 1 代表样本利用充分 |
| `train/hidden_state_init_abs_error` | chunk 初始隐状态误差 | 监控 RNN 截断/对齐正确性 | 应接近 0 |
| `train/rnn_hidden_mismatch_count` | collector 记录的隐状态不一致计数 | 监控 collector 与 trainer 状态一致性 | 应接近 0 |
| `train/rnn_hidden_max_abs_diff` | 隐状态最大偏差 | 同上 | 应很小（接近 0） |
| `train/sample_fps` | 每秒环境步数 | 训练效率 | 下降明显需查环境/IO/GPU负载 |
| `train/active_samples` | 当前更新有效样本数 | 批有效规模 | 过小会导致梯度噪声大 |
| `train/grad_steps` | 每轮反向更新步数 | 训练预算一致性 | 应与配置匹配，异常下降要排查跳过 batch |
| `train/obs_agent_id_concat_enabled` | 是否拼接 agent one-hot（0/1） | 观测设计开关 | 仅用于确认配置是否按预期生效 |

### 2.4 奖励与多头价值诊断

| 指标 | 数学意义 | 训练作用 | 评价标准 |
|---|---|---|---|
| `train/damage_reward_mean` | rollout 内伤害奖励均值 | 反映直接战斗收益 | 与 win_rate 同向更健康 |
| `train/reward_env_mean` | 环境主奖励均值 | 主优化信号强度 | 与结果指标应同向 |
| `train/reward_mode_mean` | 模式层 shaping 均值 | 高层行为塑形 | 不应长期压过 env 主奖励 |
| `train/reward_exec_mean` | 执行层 shaping 均值 | 低层机动/执行塑形 | 持续为负需复查 shaping 权重 |
| `train/kill_reward_mean` | 击杀收益均值 | 战果直接代理 | 提升通常对应胜率提升 |
| `train/survival_reward_mean` | 生存惩罚/收益均值 | 存活能力代理 | 过负说明自损严重 |
| `train/win_indicator_mean` | rollout 胜利指示均值 | 短期胜率代理 | 与 `train_episode/win_rate` 应一致趋势 |
| `train/value_contact_mean` | critic 接触头预测均值 | 监控接触子任务价值 | 长期发散/NaN 为风险 |
| `train/value_opportunity_mean` | critic 机会头预测均值 | 监控攻击机会子任务 | 同上 |
| `train/value_survival_mean` | critic 生存头预测均值 | 监控生存子任务 | 同上 |

---

## 3. train_episode/*：回合聚合战术指标

这些指标按 episode 聚合后写入 TensorBoard，通常是“训练中在线质量”最直观的信号。

### 3.1 结果与战果

| 指标 | 数学意义 | 训练作用 | 评价标准 |
|---|---|---|---|
| `train_episode/round_reward_mean` | 我方回合奖励均值 | 总体胜负收益 | 上升更好 |
| `train_episode/opponent_round_reward_mean` | 对手回合奖励均值 | 对抗强弱参照 | 下降更好 |
| `train_episode/win_flag_mean` | 回合胜利标记均值 | 胜率原始值 | 越高越好 |
| `train_episode/win_rate` | 等同 `win_flag_mean` | 主目标之一 | easy 阶段应较高，full 阶段稳步提升 |
| `train_episode/fighter_destroy_balance_end_mean` | 击毁差：蓝毁 - 红毁 | 战损平衡 | 越大越好 |
| `train_episode/red_fighter_destroyed_end_mean` | 我方损失均值 | 生存质量 | 越低越好 |
| `train_episode/blue_fighter_destroyed_end_mean` | 对手损失均值 | 打击效率 | 越高越好 |

### 3.2 战术漏斗与行为质量

| 指标 | 数学意义 | 训练作用 | 评价标准 |
|---|---|---|---|
| `train_episode/contact_frac_mean` | 接触占比 | 漏斗第一层（发现） | 过低表示侦搜/机动不足 |
| `train_episode/attack_opportunity_frac_mean` | 攻击机会占比 | 漏斗第二层（可打） | 过低说明接触转机会差 |
| `train_episode/executed_fire_action_frac_mean` | 实际执行发射占比 | 漏斗第三层（执行） | 需和机会占比联动看 |
| `train_episode/contact_to_opportunity_ratio_mean` | $\frac{opportunity}{contact}$ | 接触到机会转化率 | 越高越好 |
| `train_episode/opportunity_to_fire_ratio_mean` | $\frac{executed\_fire}{opportunity}$ | 机会到发射转化率 | 越高越好 |
| `train_episode/missed_attack_frac_mean` | 错失机会比例 | 识别“有机会不打” | 越低越好 |
| `train_episode/time_to_first_contact_mean` | 首次接触步数 | 进入战斗速度 | 越低通常越好 |
| `train_episode/visible_enemy_count_mean` | 平均可见敌机数 | 观察/态势感知代理 | 与地图/阶段相关 |
| `train_episode/nearest_enemy_distance_mean` | 最近敌机平均距离 | 机动接敌趋势 | 过大可能保守，过小可能冒进 |
| `train_episode/nearest_enemy_distance_min_mean` | 最近敌机最小距离均值 | 贴身风险代理 | 过小需结合损失率看 |

### 3.3 控制与奖励分解

| 指标 | 数学意义 | 训练作用 | 评价标准 |
|---|---|---|---|
| `train_episode/course_change_frac_mean` | 航向变化频率 | 机动活跃度 | 过低=僵化，过高=抖动 |
| `train_episode/course_unique_frac_mean` | 覆盖航向离散度 | 动作多样性 | 中等偏高更健康 |
| `train_episode/engagement_progress_reward_mean` | 接敌推进 shaping 均值 | 微观机动质量 | 长期为负需调 shaping |
| `train_episode/agent_aux_reward_mean` | agent 辅助奖励均值 | 局部行为塑形 | 不应主导主奖励 |
| `train_episode/reward_env_mean` | 环境主奖励均值 | 主目标信号 | 与胜率同向 |
| `train_episode/reward_mode_mean` | 高层 shaping 均值 | mode 学习信号 | 不应长期反向拉扯 |
| `train_episode/reward_exec_mean` | 低层 shaping 均值 | 执行层学习信号 | 同上 |
| `train_episode/invalid_action_frac_mean` | 非法动作比例 | 掩码/策略约束健康度 | 应接近 0 |
| `train_episode/episode_len_mean` | 回合长度均值 | 节奏/结束模式 | 需结合 map 与 max_step 解释 |

---

## 4. train_action/*：动作分布与塌缩检测

| 指标 | 数学意义 | 训练作用 | 评价标准 |
|---|---|---|---|
| `train_action/course_action_freq_XX` | 航向动作 XX 在有效样本中占比 | 检查航向动作是否塌缩 | 单一动作长期接近 1.0 通常异常 |
| `train_action/attack_action_freq_XX` | 攻击动作 XX 占比 | 检查攻击策略多样性 | 全部压到 no-fire 说明保守塌缩 |
| `train_action/attack_legal_usage_min_nonzero` | 合法非零攻击动作中最小使用率 | 检查“可用但不用”极端动作 | 越接近 0 越像塌缩 |
| `train_action/attack_legal_usage_mean_nonzero` | 合法非零攻击动作平均使用率 | 检查整体攻击利用 | 过低表示机会利用差 |
| `train_action/alive_sample_count` | 用于统计的存活样本数 | 指标置信度 | 太小则频率指标不稳定 |

---

## 5. eval/* 与课程学习

### 5.1 eval/*

- `eval/*` 与 `train_episode/*` 基本同一组键，只是来源于独立评估回合。
- 额外有：
  - `eval/eval_wall_time_sec`：本次评估耗时。

建议：主要决策依据用 `eval/*`，`train_episode/*` 作为在线趋势信号。

### 5.2 curriculum/stage_id

| 指标 | 含义 |
|---|---|
| `curriculum/stage_id` | 1=easy, 2=medium, 3=full |

当出现性能拐点时，先看 `stage_id` 是否发生跳变，再判断是“课表冲击”还是“数值/策略退化”。

---

## 6. 项目内可执行评价标准（建议直接用）

结合项目验收脚本 `scripts/check_hmarl_acceptance.py`：

- Level 1（接触稳定）：`contact_frac_mean >= 0.015`
- Level 2（攻击机会）：`attack_opportunity_frac_mean >= 0.005`
- Level 3（结果目标）：`win_rate >= 0.6`

建议把这三条作为硬门槛，再叠加以下稳定性软门槛：

1. `train/explained_variance` 不应长期 NaN
2. `train/actor_grad_norm` 与 `train/critic_grad_norm` 不应持续爆涨
3. `train/policy_loss_low`、`train/policy_loss_high` 不应长期精确为 0
4. `train_action/attack_legal_usage_min_nonzero` 不应长期贴近 0

---

## 7. 快速诊断对照表

| TensorBoard 现象 | 常见原因 | 优先排查 |
|---|---|---|
| `policy_loss_*` 长期 0 + `entropy` 反向上升 | 优势退化或数值问题 | 检查 EV、梯度、是否出现 NaN |
| `explained_variance` 变 NaN | value target 溢出或参数污染 | 降 LR、加强数值裁剪、检查奖励尺度 |
| `win_rate` 下滑但 `stage_id` 未变 | 不是课表切换，可能策略退化 | 重点查 `train/*` 数值稳定性 |
| `attack_legal_usage_*` 极低 | 攻击策略塌缩到 no-fire | 调整 entropy、奖励分解权重 |
| `rnn_hidden_mismatch_*` 上升 | RNN 状态对齐问题 | 检查 collector/trainer hidden 传递链 |

---

## 8. 一句话使用建议

先看 `eval/win_rate` 与验收三门槛，再看 `train/explained_variance` + 梯度范数判断“是否在健康学习”，最后用 `train_action/*` 和战术漏斗指标定位“卡在哪一层（接触/机会/执行）”。
