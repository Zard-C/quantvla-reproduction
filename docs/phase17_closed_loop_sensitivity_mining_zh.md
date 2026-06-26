# Phase 17 闭环敏感扰动离线挖掘

本报告只使用仓库中已有的 rollout、trace、oracle、compile-scope 和 latency 结果；没有重新跑推理，也不需要 5090/GPU。

## 核心结论

当前数据已经支持把问题改写成：先识别哪些扰动是闭环敏感的，再决定把校准、残差修正、mixed precision 或 fallback 预算花在哪里。

已有证据指向四类敏感性：

1. task/init 敏感性：task4 和 task8 在不同补偿模式下波动最大。
2. 时刻敏感性：同观测误差很小，但闭环里早期小差异会在接触/夹爪阶段放大。
3. 动作子空间敏感性：gripper 单独不是根因，translation 或 rotation 单独也不够，连续 6D 联合修正能恢复两个病例。
4. 模块/后端敏感性：blocks 8-15 eager-island 后端有明显 p50 加速，但 matched rollout 有净回归。

## 1. Task/Init 敏感性

| task | none | atm | ohb | volatility | best | worst |
|---|---|---|---|---|---|---|
| 4 | 8/15 | 13/15 | 12/15 | 5 | atm | none |
| 8 | 9/15 | 4/15 | 6/15 | 5 | none | atm |
| 0 | 10/15 | 11/15 | 13/15 | 3 | ohb | none |
| 2 | 13/15 | 14/15 | 15/15 | 2 | ohb | none |
| 3 | 15/15 | 13/15 | 13/15 | 2 | none | atm |
| 6 | 9/15 | 10/15 | 8/15 | 2 | atm | ohb |

最敏感的是 task4 和 task8：二者在 `none/atm/ohb` 三种模式之间都有 5/15 的成功率摆动。task4 被 ATM/OHB 明显修复，task8 被 ATM/OHB 明显打坏。这是典型的轨迹重分配，而不是单调提升。

| comparison | repairs | regressions | same success | same failure | net |
|---|---|---|---|---|---|
| `atm` vs `none` | 14 | 13 | 100 | 23 | 1 |
| `ohb` vs `none` | 16 | 13 | 100 | 21 | 3 |
| `ohb` vs `atm` | 15 | 13 | 101 | 21 | 2 |

同一批 150 个 task-init pair 里，ATM/OHB 的净收益很小，但 repairs/regressions 都有两位数。这说明聚合成功率低估了行为 churn。

| comparison | churn | churn rate | exact McNemar p |
|---|---|---|---|
| `atm` vs `none` | 27 | 18.0% | 1.000 |
| `ohb` vs `none` | 29 | 19.3% | 0.711 |
| `ohb` vs `atm` | 28 | 18.7% | 0.851 |

这些 p-value 都不支持把小净增益解释为稳定 dominance；但 churn rate 接近 18%-19%，说明闭环轨迹重分配本身非常明显。

最敏感的 init slice：

| init | none | atm | ohb | volatility | best | worst |
|---|---|---|---|---|---|---|
| 8 | 9/10 | 5/10 | 9/10 | 4 | none | atm |
| 1 | 6/10 | 6/10 | 9/10 | 3 | ohb | none |
| 5 | 7/10 | 8/10 | 10/10 | 3 | ohb | none |
| 0 | 7/10 | 6/10 | 5/10 | 2 | none | ohb |
| 2 | 8/10 | 6/10 | 7/10 | 2 | none | atm |

FP16 vs `llm_dit_mlp + atm_ohb` 在 init5-14 上也呈现同样结构：

| task | FP16 | quant | delta |
|---|---|---|---|
| 4 | 4/10 | 8/10 | 4 |
| 9 | 9/10 | 7/10 | -2 |
| 6 | 3/10 | 5/10 | 2 |
| 1 | 8/10 | 10/10 | 2 |
| 2 | 9/10 | 8/10 | -1 |
| 8 | 2/10 | 3/10 | 1 |

init5-14 的净 `+6/100` 来自 `14` 个 repair 减去 `8` 个 regression。 churn rate 为 `22.0%`，exact McNemar p 为 `0.286`。

## 2. First-Divergence 敏感时刻

病例：`task4:init9`，baseline success `224` steps，compiled fail `990` steps。

| event | step | dim | action max | EEF pre diff |
|---|---|---|---|---|
| first_action_max_ge_0.005 | 8 | y | 0.0211 | 0.048 mm |
| first_pre_pos_l2_ge_0.5mm | 15 | z | 0.0020 | 0.531 mm |
| first_pre_pos_l2_ge_1mm | 49 | z | 0.0371 | 1.113 mm |
| first_pre_pos_l2_ge_5mm | 57 | z | 0.2307 | 5.803 mm |
| first_pre_pos_l2_ge_10mm | 59 | x | 0.2673 | 11.053 mm |
| first_gripper_binary_mismatch | 46 | gripper | 2.0000 | 0.609 mm |

最大 EEF pre-position 分离出现在 step 208，约 `172.6 mm`。关键点是：第一次明显 action diff 出现在 step 8，而 gripper 二值化相反出现在 step 46；这更像闭环放大链条，而不是单步数值爆炸。

## 3. Action-Channel Oracle

| setting | oracle keys | success | task4:init9 | task6:init8 | client p50 |
|---|---|---|---|---|---|
| gripper_only | gripper | 0/2 | 990 | 990 | 251.2 ms |
| full_action | x,y,z,roll,pitch,yaw,gripper | 2/2 | 224 | 649 | 252.4 ms |
| translation_only | x,y,z | 0/2 | 990 | 990 | 252.4 ms |
| rotation_only | roll,pitch,yaw | 0/2 | 990 | 990 | 252.0 ms |
| continuous_action | x,y,z,roll,pitch,yaw | 2/2 | 224 | 649 | 252.7 ms |

这张表把 gripper 从“疑似根因”降级成“分叉表征”：gripper-only oracle 仍 0/2，continuous 6D oracle 2/2 成功且不需要替换 gripper。闭环敏感扰动更像 translation+rotation 的联合小偏移。

| channel | mean abs | p99 abs | max abs |
|---|---|---|---|
| x | 0.000679 | 0.002747 | 0.054932 |
| y | 0.000638 | 0.003319 | 0.006895 |
| z | 0.000730 | 0.002975 | 0.039139 |
| roll | 0.000160 | 0.000524 | 0.002881 |
| pitch | 0.000202 | 0.000585 | 0.033631 |
| yaw | 0.000175 | 0.000725 | 0.001088 |

## 4. Module/Backend 敏感性

| policy | success | calls | client mean | client p50 | client p90 | server mean | server p50 | server p90 |
|---|---|---|---|---|---|---|---|---|
| FP16 baseline | 16/30 | 20050 | 153.9 ms | 161.1 ms | 166.5 ms | 149.0 ms | 156.2 ms | 161.5 ms |
| blocks 8-15 eager island | 13/30 | 21316 | 108.0 ms | 81.0 ms | 150.3 ms | 103.3 ms | 76.5 ms | 145.5 ms |

30-case matched set 的逐 case 翻转：

| case | baseline | blocks 8-15 eager island | direction |
|---|---|---|---|
| 4:9 | success | fail | regression |
| 6:8 | success | fail | regression |
| 8:4 | fail | success | rescue |
| 8:7 | success | fail | regression |
| 8:8 | success | fail | regression |

这条后端路线有接近 2x 的 p50 速度收益，但 30-case 成功率从 16/30 到 13/30，说明不能只看 latency。

Phase16 continuous diff sweep 中可用于比较的 scope：

| target | success | drift steps | cont l2 p99 | trans l2 p99 | rot l2 p99 | speed ratio p50 |
|---|---|---|---|---|---|---|
| action_head_dit_ff_all | 1/2 | 484 | 0.003216 | 0.003213 | 0.000683 | 1.282 |
| action_head_dit_attn_all | 1/2 | 484 | 0.003223 | 0.003223 | 0.000612 | 1.114 |
| action_head_model_blocks_6_15_eager | 1/2 | 484 | 0.004223 | 0.004220 | 0.000941 | 0.910 |
| action_head_model_blocks_8_15_eager | 1/2 | 484 | 0.004657 | 0.004600 | 0.000920 | 0.838 |
| action_head_model | 1/2 | 484 | 0.004995 | 0.004943 | 0.001070 | 0.497 |

## 5. 现在能形成的 Risk Map

| axis | high-risk evidence | engineering implication |
|---|---|---|
| task | task4/task8 volatility = 5/15 | 校准/评估必须覆盖这些边界任务 |
| init | init-level 成功率非单调，ATM init8 从 9/10 到 5/10 | 不能用少量 init smoke test 代表整体 |
| time | step 8 小 action diff，step 46 gripper 分叉，后续 EEF 分离到厘米级 | trace 应记录 first divergence 和接触窗口 |
| action | continuous 6D oracle 2/2，单独 gripper/translation/rotation 0/2 | residual correction 应优先修连续动作联合方向 |
| module/backend | blocks 8-15 eager island p50 加速约 2x，但 16/30 -> 13/30 | 低精度/compile scope 必须同时过闭环行为门槛 |

## 后续不跑 GPU 也能继续做的事

1. 从 request trace 中抽更多 step-level action diff，扩展 first-divergence 样本数。
2. 把 keyframe 病例按接触失败、抓取时机、目标边界、多物体干扰分类。
3. 对 150-case paired outcomes 做 bootstrap/McNemar，量化 aggregate gap 与 churn 的不确定性。
4. 把 `state × action direction × module` 的 risk map 接入论文 discussion，作为 closed-loop correction 的数据依据。

## Source Files

- `docs/phase5_llm_dit_mlp_ablation_init0_14.md`
- `docs/phase5_init5_14_generalization.md`
- `docs/phase13_block_island_30case_mem_v1.md`
- `toy_quantvla/results/phase14_first_divergence_blocks_8_15_4_9_analysis.json`
- `toy_quantvla/results/phase15_action_channel_oracle_summary.json`
- `toy_quantvla/results/phase16_compile_scope_continuous_4_9_6_8_s260_v1_summary.json`
