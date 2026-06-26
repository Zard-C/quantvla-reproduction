# Phase 17: action channel 敏感性分析

## 结论摘要

这轮分析回答的是：闭环敏感扰动到底落在哪些动作通道上。结论不是某一个单独通道有决定性问题，而是 `x/y/z/roll/pitch/yaw` 这 6 个连续动作维度的联合小偏移改变了闭环轨迹。

关键证据：

- `gripper-only` oracle 是 `0/2`，说明只修夹爪救不回失败。
- `translation-only` 和 `rotation-only` oracle 都是 `0/2`，说明单独修位置或姿态也救不回。
- `continuous-action` oracle 是 `2/2`，而且不替换 gripper，说明连续 6D 联合修正足以恢复两个病例。
- `full-action` oracle 也是 `2/2`，但相比 continuous-action 没有额外收益，说明 gripper 不是必要修正项。

因此这里的 channel sensitivity 更像多维协同敏感性，而不是单轴敏感性。

## Oracle 成败矩阵

| setting | oracle keys | success | task4:init9 steps | task6:init8 steps | client p50 |
| --- | --- | --- | --- | --- | --- |
| gripper_only | `gripper` | 0/2 | 990 | 990 | 251.2 ms |
| full_action | `x,y,z,roll,pitch,yaw,gripper` | 2/2 | 224 | 649 | 252.4 ms |
| translation_only | `x,y,z` | 0/2 | 990 | 990 | 252.4 ms |
| rotation_only | `roll,pitch,yaw` | 0/2 | 990 | 990 | 252.0 ms |
| continuous_action | `x,y,z,roll,pitch,yaw` | 2/2 | 224 | 649 | 252.7 ms |

这个矩阵的逻辑很强：如果根因是夹爪通道，`gripper-only` 应该救回来；如果根因是纯位置或纯姿态，`translation-only` 或 `rotation-only` 应该至少救回一部分。但实际只有连续 6D 一起替换才成功。

## 同观测逐维差异

| channel | mean abs | p50 abs | p99 abs | max abs |
| --- | --- | --- | --- | --- |
| x | 0.000679 | 0.000458 | 0.002747 | 0.054932 |
| y | 0.000638 | 0.000458 | 0.003351 | 0.006895 |
| z | 0.00073 | 0.000479 | 0.003021 | 0.039139 |
| roll | 0.00016 | 0.000131 | 0.000524 | 0.002881 |
| pitch | 0.000202 | 0.000146 | 0.000585 | 0.033631 |
| yaw | 0.000175 | 0.000181 | 0.000725 | 0.001088 |

逐维看，`x/z/pitch` 的 max 明显大，来自 `task6:init8` 的 step253 局部 outlier；`y` 的 p99 也偏高。`roll/yaw` 较小，但它们仍属于连续动作组合的一部分。这里不能把结论简化成只修 `x` 或只修 `z`，因为 oracle 结果已经显示 translation-only 不够。

## 动作组 L2 敏感性

| group | mean L2 | p50 L2 | p99 L2 | max L2 |
| --- | --- | --- | --- | --- |
| translation | 0.001449 | 0.001233 | 0.00387 | 0.0678 |
| rotation | 0.00039 | 0.00033 | 0.000915 | 0.033757 |
| continuous | 0.001533 | 0.001307 | 0.003899 | 0.075739 |

组级别看，translation L2 通常更大，rotation L2 较小但在 outlier 步与 translation 同向出现。闭环系统吃到的是完整 6D action，不是独立通道；因此较小的姿态误差也可能改变末端接触法向、物体受力和后续视觉状态。

## 每个病例的 top channel event

| case | success | steps | top continuous | top translation | top rotation |
| --- | --- | --- | --- | --- | --- |
| task4:init9 | True | 224 | 191 / z / 0.005281 | 191 / z / 0.005276 | 210 / yaw / 0.001317 |
| task6:init8 | True | 649 | 253 / x / 0.075739 | 253 / x / 0.0678 | 253 / pitch / 0.033757 |

`task6:init8` 的 top continuous event 正是之前定位的 step253，主导维度是 `x`，但同时带着 `z/pitch` 分量。`task4:init9` 没有同等大的单点 spike，更像较早、多步的小偏移累积。

## 闭环 first-divergence 里的主导维度

| case | first cont >=0.005 | first cont >=0.01 | first cont >=0.05 | max cont before gripper split | first gripper split |
| --- | --- | --- | --- | --- | --- |
| task4:init9 | 8 / y / 0.021057 | 8 / y / 0.021057 | 55 / z / 0.10849 | 23 / y / 0.037537 | 46 / x / 0.036524 |
| task6:init8 | 40 / x / 0.005035 | 54 / z / 0.048523 | 137 / z / 0.056763 | 137 / z / 0.056763 | 244 / z / 0.173492 |

这张表连接了 open-loop channel diff 与 closed-loop 分叉。`task4:init9` 早期先由 `y` 触发连续动作阈值，随后 `z/x` 在位姿分叉和夹爪相位分裂附近占主导；`task6:init8` 早期是 `x/z` 主导。也就是说，敏感扰动主要集中在末端位姿相关的连续维度，而不是原始 gripper channel。

## Gripper 通道排除证据

| case | oracle success | steps | bin mismatch | mean abs diff | max abs diff | top step / diff / bin mismatch |
| --- | --- | --- | --- | --- | --- | --- |
| task4:init9 | False | 990 | 0 | 0.001007 | 0.061035 | 881 / -0.061035 / False |
| task6:init8 | False | 990 | 0 | 0.000946 | 0.013519 | 336 / 0.013519 / False |

两个 gripper-only oracle trace 合计 1980 步，二值 mismatch 总数是 0。这说明在 compiled 已经走到的 observation 上，primary gripper 与 eager oracle gripper 的二值判断一致。gripper mismatch 更像轨迹分叉后的表征，而不是最初触发源。

## 工程含义

后续如果要做闭环误差修正，优先级应该是：

1. 监控连续 6D action 的联合 drift，尤其是 translation L2 与 pitch 同步异常。
2. 不要只调 gripper threshold / hysteresis；这会修表征，不修触发源。
3. 局部 fallback 或校准目标应该围绕完整连续动作向量做，而不是单独修某一轴。
4. 对 compile / FP4 后端的验收指标应加入 `continuous L2 p99/max` 和 `first-divergence`，不能只看平均 action MSE。

## 限制

当前 oracle 样本只有两个重点病例，因此结论是机制性证据，不是总体统计结论。它足以指导下一轮工程验证：先保护连续 6D，再看闭环成功率是否恢复；但不能宣称所有 LIBERO 失败都由同样通道组合导致。

## 产物

- JSON: `toy_quantvla/results/phase17_action_channel_sensitivity.json`
- Report: `docs/phase17_channel/action_channel_sensitivity_zh.md`
