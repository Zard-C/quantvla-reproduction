# Phase 5 收尾报告：消融实验、Trace 病例与阶段结论

## 1. 阶段定位

Phase 5 的目标不是证明真实部署已经加速，而是回答一个更前置的问题：

```text
如果只在行为层做 selective W4A8 fake quantization，
GR00T/LIBERO 闭环策略会不会在真实 simulator rollout 里崩掉？
ATM/OHB 到底是稳定补偿，还是会重新分配成功/失败样本？
```

现在结论已经比较清楚：

```text
llm_dit_mlp selective W4A8 fake quantization 行为上可行。
它没有把策略搞崩，整体成功率还略高于 FP16。

但 ATM/OHB 不是单调增强器。
它们会改变闭环轨迹，救活一部分失败样本，也制造一部分新失败样本。
```

因此 Phase 5 的核心收获不是“量化一定更好”，而是：

```text
量化扰动会诱导闭环轨迹重分配。
成功率变化 = 被救活的失败样本 - 被破坏的成功样本。
```

## 2. 大规模消融结果回顾

主消融使用 `libero_10`、init `0..14`，每个 policy 共 `150` 个 episode。

| policy | successes | success rate | delta vs FP16 |
|---|---:|---:|---:|
| FP16 | 108/150 | 72.0% | +0 |
| `llm_dit_mlp + none` | 113/150 | 75.3% | +5 |
| `llm_dit_mlp + atm` | 114/150 | 76.0% | +6 |
| `llm_dit_mlp + ohb` | 116/150 | 77.3% | +8 |
| `llm_dit_mlp + atm_ohb` | 114/150 | 76.0% | +6 |

Paired outcome 相对 FP16：

| policy | repaired failures | new regressions | same success | same failure | net |
|---|---:|---:|---:|---:|---:|
| none | 21 | 16 | 92 | 21 | +5 |
| atm | 20 | 14 | 94 | 22 | +6 |
| ohb | 22 | 14 | 94 | 20 | +8 |
| atm_ohb | 19 | 13 | 95 | 23 | +6 |

这个表是 Phase 5 最重要的统计证据。它说明总成功率的提升不是每条轨迹都更好，而是成功/失败 basin 的重新分配。

## 3. Trace 重跑设置

为了从行为标签进入轨迹证据，我们重跑了一组小规模 trace。

```text
run id: trace_20260606_135425
remote trace root: /tmp/quantvla_trace_cases_trace_20260606_135425
local trace root: trace_cases/quantvla_trace_cases_trace_20260606_135425
local log root: trace_cases/phase5_trace_trace_20260606_135425
modes: fp16, none, ohb, atm_ohb
cases: 7 task-init pairs
total traces: 28/28
finished: 2026-06-06 15:22:51 +08:00
```

Trace 记录字段包括：

```text
task_id, init_index, success, num_steps, exception
pre/post robot0_eef_pos
pre/post robot0_eef_quat
pre/post robot0_gripper_qpos
raw_action
libero_action
reward, done
```

注意：这组 trace 是病例取证，不是新的总体成功率估计。它的任务是解释“为什么会 flip”，不是替代 `150` episode 的消融统计。

## 4. Trace Outcome 表

| case | FP16 | none | OHB | ATM+OHB |
|---|---:|---:|---:|---:|
| task0 init3 | success, 253 | fail, 990 | success, 240 | success, 349 |
| task4 init10 | success, 281 | success, 245 | success, 264 | success, 257 |
| task6 init1 | success, 230 | success, 252 | fail, 990 | success, 369 |
| task8 init0 | success, 617 | fail, 990 | success, 548 | fail, 990 |
| task8 init7 | fail, 990 | fail, 990 | success, 636 | success, 852 |
| task8 init9 | success, 409 | fail, 990 | fail, 990 | fail, 990 |
| task9 init9 | success, 272 | success, 262 | success, 285 | success, 256 |

数字表示 `success/fail, num_steps`。`990` 通常对应 rollout 跑到 horizon 后触发 `executing action in terminated episode`，这仍然是失败标签，不是 inference server crash。

Trace 小集合的 aggregate：

| mode | successes | avg steps |
|---|---:|---:|
| FP16 | 6/7 | 436.0 |
| none | 3/7 | 674.1 |
| OHB | 5/7 | 564.7 |
| ATM+OHB | 5/7 | 580.4 |

这个 aggregate 不能和 `150` episode 主消融直接比较，因为它是刻意挑选的病例集合。

## 5. Trace 轨迹特征

为了避免只看成功标签，我们提取了几个简单轨迹特征：

```text
eef_path:
  末端执行器路径长度

eef_net:
  起点到终点的末端净位移

mean_trans:
  LIBERO action 前三维平移动作的平均 L2 norm

close_frac:
  gripper action > 0 的比例

grip_switches:
  gripper action 正负符号切换次数
```

| case | mode | success | steps | eef_path | eef_net | mean_trans | close_frac | grip_switches |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| task0 init3 | fp16 | 1 | 253 | 1.782 | 0.299 | 0.544 | 0.44 | 4 |
| task0 init3 | none | 0 | 990 | 3.085 | 0.321 | 0.300 | 0.56 | 91 |
| task0 init3 | ohb | 1 | 240 | 1.754 | 0.323 | 0.569 | 0.45 | 4 |
| task0 init3 | atm_ohb | 1 | 349 | 1.790 | 0.277 | 0.478 | 0.54 | 3 |
| task6 init1 | fp16 | 1 | 230 | 1.312 | 0.264 | 0.476 | 0.50 | 3 |
| task6 init1 | none | 1 | 252 | 1.328 | 0.262 | 0.443 | 0.48 | 5 |
| task6 init1 | ohb | 0 | 990 | 1.804 | 0.251 | 0.263 | 0.54 | 31 |
| task6 init1 | atm_ohb | 1 | 369 | 1.483 | 0.272 | 0.388 | 0.69 | 11 |
| task8 init0 | fp16 | 1 | 617 | 2.129 | 0.344 | 0.308 | 0.57 | 9 |
| task8 init0 | none | 0 | 990 | 2.408 | 0.340 | 0.285 | 0.43 | 30 |
| task8 init0 | ohb | 1 | 548 | 2.119 | 0.324 | 0.323 | 0.57 | 5 |
| task8 init0 | atm_ohb | 0 | 990 | 0.948 | 0.227 | 0.146 | 0.00 | 0 |
| task8 init7 | fp16 | 0 | 990 | 1.662 | 0.277 | 0.246 | 0.13 | 4 |
| task8 init7 | none | 0 | 990 | 1.807 | 0.431 | 0.277 | 0.74 | 3 |
| task8 init7 | ohb | 1 | 636 | 2.115 | 0.319 | 0.273 | 0.61 | 3 |
| task8 init7 | atm_ohb | 1 | 852 | 2.232 | 0.337 | 0.273 | 0.61 | 20 |
| task8 init9 | fp16 | 1 | 409 | 2.076 | 0.346 | 0.446 | 0.33 | 9 |
| task8 init9 | none | 0 | 990 | 1.535 | 0.312 | 0.271 | 0.14 | 5 |
| task8 init9 | ohb | 0 | 990 | 0.525 | 0.306 | 0.221 | 0.00 | 0 |
| task8 init9 | atm_ohb | 0 | 990 | 1.699 | 0.272 | 0.249 | 0.08 | 6 |

## 6. 关键病例解读

### 6.1 task0 init3：none 退化，OHB/ATM+OHB 修复

```text
FP16: success, 253 steps
none: fail, 990 steps
OHB: success, 240 steps
ATM+OHB: success, 349 steps
```

`none` 的末端路径长度达到 `3.085`，明显高于 FP16 的 `1.782`，同时 gripper switches 达到 `91`。这不像是简单“慢一点”，更像进入了抖动或反复尝试的失败轨迹。

OHB 和 ATM+OHB 都把轨迹拉回成功 basin。这里补偿确实像稳定器。

### 6.2 task6 init1：OHB 单独失败，ATM+OHB 恢复

```text
FP16: success, 230 steps
none: success, 252 steps
OHB: fail, 990 steps
ATM+OHB: success, 369 steps
```

OHB 的 `mean_trans` 从 FP16 的 `0.476` 降到 `0.263`，gripper switches 从 `3` 增加到 `31`。这说明单独 OHB 在这个 case 上可能让动作变得更保守，同时夹爪控制出现更多反复。

ATM+OHB 能恢复成功，但步数变长到 `369`，说明它不是回到原始 FP16 轨迹，而是走了另一条能成功的轨迹。

### 6.3 task8 init7：最有价值的“修复病例”

```text
FP16: fail, 990 steps
none: fail, 990 steps
OHB: success, 636 steps
ATM+OHB: success, 852 steps
```

这是 Phase 5 trace 里最重要的正例。FP16 和 raw quant 都失败，但 OHB 与 ATM+OHB 成功。

轨迹特征上，成功版本的 `eef_path` 明显更长：

```text
FP16: 1.662
none: 1.807
OHB: 2.115
ATM+OHB: 2.232
```

同时 `close_frac` 从 FP16 的 `0.13` 提高到 OHB/ATM+OHB 的 `0.61`。这说明补偿不是只降低数值误差，而是改变了接触和夹爪时序，让策略进入了新的成功路径。

### 6.4 task8 init0：OHB 修复，ATM+OHB 过度改变

```text
FP16: success, 617 steps
none: fail, 990 steps
OHB: success, 548 steps
ATM+OHB: fail, 990 steps
```

OHB 成功并且步数比 FP16 更短，是一个漂亮的修复例子。

但 ATM+OHB 的 `eef_path` 只有 `0.948`，`mean_trans` 只有 `0.146`，`close_frac` 为 `0.00`。这很像动作被压得过保守，或者早期策略进入了几乎不进行有效夹取的失败分支。

这说明 ATM+OHB 组合不一定优于 OHB。ATM 改 attention routing，OHB 改 residual energy，两者耦合后可能过校正。

### 6.5 task8 init9：所有量化版本失败

```text
FP16: success, 409 steps
none: fail, 990 steps
OHB: fail, 990 steps
ATM+OHB: fail, 990 steps
```

这是明确的 regression 病例。OHB 和 ATM+OHB 都没有救回来。

尤其 OHB 的 `eef_path` 只有 `0.525`，`close_frac` 为 `0.00`，说明它并不是“快失败”，而是进入了几乎没有有效操作的失败轨迹。

## 7. 和数学解释的关系

我们之前把闭环量化扰动写成：

```text
FP16:
  s_{t+1} = F(s_t, pi(s_t))

Quantized:
  s'_{t+1} = F(s'_t, pi(s'_t) + delta_a(s'_t))
```

线性化后：

```text
e_{t+1} ≈ A_t e_t + B_t delta_a_t
```

这里的关键是：量化误差不是一次性 output MSE，而是会进入环境反馈。只要早期动作稍微改变，物体接触、夹爪闭合、遮挡、下一帧观测都会随之改变，后续策略就不再沿着同一条轨迹走。

Trace 结果支持这个解释：

```text
task8 init7:
  OHB/ATM+OHB 不是贴近 FP16 轨迹后成功，
  而是走出了一条新的成功轨迹。

task8 init0:
  OHB 能进入成功 basin，
  ATM+OHB 却进入低动作/不夹取的失败 basin。

task0 init3:
  none 出现明显 gripper 抖动，
  OHB/ATM+OHB 把它拉回稳定成功轨迹。
```

所以 Phase 5 的理论解释应该落在“闭环轨迹重分配”，而不是“量化噪声一定正则化”或“补偿一定降低误差”这种单调叙事上。

## 8. Phase 5 最终结论

Phase 5 可以收尾，结论如下：

```text
1. llm_dit_mlp selective W4A8 fake quantization 行为层验证通过。

2. 在 150 episode 主消融中，所有量化版本都达到或超过 FP16：
   FP16 108/150
   none 113/150
   atm 114/150
   ohb 116/150
   atm_ohb 114/150

3. OHB 是当前最好的单独补偿模式，但不是单调更好。

4. ATM 高方差：
   对 task4 帮助明显，
   对 task8 伤害明显。

5. ATM+OHB 不保证强于 OHB。
   两者分别改变 attention routing 和 attention residual energy，
   组合后可能修复，也可能过校正。

6. Trace 病例确认了量化/补偿会改变闭环接触序列、夹爪时序和轨迹 basin。

7. 当前仍然是 fake quant 行为验证。
   它不证明真实 packed integer kernel 的速度、显存或端到端延迟收益。
```

一句话版本：

```text
Phase 5 证明 selective W4A8 在 GR00T/LIBERO 上行为可行；
ATM/OHB 的作用更像闭环轨迹重分配器，而不是保证单调收益的误差消除器。
```

## 9. Phase 6 边界

下一阶段应该进入工程化验证，而不是继续在 fake quant 上打转。

建议 Phase 6 拆成两条线：

```text
Phase 6A: real W4A16 kernel
  先做 Triton packed int4 weight + fp16 activation GEMM。
  对齐当前 fake quant 的 per-output-channel scale。
  目标是验证真实 kernel 的数值误差、速度和显存收益。

Phase 6B: trace 深挖
  从已拉回的 JSON trace 中提取关键帧、action/eef 曲线、gripper 时序。
  必要时只重跑少数病例，不再大规模扫 simulator。
```

工程上优先级更高的是 Phase 6A。因为 Phase 5 已经说明行为 gate 过了；真正未证明的是部署收益。

