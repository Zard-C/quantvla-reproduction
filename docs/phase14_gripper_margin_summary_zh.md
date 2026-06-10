# Phase 14 Gripper Margin 诊断小结

本轮目标是继续解释 `action_head_model_blocks_8_15_eager` 的现象：

```text
torch.compile 的速度收益是真实存在的，但部分 LIBERO 闭环 rollout 从 FP16 success 变成 compiled fail。
我们重点检查失败是否来自 gripper 二值化阈值附近的微小抖动。
```

结论比较明确：

```text
不是简单的 gripper threshold jitter。
同观测下 gripper bin 没有翻转；闭环失败时，gripper 出现的是接近饱和的模式切换时序错位。
```

## 实验文件

诊断脚本：

```text
toy_quantvla/phase14_gripper_margin_analysis.py
```

自动汇总：

```text
docs/phase14_gripper_margin_blocks_8_15_4_9_6_8.md
toy_quantvla/results/phase14_gripper_margin_blocks_8_15_4_9_6_8_analysis.json
```

新增 `task6:init8` 结果：

```text
docs/phase14_gripper_margin_blocks_8_15_6_8_online_v1.md
toy_quantvla/results/phase14_gripper_margin_blocks_8_15_6_8_online_v1.json
toy_quantvla/results/phase14_gripper_margin_blocks_8_15_6_8_online_trace_task06_init08.json
toy_quantvla/results/phase14_gripper_margin_blocks_8_15_task06_init08_closedloop_trace.json
toy_quantvla/results/phase14_gripper_margin_blocks_8_15_6_8_closedloop_v1_compiled_client_latency.json
toy_quantvla/results/phase14_gripper_margin_blocks_8_15_6_8_closedloop_v1_compiled_server_latency.json
toy_quantvla/results/phase14_gripper_margin_blocks_8_15_6_8_closedloop_v1_compiled_server_prepare.json
```

## 口径

本轮比较两类 trace：

1. Same-observation drift：环境由 eager FP16 控制，compiled 只在同一个 live observation 上旁路算 action。
2. Closed-loop pair：baseline 和 compiled 分别闭环跑同一个 task/init，再按 policy step 对齐比较。

`raw gripper` 的二值化阈值按 `0.5` 诊断。最终送给 LIBERO 的 gripper action 是二值量，通常表现为 `-1` 和 `+1`。

## Same-Observation 结论

| case | success | steps | gripper bin mismatch | max raw gripper diff | min compiled margin to 0.5 |
|---|---:|---:|---:|---:|---:|
| `task4:init9` | true | 224 | 0 | 0.00317 | 0.47855 |
| `task6:init8` | true | 649 | 0 | 0.00488 | 0.47786 |

这说明两点：

1. 在同一个 observation 上，compiled path 没有把 gripper 二值动作直接打反。
2. gripper raw 值离 `0.5` 阈值很远，最小 margin 也接近 `0.48`。它基本在接近 `0` 或接近 `1` 的饱和状态之间切换。

所以，失败不是那种 `0.499` 和 `0.501` 在阈值附近来回抖导致的。

## Closed-Loop 结论

| case | FP16 baseline | compiled | common steps | gripper bin mismatch | first mismatch | max EEF pre-pos L2 |
|---|---:|---:|---:|---:|---:|---:|
| `task4:init9` | success | fail | 224 | 26 | 46 | 172.6 mm |
| `task6:init8` | success | fail | 649 | 142 | 244 | 169.0 mm |

两个 case 的失败都出现了 gripper 二值动作错位，但错位不是边界抖动，而是“该开/该合”的时序进入了不同相位。

### task4:init9

first gripper mismatch 在 policy step `46`：

| step | baseline raw gripper | compiled raw gripper | baseline LIBERO gripper | compiled LIBERO gripper |
|---:|---:|---:|---:|---:|
| 45 | 0.99850 | 0.99850 | -1 | -1 |
| 46 | 0.98392 | 0.00204 | -1 | +1 |
| 47 | 0.00712 | 0.00615 | +1 | +1 |

这里 compiled 相当于提前进入了下一段 gripper 模式。step 46 不是 `0.5` 附近的微扰，而是 `0.98` 对 `0.00` 的模式差异。二值化之后，gripper 维度动作差直接变成 `2.0`。

### task6:init8

first gripper mismatch 在 policy step `244`：

| step | baseline raw gripper | compiled raw gripper | baseline LIBERO gripper | compiled LIBERO gripper |
|---:|---:|---:|---:|---:|
| 243 | 0.00847 | 0.00651 | +1 | +1 |
| 244 | 0.00134 | 0.98907 | +1 | -1 |
| 245 | 0.01191 | 0.00996 | +1 | +1 |

这个 case 是反方向：compiled 突然进入 close 模式，而 baseline 仍然保持 open。它同样不是 threshold jitter，而是闭环状态已经把策略带到了不同 gripper phase。

## 如何理解

更合理的机制是：

1. torch.compile 改变了局部计算路径，单步 action 差异通常很小。
2. 机械臂闭环控制对 contact 和视觉状态非常敏感，小差异会逐步改变后续 observation。
3. 一旦 observation 进入不同局部区域，策略输出的 gripper intent 会在接近 `0` 和接近 `1` 的两个饱和模式之间跳变。
4. gripper 后处理是离散二值化，错一帧就会产生 `2.0` 的动作差。
5. 夹爪动作又正好影响接触、抓取、放置这些不可逆阶段，所以后续 EEF 轨迹会被放大到厘米级甚至十几厘米级。

这解释了为什么 same-observation drift 看起来可接受，但闭环 success 仍然回归。

## 速度和风险

速度收益仍然存在：

| run | server p50 |
|---|---:|
| 5-case FP16 baseline | 156.1 ms |
| 5-case compiled blocks 8-15 eager island | 127.9 ms |
| `task4:init9` compiled rerun | 76.9 ms |
| `task6:init8` compiled rerun | 75.8 ms |

但行为风险也很明确：

| run | success |
|---|---:|
| 5-case FP16 baseline | 3/5 |
| 5-case compiled blocks 8-15 eager island | 1/5 |
| `task4:init9` compiled rerun | fail |
| `task6:init8` compiled rerun | fail |

所以当前 `action_head_model_blocks_8_15_eager` 还不能作为 rollout 后端。它是有速度潜力的工程方向，但现在不是 behavior-transparent 的替代实现。

## 后续建议

优先做诊断性 ablation：

1. Gripper oracle：compiled 连续动作照常跑，但 gripper 维度临时使用 eager/baseline gripper。这个不用于部署，只用于确认失败是否主要由 gripper mode 错位触发。
2. Temporal guard：对 gripper 模式切换增加短窗口确认，观察是否能减少一帧错位。注意，普通 threshold hysteresis 可能不足，因为本轮看到的 raw gripper 离阈值很远。
3. Continuous-only compare：单独看 `x/y/z/rot` 小误差在没有 gripper 翻转时是否仍能把轨迹带偏。
4. 回到编译边界：优先尝试数值更保守的加速路径，例如只做不改 kernel 数值路径的 CUDA graph / replay，或者缩小 compile scope 到更可控的子模块。

目前阶段性判断：

```text
Phase 14 的核心收益不是“找到了可部署加速后端”，而是定位了为什么看似很小的推理差异会在仿真闭环中变成成功率回归。
下一步如果继续追 torch.compile，必须把 gripper mode 时序作为第一优先级验证指标。
```
