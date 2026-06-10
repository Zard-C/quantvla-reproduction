# Phase 14 First-Divergence Trace: Blocks 8-15 Eager Island on Task 4 Init 9

本轮目标是解释 `action_head_model_blocks_8_15_eager` 为什么在 `task4:init9` 上从 FP16 baseline success 变成 compiled fail。

核心结论：

```text
同观测 action drift 很小，没有单步巨大数值爆炸。
但闭环运行时，小 action 差异很早被环境反馈放大；step 46 出现 gripper 二值化相反，是明确的离散分叉点。
```

## 实验文件

同观测 eager-controlled drift：

```text
toy_quantvla/results/phase14_first_divergence_blocks_8_15_4_9_v1.json
toy_quantvla/results/phase14_first_divergence_blocks_8_15_4_9_v1_trace_task04_init09.json
docs/phase14_first_divergence_blocks_8_15_4_9_v1.md
```

闭环 paired trace：

```text
toy_quantvla/results/phase14_first_divergence_baseline_task04_init09_closedloop_trace.json
toy_quantvla/results/phase14_first_divergence_blocks_8_15_task04_init09_closedloop_trace.json
toy_quantvla/results/phase14_first_divergence_blocks_8_15_4_9_analysis.json
```

## Same-Observation Drift

边界：

- eager FP16 控制环境；
- compiled path 只在同一个 live observation 上旁路计算 action；
- eager 和 compiled 使用同一个 denoising seed；
- 所以这里隔离的是同观测 action 差异，不含闭环反馈。

结果：

| metric | value |
|---|---:|
| eager success | true |
| policy steps | 224 |
| raw relative RMSE mean | 0.00280 |
| raw relative RMSE max | 0.00768 |
| raw max abs diff max | 0.00595 |
| LIBERO action RMSE mean | 0.000611 |
| LIBERO action max abs diff max | 0.00509 |

最早超过阈值的位置：

| threshold | first policy step | dim | diff |
|---|---:|---|---:|
| LIBERO action diff >= 0.002 | 19 | y | +0.00253 |
| LIBERO action diff >= 0.003 | 22 | y | -0.00343 |
| LIBERO action diff >= 0.004 | 147 | y | +0.00446 |
| LIBERO action diff >= 0.005 | 191 | z | +0.00509 |

这个结果说明：同观测下没有大尖峰。`blocks_8_15_eager` 的数值差异在 fixed live observation 上看起来很小。

## Closed-Loop Divergence

闭环对齐：

- baseline：FP16 eager，`task4:init9` success，224 policy steps。
- compiled：`action_head_model_blocks_8_15_eager`，same task/init/seed，fail，990 policy steps。
- 对齐前 224 个 common policy steps，比较 EEF pose、gripper 和送给 LIBERO 的 7 维 action。

总览：

| policy | success | steps | server p50 |
|---|---:|---:|---:|
| FP16 baseline | true | 224 | 156 ms |
| blocks 8-15 eager island | false | 990 | 76.9 ms |

关键 first-divergence 事件：

| event | policy step | detail |
|---|---:|---|
| first action diff >= 0.005 | 8 | y diff = -0.0211, while EEF pre-position diff is still below 0.1 mm |
| EEF pre-position diff >= 0.5 mm | 15 | z/y action still small, but state feedback has started |
| gripper binarization flips | 46 | baseline gripper = -1, compiled gripper = +1 |
| EEF pre-position diff >= 1 mm | 49 | z diff = +0.0371, x diff = +0.0284 |
| EEF pre-position diff >= 5 mm | 57 | z diff = +0.2307 |
| EEF pre-position diff >= 10 mm | 59 | x/z action diffs around +0.26 |
| EEF pre-position diff >= 50 mm | 115 | y diff = +0.3447 |
| max aligned EEF pre-position diff | 208 | about 172.6 mm |

The gripper event around step 46 is especially important:

| step | baseline raw gripper | compiled raw gripper | baseline LIBERO gripper | compiled LIBERO gripper |
|---:|---:|---:|---:|---:|
| 45 | +0.998496 | +0.998496 | -1 | -1 |
| 46 | +0.983917 | +0.002044 | -1 | +1 |
| 47 | +0.007123 | +0.006147 | +1 | +1 |

So the two policies disagree on the gripper command for exactly the kind of contact-sensitive moment where a one-step discrete mismatch can redirect the rest of the rollout.

## Interpretation

This is not a story of a single catastrophic numerical spike inside the compiled DiT.

The same-observation trace says the compiled action is very close to eager on the eager trajectory. But in closed loop, the policy/environment system has high local sensitivity:

1. Tiny early action differences create small state/image differences.
2. Those state differences change later actions more than the original same-observation drift.
3. Around step 46, the gripper postprocess crosses a binarization boundary in opposite directions.
4. After that, translation actions diverge hard, especially `z`, and the EEF trajectory separates by centimeters.

This explains why simply protecting more blocks did not help. The observed failure can be triggered by small differences before there is any large EEF state divergence, and the discrete gripper boundary amplifies it.

## Engineering Takeaway

For `blocks_8_15_eager`, speed is real: server p50 is about 76.9 ms on this closed-loop trace.

But the backend is not behavior-transparent. The risk concentrates in closed-loop feedback and action postprocessing, not just raw model-output drift.

Recommended next steps:

- Run the same first-divergence trace for one more regression case, especially `6:8` or `8:7`, to see whether gripper thresholding is a repeated pattern.
- Add a diagnostic that records raw gripper margin before binarization across baseline and compiled rollouts.
- If trying to rescue this backend, test a controlled gripper-stabilization ablation separately; do not mix it with compile changes in the same experiment.
