# Phase 37B: N1.7 Held-Out Routing Validation

## 目标

Phase37A 说明更窄的全局窗口不能稳定达到 `15/15`，但 `window_5_15`
是一个很强的速度优先候选。把 Phase36B 和 Phase37A 的候选合起来后，当前
15-case probe 上的最快全成功 routing 是：

```text
task 0 -> window_5_15
task 1 -> speed_only
task 4 -> speed_only
task 6 -> speed_only
task 8 -> window_5_15
```

它在 init `21/22/23` 上达到：

```text
15/15 success
avg p50 = 77.39ms
speedup = 1.36x vs FP16
```

Phase37B 的目标是验证这个 task-conditioned routing 是否只是过拟合当前
init slice。

## 设置

Held-out cases:

```text
tasks 0/1/4/6/8 × init 24/25/26
```

运行候选：

```text
fp16
speed_only
window_5_15
window_0_20
window_10_30
```

其中 `window_0_20` 和 `window_10_30` 是 Phase36B 上的全局稳健对照，
`window_5_15` 是 Phase37A 上的速度优先窗口。

## 判读

1. 先看 FP16 held-out 本身成功率，确认该 slice 的难度。
2. 比较 speed-only 是否继续有 repair/regression。
3. 比较全局 `window_0_20 / window_10_30` 是否迁移。
4. 按上述 routing 从 `speed_only` 与 `window_5_15` 中挑选对应 task 的结果，
   统计 held-out success、latency、paired repair/regression。

如果 routing 在 held-out 上仍接近全成功且明显快于全局稳健窗口，则说明
CLSG-TS 的输出应该升级为 tactic policy，而不是单一 tactic。
