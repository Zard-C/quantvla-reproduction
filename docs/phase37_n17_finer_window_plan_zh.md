# Phase 37: N1.7 Finer Window Search

## 背景

Phase36B 说明 N1.7 上 speed-only compile 仍然会改变闭环轨迹：

- FP16: `12/15`, avg p50 `105.53ms`
- speed-only: `12/15`, avg p50 `72.33ms`, `1.46x`
- window `0--20`: `15/15`, avg p50 `97.69ms`, `1.08x`
- window `10--30`: `15/15`, avg p50 `98.89ms`, `1.07x`

这证明了 sensitivity-guided fallback 有效，但全局窗口太保守，速度收益偏小。

## Phase37A 目标

围绕 Phase36B 的两个成功窗口继续收缩：

```text
0--20
10--30
```

候选：

```text
window_0_15
window_5_15
window_5_20
window_8_18
window_10_20
window_10_25
window_12_22
```

评估集保持 Phase36B 的 15 cases：

```text
tasks 0/1/4/6/8 × init 21/22/23
```

判读：

1. 如果某个短窗口达到 `15/15`，并且 p50 明显低于 `97ms`，它就是新的全局候选。
2. 如果短窗口不能全局 `15/15`，但对某些 task 很强，则进入 task-conditioned routing。
3. 如果所有短窗口都失败，说明 N1.7 的稳健窗口至少需要覆盖更宽的 receding-horizon 段。

## Phase37B 候选方向

用 Phase36B 已经算出的 task-conditioned routing 做 held-out 验证：

```text
task 0 -> window_0_10
task 1 -> speed_only
task 4 -> speed_only
task 6 -> speed_only
task 8 -> window_10_30
```

这个 routing 在 Phase36B 的 15-case probe 上达到：

```text
15/15 success
avg p50 = 79.24ms
speedup = 1.33x vs FP16
```

但它可能过拟合当前 init `21/22/23`，需要在 held-out init 上验证。
