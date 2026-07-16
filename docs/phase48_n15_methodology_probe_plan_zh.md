# Phase48 计划：N1.5 方法论迁移 probe

## 目的

Phase47 已经在 N1.7 上离线验证了 CLSG-TS v2 的核心判断：

```text
global local winner 不能直接当 universal tactic；
task-conditioned selector 有信号，但需要 held-out confirmation；
oracle headroom 说明 routed policy 的解空间真实存在。
```

Phase48 切回另一个 checkpoint：

```text
GR00T N1.5 LIBERO long-horizon checkpoint
```

目标不是证明某个 N1.5 tactic 最优，而是检验同一套流程在另一权重上是否仍然产生类似的 speed-risk / repair-regression 结构。

## 设置

脚本：

```bash
bash toy_quantvla/run_phase48_n15_methodology_probe.sh
```

默认 cases：

```text
tasks 0/1/4/6/8 x init 21/22/23 = 15 cases
```

默认 tactics：

```text
fp16
speed_only
window_0_120
combo_blocks0_3_window_0_120
```

这些 tactic 与 N1.5 Phase30-34 使用的候选保持一致：

- `speed_only`：速度 anchor；
- `window_0_120`：speed-constrained duration fallback；
- `combo_blocks0_3_window_0_120`：behavior-first layer-duration protection。

## 要验证的问题

1. N1.5 新 slice 上，`speed_only` 是否仍然可能出现 paired regressions？
2. `window_0_120` 是否仍然是更均衡的 speed-risk candidate？
3. `combo_blocks0_3_window_0_120` 是否仍然更接近 behavior-first，但速度收益更小？
4. 与 N1.7 Phase47 相比，是否再次证明“迁移的是搜索/验证流程，不是固定窗口或固定 layer island”？

## 预期解释

如果 Phase48 复现出不同 tactic 在 success、repair/regression、latency 上的非单调排序，那么它会支持论文的跨权重主张：

> CLSG-TS transfers as a methodology. The selected tactic does not transfer as a universal recipe.

如果 Phase48 某个 tactic 在这个小 slice 上表现非常好，也仍然不能直接写成最终推荐；它只进入下一轮 held-out confirmation。
