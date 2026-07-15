# Phase43 计划：N1.7 hybrid CLSG-BO probe

## 目的

Phase40-42 的 BO 只搜索了 duration-window tactic：

```text
window_s_e
```

这证明 BO 可以帮助探索 speed-risk frontier，但搜索空间仍然太窄。Phase43 的目标是扩展到一个更有论文价值的混合空间：

```text
duration window x layer eager island
```

也就是同时验证：

- not all durations are equal；
- not all layers are equal；
- 两者组合是否能给出比单独 window 更好的 Pareto 点。

## Probe 设置

小规模 probe 使用 10 cases：

```text
tasks 0/1/4/6/8 x init 33/34
```

这些 init 没有用于 Phase40 warm-start、Phase41 BO batch 或 Phase42 held-out validation，适合作为下一轮 active-search probe。

默认脚本：

```bash
bash toy_quantvla/run_phase43_n17_hybrid_bo_probe.sh
```

## 候选 tactic

```text
fp16
speed_only
window_0_20
window_2_12
blocks0_3
blocks0_3_window_0_20
blocks0_3_window_2_12
blocks8_15_window_2_12
blocks16_31_window_2_12
```

含义：

| tactic | 意义 |
| --- | --- |
| `fp16` | behavior reference |
| `speed_only` | aggressive speed anchor |
| `window_0_20` | Phase42 behavior-first duration winner |
| `window_2_12` | Phase42 speed-constrained duration winner |
| `blocks0_3` | layer-only early-block eager island |
| `blocks0_3_window_0_20` | behavior-first window + early-block protection |
| `blocks0_3_window_2_12` | speed-constrained window + early-block protection |
| `blocks8_15_window_2_12` | mid-block contrast + speed-constrained window |
| `blocks16_31_window_2_12` | late-block contrast + speed-constrained window |

## 选择逻辑

Phase43 不是最终 held-out validation，而是 hybrid search 的第一轮扩展 probe。

我们希望回答三个问题：

1. `blocks0_3` 是否在 N1.7 上仍然是有用的 layer prior？
2. `blocks0_3_window_2_12` 是否能保持 `window_2_12` 的速度，同时进一步减少 paired regression？
3. late/mid block contrast 是否说明 layer island 的收益不是“随便保护一点都会好”？

## 成功判据

优先级从高到低：

1. paired regression 少于 `speed_only`；
2. success 不低于 FP16 或至少不低于 `window_2_12`；
3. p50 speedup 明显高于 `window_0_20`；
4. 如果 hybrid tactic 只是更慢但没有减少 regression，则不进入下一轮。

## 下一步

如果 Phase43 出现清晰 winner：

- behavior-first winner：进入 15-case 或 30-case held-out validation；
- speed-constrained winner：进入 multi-init held-out validation，并与 `window_2_12` 对照；
- 若没有 winner：说明当前 N1.7 的主要可控结构仍是 duration，不应继续扩大 layer BO。

如果 Phase43 发现 layer 维度有效，下一轮 BO 可以把 tactic 参数写成：

```text
(window_start, window_end, block_group)
```

其中 `block_group` 是 categorical variable，例如：

```text
none, blocks0_3, blocks0_7, blocks8_15, blocks16_31
```

这一步会把 CLSG-BO 从二维 window search 扩展成混合连续/离散 tactic search。
