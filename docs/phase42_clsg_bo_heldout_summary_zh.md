# Phase42 总结：CLSG-BO 从 warm-start 到 held-out validation

## 一句话结论

Phase42 证明了两件事：

1. `window_0_20` 不是 Phase40 那一批 init 上的偶然 winner。它在新的 held-out init `30/31/32` 上达到 `15/15`，相对 FP16 是 `2` repairs / `0` regressions。
2. BO 找到的 `window_2_12` 是一个更激进的 speed-risk 折中点：held-out 上 `14/15`、`1` repair / `0` regressions、`1.30x` p50 speedup。

这让 CLSG-BO 的作用更清晰：它不一定直接找到最稳 tactic，但能把搜索从人工猜窗口推进到有方向的 Pareto frontier 探索。

## Phase40/41/42 关系

| phase | 角色 | case slice | 主要作用 |
| --- | --- | --- | --- |
| Phase40 | warm-start/control batch | init `27/28/29` | 比较 `speed_only`、`0-20`、`5-15`、`8-18`、`10-20` |
| Phase41 | BO batch-1 | init `27/28/29` | 用 Phase40 结果选择新窗口，发现 `2-12` 这个 speed-risk 候选 |
| Phase42 | held-out validation | init `30/31/32` | 验证 `0-20`、`2-12` 和 BO 新提案是否跨 init 泛化 |

## Phase42 held-out 结果

Phase42 使用：

```text
tasks 0/1/4/6/8 x init 30/31/32 = 15 cases
```

候选：

```text
fp16
speed_only
window_0_20
window_2_12
window_4_9
window_6_11
```

| tactic | success | p50 ms | speedup vs FP16 | paired vs FP16 |
| --- | ---: | ---: | ---: | --- |
| `fp16` | `13/15` | `85.42` | `-` | baseline |
| `speed_only` | `13/15` | `64.25` | `1.33x` | `1` repair / `1` regression |
| `window_0_20` | `15/15` | `80.99` | `1.05x` | `2` repairs / `0` regressions |
| `window_2_12` | `14/15` | `65.76` | `1.30x` | `1` repair / `0` regressions |
| `window_4_9` | `12/15` | `62.36` | `1.37x` | `1` repair / `2` regressions |
| `window_6_11` | `13/15` | `63.26` | `1.35x` | `2` repairs / `2` regressions |

## 关键解释

### 1. `window_0_20` 是当前最强 behavior-first candidate

Phase40:

```text
window_0_20: 14/15, 1 repair / 0 regression
```

Phase42:

```text
window_0_20: 15/15, 2 repairs / 0 regressions
```

这说明 `0-20` 不只是某个 init slice 的偶然 basin 修复，而是在两个不同 held-out slice 上都保持了低 regression。它的代价是速度收益小：Phase42 只有 `1.05x` p50 speedup。

因此它适合作为论文里的 behavior-first deployment point：

```text
如果目标是尽量不打坏 FP16 成功样本，优先选 window_0_20。
```

### 2. `window_2_12` 是 BO 找到的 speed-constrained candidate

Phase41 在原 slice 上发现：

```text
window_2_12: 14/15, 2 repairs / 1 regression, 1.27x
```

Phase42 held-out 上变成：

```text
window_2_12: 14/15, 1 repair / 0 regressions, 1.30x
```

这非常有价值。它说明 BO 不是简单复述 `0-20`，而是在早期敏感区域里找到了一个更窄、更快、仍然低 regression 的候选。

因此它适合作为 speed-constrained deployment point：

```text
如果允许少量风险但仍要求不出现 paired regression，window_2_12 是当前最强折中点。
```

### 3. BO 新提案不是都泛化

`window_4_9` 和 `window_6_11` 都是 combined selector 在 Phase40+41 后推荐的窄窗口，但 Phase42 上表现不够稳：

```text
window_4_9: 12/15, 1 repair / 2 regressions
window_6_11: 13/15, 2 repairs / 2 regressions
```

这说明 surrogate/acquisition 只能提出候选，不能替代 held-out validation。BO 的价值不是“预测永远正确”，而是用有限实验预算更快逼近 Pareto frontier。

## 对论文的意义

Phase42 让算法主线更完整：

```text
closed-loop sensitivity analysis
-> warm-start tactic probes
-> BO/active candidate selection
-> held-out paired validation
-> behavior-first / speed-constrained deployment choice
```

这比单纯说“我们试了几个窗口”强很多。现在可以更正式地 claim：

> Closed-loop sensitivity-guided BO can discover acceleration tactics on the speed-risk frontier, but the final choice must still be made by held-out paired rollout validation.

## 当前推荐

### Behavior-first

```text
window_0_20
```

理由：

```text
Phase40: 14/15, 0 regression
Phase42: 15/15, 0 regression
```

缺点：

```text
speedup 小，Phase42 仅 1.05x
```

### Speed-constrained

```text
window_2_12
```

理由：

```text
Phase41: 14/15, 1.27x
Phase42: 14/15, 1.30x, 0 regression
```

缺点：

```text
不是满成功，且 Phase41 曾有 1 个 regression，需要继续跨 slice 观察。
```

## 下一步

1. 把 Phase40/41/42 整理进论文的 CLSG-BO 算法和结果章节。
2. 画一张 N1.7 speed-risk frontier 图，把 `speed_only`、`0-20`、`2-12`、`4-9`、`6-11` 放在一起。
3. 若还要继续实验，优先做 multi-fold selection，而不是继续无限细扫窗口。
