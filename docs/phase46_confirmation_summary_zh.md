# Phase46 总结：confirmation fold 反转了 hybrid winner

## 一句话结论

Phase46 是新的 all-task confirmation fold：

```text
tasks 0-9 x init 40/41/42 = 30 cases
```

它直接检验 Phase45 中表现较好的 hybrid tactic：

```text
blocks0_3_window_2_12
```

结果是一个很重要的负结果：`blocks0_3_window_2_12` 没有稳定泛化。它在 Phase46 中从 Phase45 的 `18/20` 退化到 `24/30`，相对 FP16 出现 `1` repair / `4` regressions，明显差于 `window_0_20` 和 `window_2_12`。

这说明：

> Hybrid CLSG-BO 能发现候选，但不能替代 held-out confirmation。Phase45 的 hybrid winner 是 slice-dependent，不应写成最终推荐 tactic。

更进一步说，Phase46 暴露的不是单纯的 BO 失败，而是一个更本质的闭环性质：

> duration/window sensitivity is task-conditioned. 一个 task slice 上找到的敏感阶段，不能默认迁移到另一个 task。

在 LIBERO 里，不同 task 的接近、抓取、对齐、放置和成功判定发生在不同 policy step 区间；同一个 `window_2_12` 或 `blocks0_3_window_2_12` 对某些任务是在保护接触关键段，对另一些任务可能是在保护无关段，甚至改变了原本可恢复的闭环 basin。因此 BO/active search 的输出应被理解为 task/slice-conditioned candidate，而不是跨 task 的 universal window。

## Phase46 结果

| tactic | success | p50 ms | speedup | paired vs FP16 |
| --- | ---: | ---: | ---: | --- |
| `fp16` | `27/30` | `92.62` | `-` | baseline |
| `speed_only` | `25/30` | `83.49` | `1.11x` | `1` repair / `3` regressions |
| `window_0_20` | `28/30` | `91.83` | `1.01x` | `2` repairs / `1` regression |
| `window_2_12` | `27/30` | `87.85` | `1.05x` | `1` repair / `1` regression |
| `blocks0_3_window_2_12` | `24/30` | `87.26` | `1.06x` | `1` repair / `4` regressions |

## 关键对比

### `window_0_20`

`window_0_20` 是 Phase46 的 behavior-first winner：

```text
28/30, 2 repairs / 1 regression
```

它几乎没有速度收益，p50 speedup 只有 `1.01x`，但行为最稳。

### `window_2_12`

`window_2_12` 是更好的 speed-risk compromise：

```text
27/30, 1 repair / 1 regression, 1.05x
```

它没有超过 FP16 aggregate success，但 regression 数低，而且比 `window_0_20` 快。

### `blocks0_3_window_2_12`

Phase45 中它看起来是最好的 hybrid point：

```text
18/20, 1.06x, 0 repair / 1 regression
```

但 Phase46 中退化为：

```text
24/30, 1.06x, 1 repair / 4 regressions
```

主要 regressions：

```text
task2:init40
task4:init40
task4:init41
task5:init41
```

这说明 early-block eager island 与 window fallback 的组合不是全局稳定保护，它可能在某些 init/task slice 上改变 basin，带来新的失败。

## 对论文主线的价值

Phase46 是非常有价值的 confirmation fold，因为它避免我们过度相信 Phase45 的 winner。

可以写成：

> A hybrid layer-duration candidate improved the Phase45 all-task stress slice, but failed to maintain its advantage on an independent 30-case confirmation fold. This confirms that CLSG-BO should be treated as a candidate-generation and validation protocol rather than an oracle that identifies universally safe tactics from a small probe.

这反而强化了论文贡献：

1. 不是所有 duration 相同。
2. 不是所有 layer 相同。
3. duration/window sensitivity 与 task 结构强相关，不能把某个 task slice 上的 BO winner 直接迁移为 universal tactic。
4. 但找到一个局部 Pareto candidate 还不够。
5. 真正可靠的是 closed-loop sensitivity-guided search + task-aware held-out confirmation。

## 当前推荐写法

不要把 `blocks0_3_window_2_12` 写成最终最优 tactic。

更稳的写法：

- `window_0_20`：behavior-first candidate，速度收益很小。
- `window_2_12`：当前最稳定的 speed-risk compromise。
- `blocks0_3_window_2_12`：hybrid search 找到的有趣候选，但 confirmation fold 暴露出 slice-dependent regressions。

最终 claim 应该落在：

> The reusable contribution is the closed-loop tactic search and validation protocol, not any single discovered fallback window or layer-duration combination.

如果继续发展方法，下一版 CLSG-TS 应该更明确地区分两类输出：

- global tactic：所有 task 共用，部署简单，但容易被 task-specific phase mismatch 限制；
- task-conditioned tactic/routing：不同 task 使用不同 tactic/window，更符合闭环敏感性的实际结构，但需要额外的任务识别、路由和验证成本。
