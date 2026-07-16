# Phase47: CLSG-TS v2 方法论离线验证

## 目的

Phase47 不重新跑仿真，而是把 Phase43-45 当作 tactic search / selection 数据，把 Phase46 当作独立 confirmation fold，检验几种选择策略是否能泛化。

核心问题不是哪个单点 tactic 最强，而是：

> 在 task-conditioned closed-loop sensitivity 下，global winner、task-conditioned selector 和 oracle upper bound 的差距有多大？

## 数据划分

| role | summaries |
| --- | --- |
| train/search | `toy_quantvla/results/phase43_n17_hybrid_bo_probe_10case_v1_summary.json`<br>`toy_quantvla/results/phase44_n17_hybrid_heldout_15case_v1_summary.json`<br>`toy_quantvla/results/phase45_n17_hybrid_alltask_stress_20case_v1_summary.json` |
| confirmation | `toy_quantvla/results/phase46_n17_alltask_confirmation_30case_v1_summary.json` |

候选 tactic 限制为 Phase46 已经实际跑过的集合，避免离线评估不存在的候选。

```text
fp16 speed_only window_0_20 window_2_12 blocks0_3_window_2_12
```

## Phase43-45 global ranking

| tactic | train success | repairs | regressions | avg p50 ms |
| --- | ---: | ---: | ---: | ---: |
| `blocks0_3_window_2_12` | 40/45 | 4 | 3 | 83.70 |
| `window_2_12` | 39/45 | 4 | 4 | 83.91 |
| `speed_only` | 36/45 | 2 | 5 | 80.62 |
| `window_0_20` | 35/45 | 3 | 7 | 88.23 |

按 behavior-first 规则，Phase43-45 会选择 `blocks0_3_window_2_12`。

## Phase46 confirmation 结果

| policy / tactic | success | p50 ms | speedup | repairs | regressions |
| --- | ---: | ---: | ---: | ---: | ---: |
| `fp16` | 27/30 | 92.62 | 1.00x | 0 | 0 |
| `speed_only` | 25/30 | 83.49 | 1.11x | 1 | 3 |
| `window_0_20` | 28/30 | 91.83 | 1.01x | 2 | 1 |
| `window_2_12` | 27/30 | 87.85 | 1.05x | 1 | 1 |
| `blocks0_3_window_2_12` | 24/30 | 87.26 | 1.06x | 1 | 4 |

## Selector 对比

| selector | success | p50 ms | speedup | repairs | regressions | comment |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `global_behavior_first_from_phase43_45` | 24/30 | 87.26 | 1.06x | 1 | 4 | train folds 的 global winner，Phase46 反转 |
| `task_conditioned_behavior_first_from_phase43_45` | 27/30 | 85.75 | 1.08x | 1 | 1 | 按 task 选择 tactic，有速度收益但仍需更稳 probe |
| `phase46_task_oracle_upper_bound` | 29/30 | 81.49 | 1.14x | 2 | 0 | 不可部署 upper bound，只显示 task routing 的潜在空间 |

## Task-conditioned selector 选择

| task | selected tactic | train success | train regressions | Phase46 outcome |
| ---: | --- | ---: | ---: | ---: |
| 0 | `speed_only` | 7/7 | 0 | 3/3 |
| 1 | `window_2_12` | 7/7 | 0 | 3/3 |
| 2 | `window_2_12` | 2/2 | 0 | 3/3 |
| 3 | `speed_only` | 2/2 | 0 | 3/3 |
| 4 | `speed_only` | 7/7 | 0 | 2/3 |
| 5 | `window_2_12` | 2/2 | 0 | 3/3 |
| 6 | `blocks0_3_window_2_12` | 6/7 | 1 | 3/3 |
| 7 | `speed_only` | 1/2 | 0 | 3/3 |
| 8 | `window_2_12` | 5/7 | 1 | 2/3 |
| 9 | `speed_only` | 2/2 | 0 | 2/3 |

## 结论

1. Phase43-45 的 global behavior-first selector 会选择 `blocks0_3_window_2_12`，但它在 Phase46 上只有 `24/30`，说明 global local winner 不能直接写成可迁移 tactic。
2. task-conditioned selector 在 Phase46 上达到 `27/30`、`1.08x`，和 `window_2_12` 的成功率相同但更快，并且明显优于 `speed_only` 的 regression profile。
3. Phase46 的 task oracle upper bound 是 `29/30`、`1.14x`，说明 task routing 的解空间确实存在，但当前 train probe 还不足以稳定找准每个 task 的 tactic。
4. 方法论应改写为：CLSG-TS/BO 是 candidate generator 和 task-aware validation protocol，不是 universal tactic oracle。
5. 下一步迁移到新模型时，不应复用 N1.7 的固定 window；应复用流程：warm-start probe -> task/slice-conditioned selector -> held-out confirmation。

