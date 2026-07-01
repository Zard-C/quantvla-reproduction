# Phase 34: Multi-Fold Robust Tactic Selection

Phase34 把 Phase30 slice 和 Phase32 slice 当作两个 validation folds，重新排序加速 tactic。目标不是找单个 held-out winner，而是在有限 rollout budget 下选择 worst-fold regression 更小的策略。

## Folds

| fold | source | baseline success | baseline p50 |
| --- | --- | --- | --- |
| phase30_slice_backtest | `toy_quantvla/results/phase33_combo_backtest_phase30_30case_v1_summary.json` | 23/30 | 88.81 |
| phase32_slice | `toy_quantvla/results/phase32_tactic_validation_30case_v1_summary.json` | 25/30 | 154.72 |

## Per-Fold Outcomes

| fold | tactic | success | speedup | p50 ms | repair | regress | net | regress cases |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| phase30_slice_backtest | speed_only | 25/30 | 1.75x | 50.72 | 4 | 2 | 2 | 6:17, 8:15 |
| phase30_slice_backtest | window_0_120 | 22/30 | 1.71x | 51.82 | 1 | 2 | -1 | 4:15, 8:15 |
| phase30_slice_backtest | combo_blocks0_3_window_0_120 | 22/30 | 1.07x | 83.31 | 0 | 1 | -1 | 8:15 |
| phase32_slice | speed_only | 20/30 | 2.27x | 68.01 | 0 | 5 | -5 | 4:18, 6:18, 6:19, 7:18, 8:20 |
| phase32_slice | window_0_120 | 25/30 | 1.98x | 78.26 | 1 | 1 | 0 | 6:19 |
| phase32_slice | combo_blocks0_3_window_0_120 | 25/30 | 1.75x | 88.47 | 0 | 0 | 0 | - |

## Behavior-First Ranking

行为优先分数把 paired regression 作为强惩罚，速度只作为 tie-breaker。

| rank | tactic | pooled success | mean success | worst success | mean speedup | worst speedup | total regress | worst regress | score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | combo_blocks0_3_window_0_120 | 47/60 | 0.783 | 0.733 | 1.41x | 1.07x | 1 | 1 | 86.82 |
| 2 | window_0_120 | 47/60 | 0.783 | 0.733 | 1.84x | 1.71x | 3 | 2 | 74.69 |
| 3 | speed_only | 45/60 | 0.750 | 0.667 | 2.01x | 1.75x | 7 | 5 | 39.35 |

## Speed-Constrained Ranking

只保留 worst-fold speedup >= `1.50x` 的 tactic。

| rank | tactic | pooled success | worst speedup | total regress | worst regress | score |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | window_0_120 | 47/60 | 1.71x | 3 | 2 | 74.69 |
| 2 | speed_only | 45/60 | 1.75x | 7 | 5 | 39.35 |

## Decision

- behavior-first selected: `combo_blocks0_3_window_0_120`
- speed-constrained selected: `window_0_120`
- interpretation: Use behavior-first ranking when matching FP16 outcomes is primary; use speed-constrained ranking when every fold must keep a minimum speedup.

## 结论

- `speed_only` 不能作为稳定策略：它在 Phase30 胜出，但 Phase32 出现 5 个 FP16 regression。
- `combo_blocks0_3_window_0_120` 的总 regression 最低，但 Phase30 上 worst-fold speedup 只有约 `1.07x`，工程收益偏弱。
- `window_0_120` 是当前更均衡的 speed-constrained candidate：两 fold 上都保持约 `>=1.7x` speedup，总 regression 比 speed-only 少。
- 因此下一步不应继续单押一个固定 tactic，而应把 selection objective 写成多 fold robust search：先限制 worst-fold regression，再按 worst-fold speedup 选 trade-off。
