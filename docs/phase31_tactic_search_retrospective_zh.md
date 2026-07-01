# Phase 31: Retrospective Tactic Search

Phase31 把前面阶段的经验整理成一个轻量的 tactic-search prototype。这里不重新跑 GPU，而是复用已有 closed-loop summary，检查一个打分流程会如何选择候选，以及 probe 上的选择能否通过 held-out validation。

## Scoring

- rule: `100*success_rate + 4*speedup + repair_vs_baseline - 2.5*regress_vs_baseline - 10*baseline_success_gap - 4*worst_task_gap, with a light speed-only regression penalty when available`
- priority: closed-loop success first, speed second, regressions and worst-slice risk as penalties

解释上要注意：这个分数不是理论最优目标，只是一个启发式排序器。它的作用是把 rollout budget 集中到少数候选上，然后必须用 held-out closed-loop validation 验收。

## phase28A_proxy_guided_mixed_precision (phase28A_proxy_guided_15case_v1)

- source: `toy_quantvla/results/phase28A_proxy_guided_15case_v1_summary.json`
- baseline: 7/15 at p50 84.76 ms

| rank | candidate | family | success | p50 ms | speedup | worst task | net vs base | regress | score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | proxy_blocks8_15 | layer | 7/15 | 67.36 | 1.26x | 0.40 | 0 | 1 | 48.31 |
| 2 | proxy_block0 | layer | 6/15 | 50.96 | 1.66x | 0.20 | -1 | 1 | 39.77 |
| 3 | speed_only | compile | 5/15 | 50.35 | 1.68x | 0.20 | -2 | 3 | 29.02 |
| 4 | random_block1 | layer | 5/15 | 51.54 | 1.64x | 0.00 | -2 | 3 | 27.06 |

## phase28A_proxy_guided_mixed_precision (phase28B_proxy_guided_33case_v1)

- source: `toy_quantvla/results/phase28B_proxy_guided_33case_v1_summary.json`
- baseline: 19/33 at p50 156.22 ms

| rank | candidate | family | success | p50 ms | speedup | worst task | net vs base | regress | score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | speed_only | compile | 16/33 | 70.20 | 2.23x | 0.27 | -3 | 6 | 41.59 |
| 2 | random_block1 | layer | 16/33 | 68.37 | 2.28x | 0.36 | -3 | 7 | 37.65 |
| 3 | proxy_blocks8_15 | layer | 15/33 | 82.64 | 1.89x | 0.27 | -4 | 5 | 33.89 |

## phase28D_duration_window_sweep (phase28D_duration_window_33case_v1)

- source: `toy_quantvla/results/phase28D_duration_window_33case_v1_summary.json`
- baseline: 19/33 at p50 156.22 ms

| rank | candidate | family | success | p50 ms | speedup | worst task | net vs base | regress | score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | window_0_250 | duration | 18/33 | 88.75 | 1.76x | 0.36 | -1 | 2 | 54.74 |
| 2 | window_180_420 | duration | 16/33 | 78.82 | 1.98x | 0.36 | -3 | 5 | 41.95 |
| 3 | speed_only | compile | 16/33 | 70.20 | 2.23x | 0.27 | -3 | 6 | 41.59 |
| 4 | window_120_320 | duration | 16/33 | 78.09 | 2.00x | 0.36 | -3 | 5 | 41.53 |
| 5 | window_80_220 | duration | 14/33 | 75.11 | 2.08x | 0.27 | -5 | 5 | 30.32 |

## phase29_finer_duration_proxy (phase29_finer_duration_proxy_33case_v1)

- source: `toy_quantvla/results/phase29_finer_duration_proxy_33case_v1_summary.json`
- baseline: 19/33 at p50 156.22 ms

| rank | candidate | family | success | p50 ms | speedup | worst task | net vs base | regress | score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | window_0_120 | duration | 19/33 | 69.66 | 2.24x | 0.45 | 0 | 2 | 60.85 |
| 2 | phase28d_window_0_250 | duration | 18/33 | 88.75 | 1.76x | 0.36 | -1 | 2 | 54.74 |
| 3 | window_0_180 | duration | 18/33 | 77.55 | 2.01x | 0.45 | -1 | 3 | 54.10 |
| 4 | window_240_320 | duration | 17/33 | 66.71 | 2.34x | 0.36 | -2 | 4 | 49.72 |
| 5 | window_0_220 | duration | 16/33 | 90.49 | 1.73x | 0.27 | -3 | 3 | 42.09 |
| 6 | window_120_280 | duration | 16/33 | 73.48 | 2.13x | 0.36 | -3 | 5 | 42.05 |
| 7 | speed_only | compile | 16/33 | 70.20 | 2.23x | 0.27 | -3 | 6 | 41.59 |
| 8 | window_160_240 | duration | 15/33 | 69.29 | 2.25x | 0.27 | -4 | 5 | 36.83 |

## phase30_heldout_sanity (phase30_heldout_sanity_30case_v1)

- source: `toy_quantvla/results/phase30_heldout_sanity_30case_v1_summary.json`
- baseline: 23/30 at p50 88.81 ms

| rank | candidate | family | success | p50 ms | speedup | worst task | net vs base | regress | score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | speed_only | compile | 25/30 | 50.72 | 1.75x | 0.33 | 2 | 2 | 86.67 |
| 2 | window_0_120 | duration | 22/30 | 51.82 | 1.71x | 0.00 | -1 | 2 | 68.34 |
| 3 | blocks0_3 | layer | 21/30 | 61.96 | 1.43x | 0.00 | -2 | 5 | 57.55 |

## phase32_tactic_validation (phase32_tactic_validation_30case_v1)

- source: `toy_quantvla/results/phase32_tactic_validation_30case_v1_summary.json`
- baseline: 25/30 at p50 154.72 ms

| rank | candidate | family | success | p50 ms | speedup | worst task | net vs base | regress | score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | combo_blocks0_3_window_0_120 | duration | 25/30 | 88.47 | 1.75x | 0.67 | 0 | 0 | 91.50 |
| 2 | window_0_120 | duration | 25/30 | 78.26 | 1.98x | 0.67 | 0 | 1 | 90.92 |
| 3 | speed_only | compile | 20/30 | 68.01 | 2.27x | 0.00 | -5 | 5 | 57.58 |

## Probe To Held-Out Transfer

| candidate | probe rank | probe success | probe speedup | held-out rank | held-out success | held-out speedup | held-out net |
| --- | --- | --- | --- | --- | --- | --- | --- |
| window_0_120 | 1 | 19/33 | 2.24x | 2 | 25/30 | 1.98x | 0 |
| speed_only | 7 | 16/33 | 2.23x | 3 | 20/30 | 2.27x | -5 |

## Held-Out Selection Sensitivity

| held-out slice | selected | success | speedup | net vs base | regress | score |
| --- | --- | --- | --- | --- | --- | --- |
| phase30_heldout_sanity (phase30_heldout_sanity_30case_v1) | speed_only | 25/30 | 1.75x | 2 | 2 | 86.67 |
| phase32_tactic_validation (phase32_tactic_validation_30case_v1) | combo_blocks0_3_window_0_120 | 25/30 | 1.75x | 0 | 0 | 91.50 |

## Decision

- probe selected: `window_0_120`
- held-out selected: `combo_blocks0_3_window_0_120`
- current incumbent: `combo_blocks0_3_window_0_120`
- interpretation: Probe search and held-out validation select different tactics. Treat fixed tactics as candidates, not conclusions; ranking must be checked across multiple held-out slices.

## 结论

- Phase29 probe 会选择 `window_0_120`，Phase30 held-out 会选择 `speed_only`，Phase32 held-out 会选择 `combo_blocks0_3_window_0_120`。
- 这说明 held-out slice 本身是 tactic-search 设计的一部分；只用一组 held-out 很容易把某个 trajectory basin 上的偶然优势误认为通用优势。
- `speed_only` 速度最快，但 Phase32 出现 `5` 个 FP16 regression，不能再作为稳定 incumbent。
- `combo_blocks0_3_window_0_120` 在 Phase32 上做到 `0` repair / `0` regression，并有 `1.75x` p50 speedup；它是当前最强的 behavior-preserving candidate，但还需要回测 Phase30 或第三组 held-out 才能写成最终推荐 tactic。
