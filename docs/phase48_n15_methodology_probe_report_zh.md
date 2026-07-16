# Phase48: N1.5 Methodology Transfer Probe

Phase48 是 N1.5 checkpoint 上的方法论迁移 probe。它使用一个新的 15-case slice，检查 CLSG-TS v2 的候选组织方式是否仍然产生可解释的 speed-risk / repair-regression 结构。

## 实验设计

- tag prefix: `phase48_n15_methodology_probe_15case_v1`
- case list: `0:21,0:22,0:23,1:21,1:22,1:23,4:21,4:22,4:23,6:21,6:22,6:23,8:21,8:22,8:23`
- deterministic policy seed base: `20260716`
- 候选: FP16 baseline, speed-only compile, duration 0-120, blocks0-3 + duration 0-120。

## 汇总

| run | policy | status | success | p50 ms | p90 ms | speedup | eager frac | reserved MiB | max reserved MiB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| FP16 baseline | no compile | complete | 7/15 | 149.28 | 176.22 | 1.00x | - | 4918.00 | 4918.00 |
| Speed-only compile | compile action_head.model | complete | 9/15 | 75.40 | 155.99 | 1.98x | - | 4960.00 | 5122.00 |
| Duration 0-120 | compile action_head.model; eager fallback for policy steps [0, 120) | complete | 9/15 | 79.38 | 167.60 | 1.88x | 0.20 | 4964.00 | 5126.00 |
| Blocks0-3 + duration 0-120 | compile action_head.model with blocks0-3 eager; eager fallback for policy steps [0, 120) | complete | 9/15 | 119.07 | 183.39 | 1.25x | 0.20 | 4944.00 | 5106.00 |

## Paired repair/regression

| comparison | common | repair | regress | net | repair cases | regress cases |
| --- | --- | --- | --- | --- | --- | --- |
| speed_only vs baseline | 15 | 2 | 0 | 2 | 1:23, 8:23 | - |
| window_0_120 vs baseline | 15 | 2 | 0 | 2 | 1:23, 4:22 | - |
| combo_blocks0_3_window_0_120 vs baseline | 15 | 2 | 0 | 2 | 1:23, 4:22 | - |
| window_0_120 vs speed_only | 15 | 1 | 1 | 0 | 4:22 | 8:23 |
| combo_blocks0_3_window_0_120 vs speed_only | 15 | 1 | 1 | 0 | 4:22 | 8:23 |

## Per-case outcomes

| case | baseline | speed_only | window_0_120 | combo_blocks0_3_window_0_120 |
| --- | --- | --- | --- | --- |
| 0:21 | S284 | S282 | S284 | S282 |
| 0:22 | S274 | S282 | S278 | S277 |
| 0:23 | S256 | S254 | S256 | S256 |
| 1:21 | S344 | S380 | S363 | S363 |
| 1:22 | S226 | S224 | S226 | S226 |
| 1:23 | F991 | S426 | S472 | S440 |
| 4:21 | S199 | S196 | S199 | S199 |
| 4:22 | F991 | F991 | S643 | S689 |
| 4:23 | S235 | S220 | S247 | S236 |
| 6:21 | F991 | F991 | F991 | F991 |
| 6:22 | F991 | F991 | F991 | F991 |
| 6:23 | F991 | F991 | F991 | F991 |
| 8:21 | F991 | F991 | F991 | F991 |
| 8:22 | F991 | F991 | F991 | F991 |
| 8:23 | F991 | S463 | F991 | F991 |

## 判读方式

- 如果 `speed_only` 继续在 held-out set 上胜出，它就是当前 checkpoint/task distribution 的工程 incumbent。
- 如果组合候选胜出，则说明 layer x duration sensitivity-guided refinement 能超过 naive compile。
- 如果所有候选都出现大量 regression，下一步应扩大 candidate pool，而不是固定追某个 window。

## 产物

- JSON: `toy_quantvla/results/phase48_n15_methodology_probe_15case_v1_summary.json`
- Report: `docs/phase48_n15_methodology_probe_report_zh.md`
