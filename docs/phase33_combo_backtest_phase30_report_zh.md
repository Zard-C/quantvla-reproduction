# Phase 33: Combo Backtest on Phase30 Slice

Phase33 将 Phase32 上最稳的 `blocks0-3 + window 0-120` 组合候选回测到 Phase30 的 init `15/16/17` slice。目标是检查该 combo 是否跨 held-out slice 保持 FP16 behavior。

## 实验设计

- tag prefix: `phase33_combo_backtest_phase30_30case_v1`
- reference summary: `toy_quantvla/results/phase30_heldout_sanity_30case_v1_summary.json`
- case list: `0:15,0:16,0:17,1:15,1:16,1:17,2:15,2:16,2:17,3:15,3:16,3:17,4:15,4:16,4:17,5:15,5:16,5:17,6:15,6:16,6:17,7:15,7:16,7:17,8:15,8:16,8:17,9:15,9:16,9:17`
- deterministic policy seed base: `20260701`
- 新跑候选: `blocks0-3 + window 0-120`。

## 汇总

| run | policy | status | success | p50 ms | p90 ms | speedup | eager frac |
| --- | --- | --- | --- | --- | --- | --- | --- |
| FP16 baseline | no compile | complete | 23/30 | 88.81 | 94.18 | 1.00x | - |
| Speed-only compile | compile action_head.model | complete | 25/30 | 50.72 | 61.07 | 1.75x | - |
| Blocks0-3 eager | compile action_head.model; keep blocks0-3 eager | complete | 21/30 | 61.96 | 66.39 | 1.43x | - |
| Duration 0-120 | compile action_head.model; eager fallback for policy steps [0, 120) | complete | 22/30 | 51.82 | 87.97 | 1.71x | 0.26 |
| Blocks0-3 + duration 0-120 | compile action_head.model with blocks0-3 eager; eager fallback for policy steps [0, 120) | complete | 22/30 | 83.31 | 158.08 | 1.07x | 0.26 |

## Combo Paired Repair/Regression

| comparison | common | repair | regress | net | repair cases | regress cases |
| --- | --- | --- | --- | --- | --- | --- |
| combo_blocks0_3_window_0_120 vs baseline | 30 | 0 | 1 | -1 | - | 8:15 |
| combo_blocks0_3_window_0_120 vs speed_only | 30 | 1 | 4 | -3 | 6:17 | 0:15, 1:16, 6:15, 8:16 |
| combo_blocks0_3_window_0_120 vs window_0_120 | 30 | 1 | 1 | 0 | 4:15 | 0:15 |
| combo_blocks0_3_window_0_120 vs blocks0_3 | 30 | 4 | 3 | 1 | 0:17, 1:17, 2:15, 6:17 | 0:15, 1:16, 8:16 |

## Per-case outcomes

| case | baseline | speed_only | blocks0_3 | window_0_120 | combo_blocks0_3_window_0_120 |
| --- | --- | --- | --- | --- | --- |
| 0:15 | F991 | S247 | S241 | S237 | F991 |
| 0:16 | S500 | S272 | S539 | S464 | S465 |
| 0:17 | S269 | S269 | F991 | S270 | S270 |
| 1:15 | F991 | F991 | F991 | F991 | F991 |
| 1:16 | F991 | S369 | S328 | F991 | F991 |
| 1:17 | S264 | S268 | F991 | S236 | S236 |
| 2:15 | S806 | S961 | F991 | S760 | S593 |
| 2:16 | S238 | S248 | S254 | S254 | S257 |
| 2:17 | S283 | S280 | S280 | S279 | S282 |
| 3:15 | S252 | S238 | S245 | S242 | S251 |
| 3:16 | S233 | S268 | S223 | S232 | S233 |
| 3:17 | S262 | S260 | S252 | S268 | S263 |
| 4:15 | S382 | S299 | S287 | F991 | S305 |
| 4:16 | S240 | S232 | S240 | S240 | S239 |
| 4:17 | S218 | S225 | S219 | S216 | S213 |
| 5:15 | S193 | S194 | S194 | S193 | S193 |
| 5:16 | S173 | S173 | S172 | S173 | S173 |
| 5:17 | S171 | S170 | S170 | S171 | S171 |
| 6:15 | F991 | S594 | F991 | F991 | F991 |
| 6:16 | F991 | F991 | F991 | F991 | F991 |
| 6:17 | S291 | F991 | F991 | S287 | S297 |
| 7:15 | S227 | S227 | S230 | S225 | S226 |
| 7:16 | S231 | S245 | S229 | S233 | S231 |
| 7:17 | S218 | S220 | S226 | S221 | S220 |
| 8:15 | S512 | F991 | F991 | F991 | F991 |
| 8:16 | F991 | S380 | S382 | F991 | F991 |
| 8:17 | F991 | F991 | F991 | F991 | F991 |
| 9:15 | S273 | S273 | S271 | S268 | S258 |
| 9:16 | S289 | S318 | S295 | S298 | S297 |
| 9:17 | S360 | S341 | S325 | S311 | S331 |

## 判读方式

- 如果 combo 在 Phase30 与 Phase32 都保持低 regression，它可以成为当前 behavior-preserving acceleration candidate。
- 如果 combo 在 Phase30 退化，则说明 combo 也依赖 held-out slice，应改用多 fold tactic search，而不是固定 tactic。

## 产物

- JSON: `toy_quantvla/results/phase33_combo_backtest_phase30_30case_v1_summary.json`
- Report: `docs/phase33_combo_backtest_phase30_report_zh.md`
