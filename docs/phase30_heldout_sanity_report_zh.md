# Phase 30: Held-Out Sanity Set

Phase30 是 ICRA held-out sanity set。它不参与 Phase29 `0-120` proxy 的选择，用来检查该 proxy 是否只是在 33-case 上偶然有效。

## 实验设计

- tag prefix: `phase30_heldout_sanity_30case_v1`
- case list: `0:15,0:16,0:17,1:15,1:16,1:17,2:15,2:16,2:17,3:15,3:16,3:17,4:15,4:16,4:17,5:15,5:16,5:17,6:15,6:16,6:17,7:15,7:16,7:17,8:15,8:16,8:17,9:15,9:16,9:17`
- deterministic policy seed base: `20260701`
- 候选: FP16 baseline, speed-only compile, blocks0-3 eager, duration 0-120。

## 汇总

| run | policy | status | success | p50 ms | p90 ms | speedup | eager frac | reserved MiB | max reserved MiB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| FP16 baseline | no compile | complete | 23/30 | 88.81 | 94.18 | 1.00x | - | 5512.00 | 5512.00 |
| Speed-only compile | compile action_head.model | complete | 25/30 | 50.72 | 61.07 | 1.75x | - | 5554.00 | 5716.00 |
| Blocks0-3 eager | compile action_head.model; keep blocks0-3 eager | complete | 21/30 | 61.96 | 66.39 | 1.43x | - | 5536.00 | 5698.00 |
| Duration 0-120 | compile action_head.model; eager fallback for policy steps [0, 120) | complete | 22/30 | 51.82 | 87.97 | 1.71x | 0.26 | 5558.00 | 5720.00 |

## Paired repair/regression

| comparison | common | repair | regress | net | repair cases | regress cases |
| --- | --- | --- | --- | --- | --- | --- |
| speed_only vs baseline | 30 | 4 | 2 | 2 | 0:15, 1:16, 6:15, 8:16 | 6:17, 8:15 |
| blocks0_3 vs baseline | 30 | 3 | 5 | -2 | 0:15, 1:16, 8:16 | 0:17, 1:17, 2:15, 6:17, 8:15 |
| window_0_120 vs baseline | 30 | 1 | 2 | -1 | 0:15 | 4:15, 8:15 |
| blocks0_3 vs speed_only | 30 | 0 | 4 | -4 | - | 0:17, 1:17, 2:15, 6:15 |
| window_0_120 vs speed_only | 30 | 1 | 4 | -3 | 6:17 | 1:16, 4:15, 6:15, 8:16 |

## Per-case outcomes

| case | baseline | speed_only | blocks0_3 | window_0_120 |
| --- | --- | --- | --- | --- |
| 0:15 | F991 | S247 | S241 | S237 |
| 0:16 | S500 | S272 | S539 | S464 |
| 0:17 | S269 | S269 | F991 | S270 |
| 1:15 | F991 | F991 | F991 | F991 |
| 1:16 | F991 | S369 | S328 | F991 |
| 1:17 | S264 | S268 | F991 | S236 |
| 2:15 | S806 | S961 | F991 | S760 |
| 2:16 | S238 | S248 | S254 | S254 |
| 2:17 | S283 | S280 | S280 | S279 |
| 3:15 | S252 | S238 | S245 | S242 |
| 3:16 | S233 | S268 | S223 | S232 |
| 3:17 | S262 | S260 | S252 | S268 |
| 4:15 | S382 | S299 | S287 | F991 |
| 4:16 | S240 | S232 | S240 | S240 |
| 4:17 | S218 | S225 | S219 | S216 |
| 5:15 | S193 | S194 | S194 | S193 |
| 5:16 | S173 | S173 | S172 | S173 |
| 5:17 | S171 | S170 | S170 | S171 |
| 6:15 | F991 | S594 | F991 | F991 |
| 6:16 | F991 | F991 | F991 | F991 |
| 6:17 | S291 | F991 | F991 | S287 |
| 7:15 | S227 | S227 | S230 | S225 |
| 7:16 | S231 | S245 | S229 | S233 |
| 7:17 | S218 | S220 | S226 | S221 |
| 8:15 | S512 | F991 | F991 | F991 |
| 8:16 | F991 | S380 | S382 | F991 |
| 8:17 | F991 | F991 | F991 | F991 |
| 9:15 | S273 | S273 | S271 | S268 |
| 9:16 | S289 | S318 | S295 | S298 |
| 9:17 | S360 | S341 | S325 | S311 |

## 判读方式

- 如果 `window_0_120` 在 held-out set 上仍然不低于 speed-only，并且 latency 接近 speed-only，则它支持 ICRA 版的 sensitivity-guided acceleration claim。
- 如果 `window_0_120` 明显退化，则论文应把 `0-120` 降级为 33-case diagnostic，而不是泛化候选。
- 如果 `blocks0_3` 比 `window_0_120` 更稳，则下一步应进入 layer x duration 的组合保护。

## 产物

- JSON: `toy_quantvla/results/phase30_heldout_sanity_30case_v1_summary.json`
- Report: `docs/phase30_heldout_sanity_report_zh.md`
