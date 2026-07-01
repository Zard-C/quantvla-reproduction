# Phase 32: Held-Out Tactic Validation v2

Phase32 是第二个 held-out tactic validation set。它使用 init `18/19/20`，用于检查 Phase31 当前 incumbent `speed_only` 是否稳定，并测试一个 layer x duration 组合候选是否能超过 naive compile。

## 实验设计

- tag prefix: `phase32_tactic_validation_30case_v1`
- case list: ``
- deterministic policy seed base: `20260702`
- 候选: FP16 baseline, speed-only compile, duration 0-120, blocks0-3 + duration 0-120。

## 汇总

| run | policy | status | success | p50 ms | p90 ms | speedup | eager frac | reserved MiB | max reserved MiB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| FP16 baseline | no compile | complete | 25/30 | 154.72 | 163.94 | 1.00x | - | 5512.00 | 5512.00 |
| Speed-only compile | compile action_head.model | complete | 20/30 | 68.01 | 144.72 | 2.27x | - | 5554.00 | 5716.00 |
| Duration 0-120 | compile action_head.model; eager fallback for policy steps [0, 120) | complete | 25/30 | 78.26 | 159.40 | 1.98x | 0.27 | 5558.00 | 5720.00 |
| Blocks0-3 + duration 0-120 | compile action_head.model with blocks0-3 eager; eager fallback for policy steps [0, 120) | complete | 25/30 | 88.47 | 159.19 | 1.75x | 0.29 | 5538.00 | 5700.00 |

## Paired repair/regression

| comparison | common | repair | regress | net | repair cases | regress cases |
| --- | --- | --- | --- | --- | --- | --- |
| speed_only vs baseline | 30 | 0 | 5 | -5 | - | 4:18, 6:18, 6:19, 7:18, 8:20 |
| window_0_120 vs baseline | 30 | 1 | 1 | 0 | 6:20 | 6:19 |
| combo_blocks0_3_window_0_120 vs baseline | 30 | 0 | 0 | 0 | - | - |
| window_0_120 vs speed_only | 30 | 5 | 0 | 5 | 4:18, 6:18, 6:20, 7:18, 8:20 | - |
| combo_blocks0_3_window_0_120 vs speed_only | 30 | 5 | 0 | 5 | 4:18, 6:18, 6:19, 7:18, 8:20 | - |

## Per-case outcomes

| case | baseline | speed_only | window_0_120 | combo_blocks0_3_window_0_120 |
| --- | --- | --- | --- | --- |
| 0:18 | F991 | F991 | F991 | F991 |
| 0:19 | S289 | S287 | S285 | S285 |
| 0:20 | S268 | S263 | S264 | S264 |
| 1:18 | S232 | S228 | S234 | S231 |
| 1:19 | S889 | S229 | S927 | S922 |
| 1:20 | S229 | S232 | S229 | S230 |
| 2:18 | S270 | S241 | S263 | S264 |
| 2:19 | S308 | S306 | S315 | S308 |
| 2:20 | F991 | F991 | F991 | F991 |
| 3:18 | S220 | S220 | S220 | S220 |
| 3:19 | S249 | S245 | S249 | S250 |
| 3:20 | S385 | S270 | S329 | S327 |
| 4:18 | S266 | F991 | S255 | S269 |
| 4:19 | S216 | S215 | S216 | S216 |
| 4:20 | F991 | F991 | F991 | F991 |
| 5:18 | S236 | S236 | S234 | S235 |
| 5:19 | S334 | S247 | S361 | S344 |
| 5:20 | S197 | S199 | S197 | S197 |
| 6:18 | S216 | F991 | S216 | S216 |
| 6:19 | S228 | F991 | F991 | S234 |
| 6:20 | F991 | F991 | S356 | F991 |
| 7:18 | S225 | F991 | S213 | S210 |
| 7:19 | F991 | F991 | F991 | F991 |
| 7:20 | S258 | S243 | S268 | S244 |
| 8:18 | S516 | S483 | S761 | S491 |
| 8:19 | S383 | S691 | S483 | S375 |
| 8:20 | S449 | F991 | S447 | S393 |
| 9:18 | S334 | S349 | S317 | S343 |
| 9:19 | S255 | S253 | S254 | S254 |
| 9:20 | S281 | S283 | S281 | S281 |

## 实验结论

- `speed_only` 是最快候选，但在这个 held-out slice 上出现明显闭环退化：相对 FP16 是 0 repair / 5 regress。
- `window_0_120` 恢复到 FP16 aggregate success，并保持接近 `2x` 的 p50 speedup，但仍有 1 个 repair / 1 个 regression。
- `blocks0-3 + window_0_120` 在这个 slice 上逐 case 复现 FP16 outcome，0 repair / 0 regression，同时保留约 `1.75x` p50 speedup。
- Phase30 和 Phase32 的 winner 不同，说明 held-out slice selection 会影响 tactic ranking；最终 tactic 不能只凭单个 held-out set 决定。


## 判读方式

- `speed_only` 不应再被称为稳定 incumbent；它在 Phase30 胜出，但在 Phase32 出现 5 个 FP16 regression。
- `combo_blocks0_3_window_0_120` 是 Phase32 上最强的 behavior-preserving candidate，因为它逐 case 保持 FP16 outcome。
- 下一步应回测 combo 在 Phase30 init `15/16/17` 上的表现，或者引入第三个 held-out slice 做 cross-validation 式 tactic selection。

## 产物

- JSON: `toy_quantvla/results/phase32_tactic_validation_30case_v1_summary.json`
- Report: `docs/phase32_tactic_validation_report_zh.md`
