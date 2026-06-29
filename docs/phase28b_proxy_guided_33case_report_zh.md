# Phase 28B: Proxy-Guided 33-Case Mixed Precision

本阶段验证论文里的 guide 是否能落到工程策略上：用 sensitivity proxy 决定哪些 action-head 边界保留高精度/eager，比较它是否比纯速度导向或随机保护更稳。

## 实验设计

- case list: `4:0,4:1,4:2,4:3,4:4,4:5,4:6,4:7,4:8,4:9,4:10,6:0,6:1,6:2,6:3,6:4,6:5,6:6,6:7,6:8,6:9,6:10,8:0,8:1,8:2,8:3,8:4,8:5,8:6,8:7,8:8,8:9,8:10`
- deterministic policy seed base: `20260613`
- speed-only: compile whole `action_head.model`
- proxy blocks8-15: 来自闭环 repair/regression proxy，保护 `transformer_blocks.8..15`
- random/sanity block1: 同样保护一个 block，但不来自 proxy

## 汇总

| run | policy | status | success | server p50 ms | speedup | reserved MiB | max reserved MiB | prepare s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| FP16 baseline | no compile | complete | 19/33 | 156.22 | 1.0x | 5512.0 | 5512.0 | 13.8 |
| Speed-only compile | compile action_head.model | complete | 16/33 | 70.20 | 2.23x | 5554.0 | 5554.0 | 19.53 |
| Proxy-guided blocks8-15 eager | compile action_head.model; keep blocks8-15 eager | complete | 15/33 | 82.64 | 1.89x | 5538.0 | 5538.0 | 18.98 |
| Random/sanity block1 eager | compile action_head.model; keep block1 eager | complete | 16/33 | 68.37 | 2.28x | 5536.0 | 5536.0 | 21.11 |

## Paired repair/regression

| comparison | common | repaired | regressed | net | repaired cases | regressed cases |
| --- | --- | --- | --- | --- | --- | --- |
| speed_only vs baseline | 33 | 3 | 6 | -3 | 6:7, 6:9, 8:9 | 4:6, 6:0, 6:6, 8:10, 8:3, 8:7 |
| proxy_blocks8_15 vs baseline | 33 | 1 | 5 | -4 | 8:4 | 4:9, 6:8, 8:10, 8:7, 8:8 |
| random_block1 vs baseline | 33 | 4 | 7 | -3 | 4:2, 4:7, 8:1, 8:9 | 4:9, 6:0, 6:10, 6:8, 8:10, 8:7, 8:8 |
| proxy_blocks8_15 vs speed_only | 33 | 5 | 6 | -1 | 4:6, 6:0, 6:6, 8:3, 8:4 | 4:9, 6:7, 6:8, 6:9, 8:8, 8:9 |
| random_block1 vs speed_only | 33 | 6 | 6 | 0 | 4:2, 4:6, 4:7, 6:6, 8:1, 8:3 | 4:9, 6:10, 6:7, 6:8, 6:9, 8:8 |

## Per-case outcomes

| case | baseline | speed_only | proxy_blocks8_15 | random_block1 |
| --- | --- | --- | --- | --- |
| 4:0 | S221 | S220 | S222 | S222 |
| 4:1 | F991 | F991 | F991 | F991 |
| 4:10 | S212 | S244 | S212 | S213 |
| 4:2 | F991 | F991 | F991 | S253 |
| 4:3 | S264 | S252 | S269 | S279 |
| 4:4 | S226 | S228 | S225 | S227 |
| 4:5 | S211 | S222 | S216 | S224 |
| 4:6 | S245 | F991 | S241 | S239 |
| 4:7 | F991 | F991 | F991 | S940 |
| 4:8 | F991 | F991 | F991 | F991 |
| 4:9 | S224 | S222 | F991 | F991 |
| 6:0 | S210 | F991 | S206 | F991 |
| 6:1 | S589 | S236 | S234 | S235 |
| 6:10 | S525 | S225 | S230 | F991 |
| 6:2 | S275 | S228 | S235 | S238 |
| 6:3 | F991 | F991 | F991 | F991 |
| 6:4 | F991 | F991 | F991 | F991 |
| 6:5 | S196 | S294 | S192 | S197 |
| 6:6 | S224 | F991 | S249 | S540 |
| 6:7 | F991 | S261 | F991 | F991 |
| 6:8 | S649 | S916 | F991 | F991 |
| 6:9 | F991 | S241 | F991 | F991 |
| 8:0 | S575 | S592 | S641 | S524 |
| 8:1 | F991 | F991 | F991 | S854 |
| 8:10 | S458 | F991 | F991 | F991 |
| 8:2 | F991 | F991 | F991 | F991 |
| 8:3 | S986 | F991 | S573 | S822 |
| 8:4 | F991 | F991 | S966 | F991 |
| 8:5 | F991 | F991 | F991 | F991 |
| 8:6 | F991 | F991 | F991 | F991 |
| 8:7 | S702 | F991 | F991 | F991 |
| 8:8 | S379 | S400 | F991 | F991 |
| 8:9 | F991 | S624 | F991 | S539 |

## 初步判读规则

1. 如果 proxy variant 相比 speed-only 减少 regressions，同时保留大部分 speedup，说明 guide 有工程价值。
2. 如果 block0 proxy 改善 same-observation spike 但闭环不提升，说明局部数值 proxy 不足，论文需要强调 proxy 必须闭环校准。
3. 如果 blocks8-15 proxy 优于 random block1，说明 repair/regression/trajectory proxy 比随机保护更有用。
4. 如果所有 proxy 都不优于 speed-only，则 guide 仍可作为诊断框架，但暂不能声称能直接优化策略。

## 产物

- JSON: `toy_quantvla/results/phase28B_proxy_guided_33case_v1_summary.json`
- Report: `docs/phase28b_proxy_guided_33case_report_zh.md`
