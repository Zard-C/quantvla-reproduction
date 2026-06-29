# Phase 28A: Sensitivity-Proxy Guided Mixed Precision

本阶段验证论文里的 guide 是否能落到工程策略上：用 sensitivity proxy 决定哪些 action-head 边界保留高精度/eager，比较它是否比纯速度导向或随机保护更稳。

## 实验设计

- case list: `4:6,4:7,4:8,4:9,4:10,6:0,6:1,6:2,6:3,6:4,8:6,8:7,8:8,8:9,8:10`
- deterministic policy seed base: `20260613`
- speed-only: compile whole `action_head.model`
- proxy block0: 来自 same-observation spike proxy，保护 `transformer_blocks.0`
- proxy blocks8-15: 来自闭环 repair/regression proxy，保护 `transformer_blocks.8..15`
- random/sanity block1: 同样保护一个 block，但不来自 proxy

## 汇总

| run | policy | status | success | server p50 ms | speedup | prepare s |
| --- | --- | --- | --- | --- | --- | --- |
| FP16 baseline | no compile | complete | 7/15 | 84.76 | 1.0x | 10.5 |
| Speed-only compile | compile action_head.model | complete | 5/15 | 50.35 | 1.68x | 17.5 |
| Proxy-guided block0 eager | compile action_head.model; keep block0 eager | complete | 6/15 | 50.96 | 1.66x | 19.91 |
| Proxy-guided blocks8-15 eager | compile action_head.model; keep blocks8-15 eager | complete | 7/15 | 67.36 | 1.26x | 17.26 |
| Random/sanity block1 eager | compile action_head.model; keep block1 eager | complete | 5/15 | 51.54 | 1.64x | 19.15 |

## Paired repair/regression

| comparison | common | repaired | regressed | net | repaired cases | regressed cases |
| --- | --- | --- | --- | --- | --- | --- |
| speed_only vs baseline | 15 | 1 | 3 | -2 | 8:9 | 4:6, 6:0, 8:10 |
| proxy_block0 vs baseline | 15 | 0 | 1 | -1 | - | 4:9 |
| proxy_blocks8_15 vs baseline | 15 | 1 | 1 | 0 | 8:9 | 4:9 |
| random_block1 vs baseline | 15 | 1 | 3 | -2 | 4:7 | 4:9, 6:0, 8:10 |
| proxy_block0 vs speed_only | 15 | 3 | 2 | 1 | 4:6, 6:0, 8:10 | 4:9, 8:9 |
| proxy_blocks8_15 vs speed_only | 15 | 3 | 1 | 2 | 4:6, 6:0, 8:10 | 4:9 |
| random_block1 vs speed_only | 15 | 2 | 2 | 0 | 4:6, 4:7 | 4:9, 8:9 |

## Per-case outcomes

| case | baseline | speed_only | proxy_block0 | proxy_blocks8_15 | random_block1 |
| --- | --- | --- | --- | --- | --- |
| 4:10 | S212 | S244 | S215 | S212 | S213 |
| 4:6 | S245 | F991 | S240 | S241 | S239 |
| 4:7 | F991 | F991 | F991 | F991 | S940 |
| 4:8 | F991 | F991 | F991 | F991 | F991 |
| 4:9 | S224 | S222 | F991 | F991 | F991 |
| 6:0 | S210 | F991 | S204 | S206 | F991 |
| 6:1 | S589 | S236 | S233 | S234 | S235 |
| 6:2 | S275 | S228 | S237 | S235 | S238 |
| 6:3 | F991 | F991 | F991 | F991 | F991 |
| 6:4 | F991 | F991 | F991 | F991 | F991 |
| 8:10 | S385 | F991 | S444 | S604 | F991 |
| 8:6 | F991 | F991 | F991 | F991 | F991 |
| 8:7 | F991 | F991 | F991 | F991 | F991 |
| 8:8 | F991 | F991 | F991 | F991 | F991 |
| 8:9 | F991 | S476 | F991 | S424 | F991 |

## 初步判读规则

1. 如果 proxy variant 相比 speed-only 减少 regressions，同时保留大部分 speedup，说明 guide 有工程价值。
2. 如果 block0 proxy 改善 same-observation spike 但闭环不提升，说明局部数值 proxy 不足，论文需要强调 proxy 必须闭环校准。
3. 如果 blocks8-15 proxy 优于 random block1，说明 repair/regression/trajectory proxy 比随机保护更有用。
4. 如果所有 proxy 都不优于 speed-only，则 guide 仍可作为诊断框架，但暂不能声称能直接优化策略。

## 产物

- JSON: `toy_quantvla/results/phase28A_proxy_guided_15case_v1_summary.json`
- Report: `docs/phase28_proxy_guided_mixed_precision_report_zh.md`
