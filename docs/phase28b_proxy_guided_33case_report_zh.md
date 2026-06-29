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
| FP16 baseline | no compile | missing | missing | - | - | - | - | - |
| Speed-only compile | compile action_head.model | missing | missing | - | - | - | - | - |
| Proxy-guided blocks8-15 eager | compile action_head.model; keep blocks8-15 eager | missing | missing | - | - | - | - | - |
| Random/sanity block1 eager | compile action_head.model; keep block1 eager | missing | missing | - | - | - | - | - |

## Paired repair/regression

结果尚未完整，paired comparison 暂不可用。

## 初步判读规则

1. 如果 proxy variant 相比 speed-only 减少 regressions，同时保留大部分 speedup，说明 guide 有工程价值。
2. 如果 block0 proxy 改善 same-observation spike 但闭环不提升，说明局部数值 proxy 不足，论文需要强调 proxy 必须闭环校准。
3. 如果 blocks8-15 proxy 优于 random block1，说明 repair/regression/trajectory proxy 比随机保护更有用。
4. 如果所有 proxy 都不优于 speed-only，则 guide 仍可作为诊断框架，但暂不能声称能直接优化策略。

## 产物

- JSON: `toy_quantvla/results/phase28B_proxy_guided_33case_v1_summary.json`
- Report: `docs/phase28b_proxy_guided_33case_report_zh.md`
