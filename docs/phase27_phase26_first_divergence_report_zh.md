# Phase 27: Phase26 First-Divergence Analysis

本阶段离线读取 Phase26 的 closed-loop traces，按 `policy_step` 对齐不同 layer/boundary variants，寻找最早的 action/state/gripper divergence。

读表说明：`s17 / 0.505mm / 0.00137 z` 表示 policy step 17，EEF pre-position 差为 0.505mm，该 step 的连续 action 最大差为 0.00137，最大差维度为 z。

注意：表里的 `steps L/R` 使用 trace 中已写入的 step 数。失败 rollout 在最后一次 server action 请求之后可能因 `executing action in terminated episode` 结束，因此会比 Phase26 latency JSON 的 request count 少 1。

## Baseline/Full 对各 Eager-Island Variant

| case | pair | success L/R | trace steps L/R | first action>0.01 | first pos>1mm | first pos>10mm | first gripper mismatch | max pos |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 4:9 | baseline -> full | True/True | 224/222 | 13 | s49 / 1.145mm / 0.036163 z | s59 / 10.228mm / 0.24147 z | s46 / 0.584mm / 0.03864 x | s70 / 27.685mm / 0.22522 z |
| 4:9 | baseline -> block0 | True/False | 224/990 | 12 | s44 / 1.005mm / 0.006409 y | s60 / 11.971mm / 0.130463 z | s46 / 1.005mm / 0.043631 x | s194 / 58.769mm / 0.400929 z |
| 4:9 | baseline -> block1 | True/False | 224/990 | 8 | s48 / 1.084mm / 0.049438 x | s59 / 10.024mm / 0.244675 z | s46 / 0.884mm / 0.04683 x | s146 / 80.322mm / 0.158844 z |
| 4:9 | full -> block0 | True/False | 222/990 | 21 | s59 / 1.186mm / 0.014191 z | s107 / 10.052mm / 0.079021 z | s99 / 4.216mm / 0.010986 z | s193 / 35.961mm / 0.308762 z |
| 4:9 | full -> block1 | True/False | 222/990 | 8 | s59 / 1.012mm / 0.032043 x | s104 / 11.746mm / 0.26825 z | s97 / 3.962mm / 0.084708 z | s145 / 55.717mm / 0.134583 z |
| 6:8 | baseline -> full | True/True | 649/916 | 23 | s57 / 1.416mm / 0.019913 x | s268 / 12.046mm / 0.384636 z | s262 / 3.755mm / 0.171204 z | s648 / 147.273mm / 0.863113 x |
| 6:8 | baseline -> block0 | True/True | 649/404 | 23 | s86 / 1.128mm / 0.013733 z | s268 / 11.857mm / 0.370903 z | s262 / 4.0mm / 0.155182 z | s403 / 206.329mm / 0.373993 x |
| 6:8 | baseline -> block1 | True/False | 649/990 | 54 | s85 / 1.155mm / 0.016823 x | s274 / 10.754mm / 0.185852 z | s261 / 3.817mm / 0.142707 z | s567 / 267.163mm / 0.036436 yaw |
| 6:8 | full -> block0 | True/True | 916/404 | 8 | s58 / 1.041mm / 0.020142 x | s286 / 11.144mm / 0.113525 z | s327 / 7.998mm / 0.036163 y | s403 / 206.55mm / 0.384979 x |
| 6:8 | full -> block1 | True/False | 916/990 | 53 | s58 / 1.133mm / 0.024948 y | s269 / 11.996mm / 0.343781 z | s261 / 0.978mm / 0.123252 z | s570 / 256.048mm / 0.249023 z |

## Direct Comparison: Full Compile vs Eager-Island Variants

| case | pair | success L/R | trace steps L/R | first action>0.01 | first pos>1mm | first gripper mismatch | max action |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 4:9 | full -> block0 | True/False | 222/990 | 21 | s59 / 1.186mm / 0.014191 z | s99 / 4.216mm / 0.010986 z | s208 / 30.381mm / 0.485763 x |
| 4:9 | full -> block1 | True/False | 222/990 | 8 | s59 / 1.012mm / 0.032043 x | s97 / 3.962mm / 0.084708 z | s152 / 46.794mm / 0.488977 y |
| 6:8 | full -> block0 | True/True | 916/404 | 8 | s58 / 1.041mm / 0.020142 x | s327 / 7.998mm / 0.036163 y | s380 / 73.235mm / 1.137764 x |
| 6:8 | full -> block1 | True/False | 916/990 | 53 | s58 / 1.133mm / 0.024948 y | s261 / 0.978mm / 0.123252 z | s438 / 161.917mm / 0.962563 x |

## 关键解读

1. 早期 action 分歧很早出现，但状态分叉滞后。`4:9` 中 baseline vs block0 在 step 10-12 就有 `y` 方向 action 差超过 0.005/0.01，但 EEF 位置到 step 44 才超过 1mm，到 step 60 才超过 10mm。
2. full compile 与 block0/block1 的差异也不是立刻造成失败。`4:9` 中 full vs block0 的位置差到 step 59 才超过 1mm，gripper mismatch 到 step 99 才出现；full vs block1 的 gripper mismatch 在 step 97。
3. gripper mismatch 不是充分条件。`4:9` baseline vs full 在 step 46 就有 gripper mismatch，但两者都成功；`6:8` baseline vs full 在 step 262 有 mismatch，也都成功。这说明 gripper phase 是重要分叉信号，但是否 flip 还取决于当时状态 margin。
4. `block0 eager` 的行为是 case-dependent：它让 `4:9` 失败，却让 `6:8` 从 full compile 的 916 步缩短到 404 步成功。这符合 Phase25 的结论：敏感性同时依赖 task、phase、action direction 和 margin。
5. action channel 上，早期 first-action divergence 多出现在 `y`，中后期和最大差常转向 `x/z`。这与 Phase25 的 action-channel anisotropy 能接上：不同阶段的敏感方向并不固定。

## 对工程路线的含义

这轮分析说明，不能把 layer 保护策略写成简单规则，比如“发现 block0 spike 就把 block0 留 eager”。局部 spike removal 可能修复某个 same-observation outlier，但闭环里它也可能改变早期 action 方向，经过几十步放大后在接触阶段跨过成功/失败边界。

更合适的路线是：先用 first-divergence 找出 outcome flip 前的早期 action/state signature，再把它和 Phase25 的 action-channel/phase sensitivity map 结合，形成 sensitivity-guided layer policy。

## 产物

- JSON: `toy_quantvla/results/phase27_phase26_first_divergence.json`
- Report: `docs/phase27_phase26_first_divergence_report_zh.md`
