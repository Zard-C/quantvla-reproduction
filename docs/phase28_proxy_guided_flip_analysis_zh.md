# Phase 28A Flip Analysis: Proxy-Guided Mixed Precision

这份报告只分析已完成的 15 个 matched task-init pair。它的目标不是证明最终部署配置已经收敛，而是回答一个更具体的问题：论文里的 sensitivity-guided 设计思想，是否能把 speed-only 编译造成的闭环退化拉回来。

## 速度-成功率前沿

| run | success | delta vs baseline | server p50 ms | speedup | repairs vs baseline | regressions vs baseline | net |
| --- | --- | --- | --- | --- | --- | --- | --- |
| FP16 baseline | 7/15 | 0 | 84.76 | 1.0x | - | - | - |
| Speed-only compile | 5/15 | -2 | 50.35 | 1.68x | 1 | 3 | -2 |
| Proxy-guided block0 eager | 6/15 | -1 | 50.96 | 1.66x | 0 | 1 | -1 |
| Proxy-guided blocks8-15 eager | 7/15 | 0 | 67.36 | 1.26x | 1 | 1 | 0 |
| Random/sanity block1 eager | 5/15 | -2 | 51.54 | 1.64x | 1 | 3 | -2 |

这里最重要的对照是 `speed_only` 和 `proxy_blocks8_15`。`speed_only` 把 server p50 从 `84.76 ms` 降到 `50.35 ms`，但成功数从 `7/15` 掉到 `5/15`。`proxy_blocks8_15` 牺牲一部分速度，把 p50 放到 `67.36 ms`，成功数恢复到 `7/15`。

## 相对 speed-only 的修复

| run | success | server p50 ms | speedup | repaired | regressed | net | repaired cases | regressed cases |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Proxy-guided block0 eager | 6/15 | 50.96 | 1.66x | 3 | 2 | 1 | 4:6, 6:0, 8:10 | 4:9, 8:9 |
| Proxy-guided blocks8-15 eager | 7/15 | 67.36 | 1.26x | 3 | 1 | 2 | 4:6, 6:0, 8:10 | 4:9 |
| Random/sanity block1 eager | 5/15 | 51.54 | 1.64x | 2 | 2 | 0 | 4:6, 4:7 | 4:9, 8:9 |

`proxy_blocks8_15` 是这轮最像“按 guide 起效”的配置：它相对 `speed_only` 修复了 `4:6, 6:0, 8:10` 三个失败，只新增 `4:9` 一个失败，净 `+2`。`random_block1` 速度相近，但净收益为 `0`；这说明并不是随便保留一个 eager block 就能稳定改善闭环表现。

`proxy_block0` 也有价值，但证据更复杂。它几乎保留了 speed-only 的速度，并修复 `4:6, 6:0, 8:10`，但丢掉了 speed-only 在 `8:9` 上的 beneficial branch，同时也打坏 `4:9`。这支持一个更细的结论：same-observation spike proxy 能找到局部数值敏感边界，但不一定等价于闭环 outcome proxy。

## 关键病例

| case | baseline | speed_only | block0 | blocks8_15 | random | 解读 |
| --- | --- | --- | --- | --- | --- | --- |
| 4:6 | S245 | F991 | S240 | S241 | S239 | speed-only 打坏 FP16 成功；三个保护策略都救回，说明这里对 full compile 边界敏感。 |
| 6:0 | S210 | F991 | S204 | S206 | F991 | speed-only/regression；block0 和 blocks8-15 救回，random 没救回，更支持 proxy 有信息量。 |
| 8:10 | S385 | F991 | S444 | S604 | F991 | speed-only/regression；两个 proxy 救回，random 没救回，是 proxy-guided 的强证据之一。 |
| 8:9 | F991 | S476 | F991 | S424 | F991 | baseline 失败、speed-only 成功；blocks8-15 保留了这个 beneficial branch，block0/random 没保住。 |
| 4:9 | S224 | S222 | F991 | F991 | F991 | baseline 和 speed-only 都成功，但三个 eager-protection 版本都失败，说明保护本身也是策略扰动。 |
| 4:7 | F991 | F991 | F991 | F991 | S940 | 只有 random 成功且接近 horizon，像低 margin / 偶然分支，不宜作为主证据。 |

## 对论文 claim 的含义

1. 这不是单纯的数值加速问题。`speed_only` 和 `proxy_blocks8_15` 都是无损语义的编译边界选择，但闭环成功率不同，说明实现边界也会成为 policy perturbation。
2. 这支持 `not all layers are equal`：保护 `blocks8-15` 的收益明显高于随机 `block1`，并且能保住 `8:9` 这个 speed-only repair。
3. 这也支持 `not all proxies are equal`：局部数值 spike proxy 和闭环 repair/regression proxy 给出的选择不同，后者更接近我们真正关心的 rollout outcome。
4. 当前样本只有 15 个 matched cases，结论应该写成 evidence / diagnostic result，而不是最终显著性结论。

## 下一步

1. 把 `4:6, 6:0, 8:10, 8:9, 4:9` 做 first-divergence / trajectory branching，对比 speed-only 和 proxy_blocks8_15。
2. 把样本扩到 30 或 50 个 matched cases，优先覆盖 task 4、6、8 的 init 段。
3. 分离 cold compile、warm serving latency 和显存峰值，避免把部署准备时间混进在线推理速度。

## 输入产物

- Summary JSON: `toy_quantvla/results/phase28A_proxy_guided_15case_v1_summary.json`
- Phase 28A report: `docs/phase28_proxy_guided_mixed_precision_report_zh.md`
