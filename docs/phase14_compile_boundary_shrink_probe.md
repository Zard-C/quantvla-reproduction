# Phase 14 Compile Boundary Shrink Probe

本轮目标是继续收缩 `torch.compile` 的行为风险：在 `action_head_model_blocks_8_15_eager` 已经证明有速度但有闭环回归之后，尝试更保守的 eager island。

新增 target：

- `action_head_model_blocks_6_15_eager`
- `action_head_model_blocks_4_15_eager`

本轮实际跑了 `blocks_6_15_eager` 的 5-case flip smoke。由于结果已经明显不理想，没有继续跑更慢的 `blocks_4_15_eager`。

## 实验配置

```text
tag: phase14_block_island_6_15_5flip_v1
task suite: libero_10
case list: 4:9,6:8,8:4,8:7,8:8
baseline: FP16 eager
compiled: torch.compile(action_head.model) + transformer blocks 6..15 eager island
compile mode: reduce-overhead
policy seed base: 20260613
```

结果文件：

```text
toy_quantvla/results/phase14_block_island_6_15_5flip_v1_baseline_client_latency.json
toy_quantvla/results/phase14_block_island_6_15_5flip_v1_baseline_server_latency.json
toy_quantvla/results/phase14_block_island_6_15_5flip_v1_baseline_server_prepare.json
toy_quantvla/results/phase14_block_island_6_15_5flip_v1_compiled_client_latency.json
toy_quantvla/results/phase14_block_island_6_15_5flip_v1_compiled_server_latency.json
toy_quantvla/results/phase14_block_island_6_15_5flip_v1_compiled_server_prepare.json
```

## 总表

| policy | success | calls | client p50 | client p90 | server p50 | server p90 |
|---|---:|---:|---:|---:|---:|---:|
| FP16 baseline | 3/5 | 3370 | 161.1 ms | 167.9 ms | 156.1 ms | 162.8 ms |
| blocks 6-15 eager island | 1/5 | 4662 | 132.8 ms | 153.2 ms | 127.9 ms | 148.3 ms |

速度收益：

| metric | speedup |
|---|---:|
| server p50 | 1.22x |
| server p90 | 1.10x |
| client p50 | 1.21x |
| client p90 | 1.10x |

对比之前的 `blocks 8-15 eager island`：

| target | server p50 | 行为信号 |
|---|---:|---|
| blocks 8-15 eager island | about 76.5 ms on 30-case | 13/30 vs baseline 16/30 |
| blocks 6-15 eager island | 127.9 ms on 5 flip cases | 1/5 vs baseline 3/5 |

`blocks_6_15` 比 `blocks_8_15` 保护更多 eager blocks，但速度收益从约 2x p50 掉到约 1.2x p50。

## 逐 Case

| case | baseline | blocks 6-15 eager island | direction |
|---|---|---|---|
| 4:9 | success | fail | regression |
| 6:8 | success | success | same |
| 8:4 | success | fail | regression |
| 8:7 | fail | fail | same |
| 8:8 | fail | fail | same |

两个关键信号：

1. `4:9` 仍然从 baseline success 变成 compiled fail。这个 case 是之前 30-case 中最重要的 regression 之一，`blocks_6_15` 没有救回来。
2. `8:4` 在本轮 baseline 是 success，但 `blocks_6_15` 变成 fail。也就是说更保守的 eager island 还引入了新的失败。

## Prewarm 和显存

| policy | prewarm p50 | prewarm max | reserved memory |
|---|---:|---:|---:|
| FP16 baseline | 87.1 ms | 522.4 ms | 5512 MiB |
| blocks 6-15 eager island | 1056.8 ms | 3614.1 ms | 5538 MiB |

显存仍不是主要问题；问题是 cold compile/prewarm 成本和闭环成功率。

## 为什么不继续跑 Blocks 4-15

`blocks_4_15_eager` 会保护更多 DiT blocks，只留下更短的 compiled 前段和 compiled output path。直觉上它更慢，而且不一定能解决 `4:9`：

- `blocks_6_15` 已经把 server p50 拉回到 127.9 ms，只剩 1.22x。
- `blocks_6_15` 没有救回 `4:9`，还把 `8:4` 从 success 打成 fail。
- 继续到 `blocks_4_15` 很可能进一步损失速度，而行为仍可能不透明。

所以本轮停止在 `blocks_6_15` 是合理的。继续盲目扩大 eager island，不像是一条好工程路线。

## 当前判断

更保守的 block island 没有带来预期收益：

1. 保护更多后段 blocks 并不单调改善闭环行为。
2. 行为风险不只来自 blocks 8..15；剩余 compiled 前段、compiled output path、以及 graph break 后的数值/调度差异都可能改变轨迹。
3. 速度收益对 compile boundary 很敏感。保护 blocks 6..15 后，大图收益明显塌缩。

现在最有价值的结论反而是排除：

```text
不要继续沿 “保护越来越多后段 blocks” 盲目扩 rollout。
```

后续更合理的方向：

- 回到 `blocks_8_15_eager`，针对 flip case 做 first-divergence trace，而不是继续改边界。
- 或者做真正改变 kernel 成本的低比特/fused path，并用 fixed replay + small rollout 双门槛验证。
- 如果继续 torch.compile，应优先分析 `4:9` 的 action divergence 来定位具体哪个 action dimension/阶段触发失败。
