# Phase 13 Eager Island Probe

本轮目标：

```text
保留 whole action_head.model compile 的大图速度收益，
同时让 transformer_blocks.8..15.ff 走 eager path，
观察能否压掉 task4:init6 的后半 FFN drift spike。
```

新增 target：

```text
action_head_model_ff_8_15_eager
action_head_model_blocks_8_15_eager
```

实现方式：

```text
compile policy.model.action_head.model
mark policy.model.action_head.model.transformer_blocks.8..15.ff.forward with torch._dynamo.disable
or mark policy.model.action_head.model.transformer_blocks.8..15.forward with torch._dynamo.disable
```

代码：

```text
toy_quantvla/phase13_compile_targets.py
toy_quantvla/timed_fp16_inference_service.py
toy_quantvla/phase13_torch_compile_online_drift.py
```

## Online Drift

输出：

```text
docs/phase13_eager_island_online_drift_4_6_6_0_v1.md
toy_quantvla/results/phase13_eager_island_online_drift_4_6_6_0_v1.json
```

| target | case | eager success | steps | raw rel RMSE mean | raw rel RMSE max | raw max abs diff | LIBERO RMSE mean |
|---|---|---|---:|---:|---:|---:|---:|
| whole action_head.model | 4:6 | success | 245 | 0.003573 | 0.045184 | 0.142975 | 0.000697 |
| whole action_head.model | 6:0 | success | 210 | 0.003889 | 0.011721 | 0.011353 | 0.000645 |
| action_head + FFN 8-15 eager island | 4:6 | success | 245 | 0.002353 | 0.038986 | 0.129303 | 0.000454 |
| action_head + FFN 8-15 eager island | 6:0 | success | 210 | 0.002482 | 0.007734 | 0.005493 | 0.000451 |
| action_head + blocks 8-15 eager island | 4:6 | success | 245 | 0.003185 | 0.041761 | 0.134369 | 0.000593 |

解读：

- 平均 drift 有下降。
- `6:0` 的 drift 也下降明显。
- 但 `4:6` 的大尖峰没有被压掉：raw max abs diff 仍是 `0.129303`，接近 whole action head compile 的 `0.142975`。

这说明：

```text
简单 torch._dynamo.disable 后半 FFN forward，并不能恢复到“FFN 8-15 单独 eager/compile 对比”里的低风险路径。
```

可能原因：

- 整块 DiT compile 的上下游图仍改变了进入后半 FFN 的 hidden state；
- graph break 只保护 FFN forward 本身，不保护 FFN 前后的 residual/add/norm 数据路径；
- Dynamo/Inductor 仍可能在 graph break 两侧生成不同的 tensor layout、精度路径或调度边界。

整 block 8-15 eager island 的 online drift 也没有显著压低 `4:6` 的 raw max spike，仍有 `0.134369`。这说明同观测 raw max spike 不是唯一决定闭环成败的指标，还要看动作误差的方向、持续时间以及它落在接触过程的哪个阶段。

## Closed-Loop

输出：

```text
toy_quantvla/results/phase13_eager_island_2case_v1_compiled_client_latency.json
toy_quantvla/results/phase13_eager_island_2case_v1_compiled_server_latency.json
toy_quantvla/results/phase13_eager_island_2case_v1_compiled_server_prepare.json
```

| target | success | calls | client p50 | client p90 | server p50 | server p90 |
|---|---:|---:|---:|---:|---:|---:|
| whole action_head.model | 0/4 flip-case | 3964 | 74.1 ms | 148.7 ms | 69.7 ms | 148.7 ms |
| action_head + FFN 8-15 eager island | 1/2 | 1201 | 75.4 ms | 146.5 ms | 70.4 ms | 141.7 ms |
| action_head + blocks 8-15 eager island | 2/2 | 447 | 77.7 ms | 148.1 ms | 72.6 ms | 143.2 ms |

FFN 8-15 eager island 逐 case：

| case | success | calls | client p50 |
|---|---|---:|---:|
| 4:6 | fail | 991 | 75.9 ms |
| 6:0 | success | 210 | 74.3 ms |

Blocks 8-15 eager island 逐 case：

| case | success | calls | client p50 |
|---|---|---:|---:|
| 4:6 | success | 241 | 75.4 ms |
| 6:0 | success | 206 | 79.5 ms |

速度结论：

```text
FFN eager island 和 block eager island 都保住了 whole action_head.model compile 的 p50 速度。
```

行为结论：

```text
FFN eager island 没有解决 task4:init6 回归。
Blocks 8-15 eager island 救回了 task4:init6 和 task6:init0 两个稳定回归 case。
```

## 当前判断

本轮结果把工程路线进一步收窄：

1. `torch._dynamo.disable` 作为局部 eager island，能保留大图速度。
2. 单独保护后半 FFN forward 不够。
3. 保护后半完整 transformer block 后，两个稳定回归 case 都恢复成功。
4. 行为风险不是单个 FFN forward 内部这么简单，更可能来自后半 DiT block 的整段 residual/norm/add/attention/FFN 数据路径。

## 15-Case Matched Set

已经把 `action_head_model_blocks_8_15_eager` 扩大到 15-case matched set：

```text
target: action_head_model_blocks_8_15_eager
case list: 之前 15-case matched set
compare: FP16 baseline vs whole action_head.model compile vs blocks 8-15 eager island
```

输出：

```text
toy_quantvla/results/phase13_block_island_15case_v1_compiled_client_latency.json
toy_quantvla/results/phase13_block_island_15case_v1_compiled_server_latency.json
toy_quantvla/results/phase13_block_island_15case_v1_compiled_server_prepare.json
```

总表：

| policy | success | calls | client p50 | client p90 | server p50 | server p90 |
|---|---:|---:|---:|---:|---:|---:|
| FP16 baseline | 7/15 | 10068 | 160.6 ms | 约 166 ms | 155.3 ms | 162.0 ms |
| whole action_head.model compile | 5/15 | 11316 | 74.6 ms | 约 152 ms | 70.1 ms | 151.9 ms |
| action_head + blocks 8-15 eager island | 7/15 | 10084 | 77.3 ms | 149.1 ms | 72.5 ms | 144.4 ms |

分任务：

| task | baseline | whole compile | blocks 8-15 eager island |
|---|---:|---:|---:|
| task4 | 3/5 | 2/5 | 2/5 |
| task6 | 3/5 | 2/5 | 3/5 |
| task8 | 1/5 | 1/5 | 2/5 |

逐 case：

| case | blocks 8-15 eager island |
|---|---|
| 4:6 | success, 241 calls |
| 4:7 | fail, 991 calls |
| 4:8 | fail, 991 calls |
| 4:9 | fail, 991 calls |
| 4:10 | success, 212 calls |
| 6:0 | success, 206 calls |
| 6:1 | success, 234 calls |
| 6:2 | success, 235 calls |
| 6:3 | fail, 991 calls |
| 6:4 | fail, 991 calls |
| 8:6 | fail, 991 calls |
| 8:7 | fail, 991 calls |
| 8:8 | fail, 991 calls |
| 8:9 | success, 424 calls |
| 8:10 | success, 604 calls |

这条结果比 2-case smoke 更重要：

```text
blocks 8-15 eager island 追平 FP16 baseline 的 7/15，
同时保住 whole compile 的大部分速度收益。
```

它仍不是严格透明替换。task4 从 baseline `3/5` 降到 `2/5`，但 task6 恢复到 baseline `3/5`，task8 从 `1/5` 提升到 `2/5`。这说明 compile/eager-island backend 仍在做闭环轨迹重分配，只是这次总成功率回到了 baseline 水平。

备选路线：

- `action_head.model` 手工分段：blocks 0-7 compile，blocks 8-15 eager，out projection eager/compile 分别测；
- CUDA graph 捕获 eager denoise loop，优先保持数值路径不变。

## 30-Case Matched Set + Memory

把 `action_head_model_blocks_8_15_eager` 进一步扩大到 30-case matched set：

```text
tag: phase13_block_island_30case_mem_v1
case list: task4 init0..9, task6 init0..9, task8 init0..9
compare: FP16 baseline vs blocks 8-15 eager island
```

详细报告：

```text
docs/phase13_block_island_30case_mem_v1.md
```

总表：

| policy | success | calls | client p50 | client p90 | server p50 | server p90 | reserved memory |
|---|---:|---:|---:|---:|---:|---:|---:|
| FP16 baseline | 16/30 | 20050 | 161.1 ms | 166.5 ms | 156.2 ms | 161.5 ms | 5512 MiB |
| blocks 8-15 eager island | 13/30 | 21316 | 81.0 ms | 150.3 ms | 76.5 ms | 145.5 ms | 5538 MiB |

分任务：

| task | baseline | blocks 8-15 eager island |
|---|---:|---:|
| task4 | 6/10 | 5/10 |
| task6 | 6/10 | 5/10 |
| task8 | 4/10 | 3/10 |

逐 case 翻转：

| case | baseline | blocks 8-15 eager island |
|---|---|---|
| 4:9 | success | fail |
| 6:8 | success | fail |
| 8:4 | fail | success |
| 8:7 | success | fail |
| 8:8 | success | fail |

这轮把结论从 15-case 的乐观状态拉回谨慎：

```text
速度收益仍然很强：server p50 约 2.04x，client p50 约 1.99x。
steady-state 显存成本很小：reserved memory 只增加约 26 MiB。
但行为没有追平 baseline：success 从 16/30 降到 13/30。
```

因此 blocks 8-15 eager island 是一个很好的速度候选 backend，但不能作为透明替换。它仍然会改变闭环轨迹分布：既有 `8:4` 这种失败转成功，也有 `4:9`, `6:8`, `8:7`, `8:8` 这种成功转失败。

## 本轮结论

```text
whole action_head.model compile + FFN 8-15 eager island:
速度 OK，行为不 OK。

whole action_head.model compile + blocks 8-15 eager island:
速度 OK，两个稳定回归 case 行为 OK，并在 15-case 上追平 FP16 baseline。
但 30-case 扩大后 success 为 13/30，低于 FP16 baseline 16/30。

下一步应该收缩 patch scope 或改测 CUDA graph，目标是在保留大部分 p50 速度收益的同时减少闭环轨迹重分配。
```
