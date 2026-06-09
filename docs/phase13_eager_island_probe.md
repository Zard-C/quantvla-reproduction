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

下一步更合理的是把 blocks 8-15 eager island 扩大到更多 matched cases，而不是立刻手工拆 DiT forward：

```text
target: action_head_model_blocks_8_15_eager
case list: 之前 15-case matched set
compare: FP16 baseline vs whole action_head.model compile vs blocks 8-15 eager island
```

备选路线：

- `action_head.model` 手工分段：blocks 0-7 compile，blocks 8-15 eager，out projection eager/compile 分别测；
- CUDA graph 捕获 eager denoise loop，优先保持数值路径不变。

## 本轮结论

```text
whole action_head.model compile + FFN 8-15 eager island:
速度 OK，行为不 OK。

whole action_head.model compile + blocks 8-15 eager island:
速度 OK，两个稳定回归 case 行为 OK。

下一步应该扩大到 15-case matched set 做正式复验。
```
