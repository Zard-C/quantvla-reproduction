# Phase 13.2 Blocks 0-7 Compile Mark-Step Probe

本轮目标是验证一个更保守的 compile scope：

```text
target: action_head_dit_blocks_0_7
idea: 只 compile DiT 前 8 个 transformer blocks，外层 action_head.model、后 8 个 blocks、final norm/action decoder 保持 eager
```

它比 `action_head_model_blocks_8_15_eager` 更保守，因为后者仍然 compile 整个 `action_head.model` 大图，只是在后半 blocks 周围打 eager island。

## 为什么要加 mark-step

直接跑：

```text
COMPILE_TARGET=action_head_dit_blocks_0_7
COMPILE_MODE=reduce-overhead
```

会在 prewarm 触发 Inductor CUDAGraph overwrite：

```text
RuntimeError: accessing tensor output of CUDAGraphs that has been overwritten by a subsequent run
hint: clone the tensor outside of torch.compile() or call torch.compiler.cudagraph_mark_step_begin()
```

原因是这个 target 会把 `transformer_blocks.0..7` 分别变成 compiled submodules。它们在外层 DiT forward 和 8-step denoising loop 里重复调用，`reduce-overhead` 的 CUDAGraph 复用会撞上中间 hidden state 的生命周期。

第一版尝试在每个 compiled block forward 前调用 `cudagraph_mark_step_begin()`，仍然失败。原因是这个边界太细：block0 的输出还要被 block1 消费，不能把每个 block 当作一个新 CUDAGraph step。

修正后实现为：

```text
--torch-compile-cudagraph-mark-step
```

对 `action_head_dit_*` 小 scope target，在外层：

```text
policy.model.action_head.model.forward
```

入口调用 `torch.compiler.cudagraph_mark_step_begin()`。也就是每次 DiT model invocation 开一个新 CUDAGraph step，而不是每个 block 开一个。

相关代码：

```text
toy_quantvla/phase13_compile_targets.py
toy_quantvla/timed_fp16_inference_service.py
toy_quantvla/run_phase13_torch_compile_matched_set.sh
```

## 5-Case Baseline

case list 选自 30-case 中 blocks 8-15 eager island 的翻转点：

```text
4:9,6:8,8:4,8:7,8:8
```

输出：

```text
toy_quantvla/results/phase13_blocks_0_7_5flip_v1_baseline_client_latency.json
toy_quantvla/results/phase13_blocks_0_7_5flip_v1_baseline_server_latency.json
toy_quantvla/results/phase13_blocks_0_7_5flip_v1_baseline_server_prepare.json
```

结果：

| policy | success | calls | client p50 | server p50 | server p90 |
|---|---:|---:|---:|---:|---:|
| FP16 baseline | 3/5 | 3370 | 160.7 ms | 155.9 ms | 163.2 ms |

逐 case：

| case | result | calls | client p50 |
|---|---|---:|---:|
| 4:9 | success | 224 | 159.3 ms |
| 6:8 | success | 649 | 160.5 ms |
| 8:4 | success | 515 | 161.8 ms |
| 8:7 | fail | 991 | 148.3 ms |
| 8:8 | fail | 991 | 162.6 ms |

注意：这个 baseline 和 30-case baseline 不完全一致。30-case 中 `8:4` 是 fail，`8:7/8:8` 是 success；本轮变成 `8:4` success、`8:7/8:8` fail。这进一步说明这些 case 是高敏感边界样本，小规模 flip set 只能做压力测试，不能当稳定统计结论。

## Default Mode 诊断

为了绕开 reduce-overhead 的 CUDAGraph overwrite，先跑：

```text
COMPILE_TARGET=action_head_dit_blocks_0_7
COMPILE_MODE=default
case: 4:9
```

观察：

| policy | case | result | calls | server p50 |
|---|---|---|---:|---:|
| FP16 baseline | 4:9 | success | 224 | about 155.9 ms |
| blocks 0-7 default | 4:9 | fail | 991 | about 157.7 ms |

结论：`default` 可以避开 runtime crash，但基本没有速度收益，并且第一条就从 success 变 fail。这条不值得继续扩大。

## Reduce-Overhead + Mark-Step Smoke

配置：

```text
tag: phase13_blocks_0_7_markstep_4_9_smoke_v2
case list: 4:9
target: action_head_dit_blocks_0_7
mode: reduce-overhead
backend: inductor
cudagraph_mark_step: true
mark boundary: policy.model.action_head.model
```

输出：

```text
toy_quantvla/results/phase13_blocks_0_7_markstep_4_9_smoke_v2_compiled_client_latency.json
toy_quantvla/results/phase13_blocks_0_7_markstep_4_9_smoke_v2_compiled_server_latency.json
toy_quantvla/results/phase13_blocks_0_7_markstep_4_9_smoke_v2_compiled_server_prepare.json
```

结果：

| policy | case | result | calls | client p50 | server p50 | server p90 |
|---|---|---|---:|---:|---:|---:|
| FP16 baseline | 4:9 | success | 224 | 159.3 ms | about 155.9 ms | about 163.2 ms |
| blocks 0-7 + mark-step | 4:9 | fail | 991 | 125.7 ms | 121.4 ms | 145.9 ms |

prewarm:

| policy | prewarm mean | prewarm p50 | prewarm max |
|---|---:|---:|---:|
| blocks 0-7 + mark-step | 1381.6 ms | 1311.1 ms | 2831.2 ms |

显存：

| policy | allocated | reserved | max allocated | max reserved |
|---|---:|---:|---:|---:|
| blocks 0-7 + mark-step | 5223.1 MiB | 5538.0 MiB | 5419.4 MiB | 5538.0 MiB |

解读：

- mark-step 修复了 reduce-overhead 的 runtime crash。
- p50 有一定速度收益：server p50 从约 `156 ms` 到 `121 ms`。
- 但它明显慢于 blocks 8-15 eager island 的 `76.5 ms` server p50。
- 第一条关键 case `4:9` 仍然 success -> fail。

## 当前结论

`action_head_dit_blocks_0_7` 不是一个有吸引力的工程路线：

1. `reduce-overhead` 不加 mark-step 会直接 CUDAGraph overwrite 崩溃。
2. `default` 能跑但没有速度收益。
3. `reduce-overhead + mark-step` 能跑，也有一点速度收益，但第一条边界 case 仍失败。
4. 小 scope compile 失去了整块 DiT compile 的大图融合优势，同时没有消除闭环轨迹重分配。

保留的工程价值是：

```text
--torch-compile-cudagraph-mark-step
```

这个开关以后可以用于诊断 repeated submodule compile 与 CUDAGraph 的边界问题，但它不是当前最好的加速 backend。

## 下一步

继续追求推理速度时，不建议在 `action_head_dit_blocks_0_7` 上扩大 rollout。

更值得推进的是：

1. CUDA graph 捕获 eager denoising loop：
   - 尽量保持 eager 数值路径；
   - 目标是减少 Python/kernel launch overhead；
   - 如果成功，可能比 torch.compile 小 scope 更保守。
2. 继续沿 `action_head_model_blocks_8_15_eager` 做行为修复：
   - 它有最强的实际速度收益；
   - 但必须围绕 regression cases `4:9`, `6:8`, `8:7`, `8:8` 继续收缩风险。
3. 对边界 baseline 做复验：
   - 本轮 `8:4`, `8:7`, `8:8` baseline 和 30-case 不一致；
   - 后续对比需要多次 repeat 或更大的 init set，不能只看单次 flip。
