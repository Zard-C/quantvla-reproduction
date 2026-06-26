# Phase 19: FP16 Hygiene Baseline and No-Loss Profiling

本阶段目标不是证明量化加速，而是先回答一个更工程化的问题：

1. 当前官方 FP16 closed-loop rollout 的稳定速度和显存是多少？
2. 在不改变数值语义的前提下，真正值得优化的是哪一段？
3. 如果后续做无损加速，应该优先验证哪些开关？

结论先行：当前还没有启用加速 backend，所以加速比仍是 1.00x。Phase19 建立的是 FP16 baseline 和 profiler 证据。当前证据显示，瓶颈主要在模型内部大量 linear/GEMM 和 kernel launch，而不是 LIBERO client 侧 pre/post、网络包装、显式 D2H 同步。

## 1. 15-case FP16 baseline

运行边界：

- 模型：`gr00t-n1.5-libero-long-posttrain`
- task suite：`libero_10`
- denoising steps：8
- policy seed：deterministic，base = 20260613
- case list：`4:6,4:7,4:8,4:9,4:10,6:0,6:1,6:2,6:3,6:4,8:6,8:7,8:8,8:9,8:10`
- server：官方 FP16 `Gr00tPolicy` + server-side `get_action` timing

整体结果：

| Metric | Value |
| --- | ---: |
| Success | 7 / 15 |
| Success rate | 46.67% |
| Client policy count | 10068 |
| Client policy p50 | 159.99 ms |
| Client policy p90 | 167.13 ms |
| Client policy p99 | 188.60 ms |
| Server get_action count | 10000 |
| Server get_action p50 | 154.82 ms |
| Server get_action p90 | 162.22 ms |
| Server get_action p99 | 182.60 ms |
| Server allocated memory | 5.48 GB |
| Server reserved memory | 5.78 GB |

按任务拆分：

| Task id | Success | Total steps | Mean case p50 |
| ---: | ---: | ---: | ---: |
| 4 | 3 / 5 | 2663 | 154.54 ms |
| 6 | 3 / 5 | 3056 | 148.02 ms |
| 8 | 1 / 5 | 4349 | 161.02 ms |

解释：

- 这组 case 不是高成功率 benchmark，而是一个混合难度 workload。
- task4、task6 都是 3/5，task8 是 1/5，能覆盖成功、失败、长 horizon 三类 rollout。
- 失败 case 里常见 `executing action in terminated episode`，仍符合之前判断：多数是 LIBERO/robosuite horizon 后 eval loop 多 step 一次，不是 inference server crash。
- client policy p50 比 server get_action p50 高约 5 ms，说明主要时间在 server 内模型推理，client pre/post 和 RPC 包装不是第一优先级。

## 2. Fixed-observation torch.profiler probe

运行边界：

- 输入：真实 LeRobot observation，dataset index 115
- warmup repeats：2
- profile repeats：3
- profiler：CPU + CUDA，record shapes + profile memory
- 注意：profiler latency 含 profiling overhead，不能直接当部署延迟；这里只用于热点归因。

Profiler wall-clock：

| Metric | Value |
| --- | ---: |
| Model load | 11.63 s |
| Warmup requests | 2 |
| Profile requests | 3 |
| Profile request p50 | 163.18 ms |
| Profile request p90 | 203.77 ms |
| Profile request p99 | 212.90 ms |
| Profile reserved memory | 5.78 GB |

Top CUDA operations/events：

| Event | Count | Self CUDA | Note |
| --- | ---: | ---: | --- |
| `aten::addmm` | 3357 | 54.53 ms | dense linear main path |
| BF16 CUTLASS GEMM kernel | 1884 | 23.08 ms | child GEMM kernel |
| `aten::mm` | 258 | 18.24 ms | matmul path |
| BF16 CUTLASS GEMM kernel | 435 | 11.29 ms | child GEMM kernel |
| `aten::copy_` | 4134 | 7.70 ms | dtype/device/layout copy path |
| `aten::_flash_attention_forward` | 396 | 3.50 ms | attention is present but not dominant |
| `aten::native_layer_norm` | 987 | 2.68 ms | normalization overhead |

Top CPU/launch signals：

| Event | Count | Self CPU | Interpretation |
| --- | ---: | ---: | --- |
| `cudaLaunchKernel` | 12810 | 53.07 ms | very high launch count, about 4270 launches/request |
| `aten::addmm` | 3357 | 41.60 ms | linear dispatch overhead plus GPU work |
| `aten::linear` | 5556 | 21.26 ms self / 178.15 ms total | linear stack dominates operation-level total |
| `aten::copy_` | 4134 | 15.03 ms | many copy/cast/layout operations |
| `aten::_to_copy` | 3948 | 10.02 ms | many dtype/device conversions |
| `aten::native_layer_norm` | 987 | 8.99 ms | repeated small op overhead |

Sync / D2H candidates：

| Event | Count | CPU total | CUDA total | Interpretation |
| --- | ---: | ---: | ---: | --- |
| `cudaMemcpyAsync` | 90 | 1.22 ms | 0 ms | not a dominant D2H bottleneck |
| `cudaStreamSynchronize` | 90 | 0.39 ms | 0 ms | not dominant |
| `aten::item` | 96 | 0.79 ms | 0.01 ms | exists, but too small to explain latency |
| `cudaDeviceSynchronize` | 4 | 0.05 ms | 0.01 ms | mostly profiler/manual sync boundary |

一个注意点：v2 profiler 中 `aten::pow` 出现过较高 CPU self time，但 v1 run 没有同等强度信号。这个点应视为需要复测的候选，而不是立即认定为主要瓶颈。

## 3. 对无损加速路线的判断

当前不支持“先优化 pre/post 就能明显加速”的判断。preprocess/postprocess 的 p50 只有约 0.1 ms，client/server gap 也只有数毫秒。真正可争取的空间在 server 内部：

1. 减少 kernel launch 数量  
   每个 profiled request 大约有 4270 次 `cudaLaunchKernel`。这说明小 kernel 很碎，存在 CUDA Graph、`torch.compile(mode="reduce-overhead")`、局部融合或缓存固定 shape 路径的机会。

2. 处理大量 linear/GEMM  
   `aten::linear` / `aten::addmm` / `aten::mm` 是最主要的操作级热点。纯 FP16/BF16 GEMM 已经走 CUTLASS/cuBLAS 类 kernel，所以无损自写 GEMM 的收益未必大；更现实的是减少周边 launch、copy、layout conversion，以及稳定 shape 后让编译器更容易融合。

3. 缩减 copy / dtype conversion  
   `aten::to`、`aten::_to_copy`、`aten::copy_` 数量很高。它们不是第一大 CUDA 时间，但会贡献 CPU overhead 和内存流量。后续应查这些 copy 是否来自每步重复构造 observation tensor、task text、action head timestep / mask / position buffer。

4. 小心 torch.compile / CUDAGraph  
   之前 Phase13-17 已经看到 compile path 可能引入 closed-loop 敏感差异。无损加速必须先做 same-observation action equality / drift check，再进 closed-loop rollout。优先测试更小 scope，而不是一上来编译整个 action head。

## 4. 下一步建议

短期建议按这个顺序做：

1. 加 module-level FP16 timing  
   用已有 `--profile-linear-modules` 跑同一 observation，拆出 LLM、DiT MLP、DiT attention、projector 哪些 linear scope 最耗时。

2. 做 request trace + no-loss equality gate  
   对同一 observation 固定 seed，比较 eager vs candidate accelerated path 的 action tensor：max abs、cosine、是否 bitwise/near-equal。只有通过这个 gate 才进入 rollout。

3. 先试局部 `torch.compile` / CUDA Graph  
   优先从最小安全 scope 开始，例如 projector 或单个稳定 MLP island。每个 scope 记录 cold compile time、warm latency、action drift、closed-loop success。

4. 查固定 shape / 固定 buffer  
   重点看 task text、position ids、timestep embeddings、attention masks、RoPE / positional components 是否每 step 重建或 CPU->GPU copy。这里如果能缓存，是最可能无损且工程成本较低的收益点。

5. 暂不把低比特 kernel 当作“无损加速”  
   FP4/W4A16 是量化路线，会改变数值和闭环轨迹；它适合另一个实验分支。Phase19 的无损路线应先建立 FP16 semantic-preserving speedups。

## 5. Artifacts

Local artifacts:

- `toy_quantvla/phase19_torch_profiler_probe.py`
- `toy_quantvla/results/phase19_fp16_hygiene_15case_client_latency.json`
- `toy_quantvla/results/phase19_fp16_hygiene_15case_server_latency.json`
- `toy_quantvla/results/phase19_fp16_hygiene_15case_server_prepare.json`
- `toy_quantvla/results/phase19_fp16_hygiene_15case_eval.log`
- `toy_quantvla/results/phase19_torch_profiler_probe_obs115_r3_v2.json`
- `toy_quantvla/results/phase19_torch_profiler_probe_obs115_r3_v2_cpu_table.txt`
- `toy_quantvla/results/phase19_torch_profiler_probe_obs115_r3_v2_cuda_table.txt`
- `toy_quantvla/results/phase19_torch_profiler_probe_obs115_r3_v2.log`

Remote-only large artifact:

- `toy_quantvla/results/phase19_torch_profiler_probe_obs115_r3_v2_trace.json`

