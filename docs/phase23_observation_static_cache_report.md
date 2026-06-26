# Phase 23: Observation Static Tensor Cache

本阶段回答一个具体问题：

> 是否已经避免每步重新构造 observation dict 里那些固定 tensor？

结论：之前没有完整做。Phase20 只做了模型入口后的 `prepare_input` 分支剪枝和 action-head denoising 小 tensor cache；Phase23 新增了 observation / transform 相关的静态 cache。

## 1. 新增内容

新增两个独立开关：

- `--lossless-cache-eagle-tokenizer`
- `--lossless-cache-static-normalized-input`

已接入：

- `toy_quantvla/lossless_cache_patches.py`
- `toy_quantvla/phase20_lossless_cache_probe.py`
- `toy_quantvla/phase19_torch_profiler_probe.py`
- `toy_quantvla/phase10_fp16_linear_hotspot_profile.py`
- `toy_quantvla/timed_fp16_inference_service.py`

### 1.1 Eagle tokenizer cache

官方 `GR00TTransform` 每个 `get_action` 都会构造 Eagle prompt，并调用 tokenizer 生成：

- `eagle_input_ids`
- `eagle_attention_mask`

同一个 LIBERO task 内，语言描述和图像 token layout 通常固定，所以展开后的 prompt text 是稳定的。Phase23 给 Eagle tokenizer 包了一层 proxy：

```text
cache key = expanded prompt text + tokenizer kwargs
```

命中后复用 tokenizer 输出，不再每步重建 `input_ids / attention_mask`。

### 1.2 Static normalized-input CUDA cache

在 `prepare_input pruning` 基础上，进一步缓存小的静态 normalized tensor 的 CUDA 版本：

```text
backbone static:
  eagle_input_ids
  eagle_attention_mask
  eagle_image_sizes

action static:
  embodiment_id
```

这些 tensor 对同一 task / fixed camera layout 是固定的，且模型不会原地修改它们。

明确不缓存：

- `eagle_pixel_values`: 每步图像变化
- `state`: 每步机器人状态变化

所以该 patch 不改变模型数学路径。

## 2. Fixed-observation A/B

设置：

- observation index: `115`
- warmup repeats: `5`
- compare repeats: `3`
- profile repeats: `100`
- no CUDA Graph
- no torch.compile

### 2.1 Obs/static cache

开关：

```text
--eagle-tokenizer-cache
--prepare-input-pruning
--static-normalized-input-cache
```

Correctness:

| Metric | Value |
| --- | ---: |
| Bitwise equal after unapply | true |
| Max abs diff | 0 |

Latency:

| Path | mean | p50 | p90 | p99 |
| --- | ---: | ---: | ---: | ---: |
| baseline | 113.87 ms | 104.32 ms | 177.67 ms | 195.68 ms |
| cached | 107.79 ms | 102.68 ms | 114.81 ms | 181.80 ms |

Speedup:

- p50: `1.016x`
- mean: about `1.056x`

Cache stats:

| Cache | Hit | Miss |
| --- | ---: | ---: |
| Eagle tokenizer | 107 | 1 |
| static CUDA `eagle_input_ids` | 107 | 1 |
| static CUDA `eagle_attention_mask` | 107 | 1 |
| static CUDA `eagle_image_sizes` | 107 | 1 |
| static CUDA `embodiment_id` | 107 | 1 |

### 2.2 Full eager cache

开关：

```text
--eagle-tokenizer-cache
--prepare-input-pruning
--static-normalized-input-cache
--action-head-static-cache
```

Correctness:

| Metric | Value |
| --- | ---: |
| Bitwise equal after unapply | true |
| Max abs diff | 0 |

Latency:

| Path | mean | p50 | p90 | p99 |
| --- | ---: | ---: | ---: | ---: |
| baseline | 119.85 ms | 104.96 ms | 185.74 ms | 199.05 ms |
| cached | 108.87 ms | 102.42 ms | 130.34 ms | 183.75 ms |

Speedup:

- p50: `1.025x`
- mean: about `1.101x`

说明：在 eager path 下，obs/static cache 可以和 action-head static cache 组合，并保持 bitwise equality。

但不要把这个组合直接用于 CUDA Graph。Phase22 已经发现：

```text
action-head static cache + CUDA Graph -> max abs drift 0.03284
```

所以 CUDA Graph 路线仍建议只组合：

```text
prepare-input pruning + static normalized-input cache + Eagle tokenizer cache
```

## 3. Cached torch.profiler 对比

设置：

- observation index: `115`
- profile repeats: `3`
- record shapes + profile memory
- full eager cache enabled

Request-level profiler latency:

| Path | mean | p50 | p90 | p99 |
| --- | ---: | ---: | ---: | ---: |
| Phase19 baseline profiler | 176.27 ms | 163.18 ms | 203.77 ms | 212.90 ms |
| Phase23 cached profiler | 144.68 ms | 144.43 ms | 145.35 ms | 145.56 ms |

注意：torch.profiler 绝对时延不能当部署时延，只用于热点归因。

关键事件变化：

| Event | Count before | Count after | Self CPU before | Self CPU after |
| --- | ---: | ---: | ---: | ---: |
| `aten::to` | 4635 | 4599 | 3.59 ms | 3.17 ms |
| `aten::_to_copy` | 3948 | 3906 | 10.02 ms | 9.56 ms |
| `aten::copy_` | 4134 | 4086 | 15.03 ms | 14.15 ms |
| `cudaMemcpyAsync` | 90 | 54 | 1.22 ms | 0.89 ms |
| `cudaStreamSynchronize` | 90 | 54 | 0.39 ms | 0.23 ms |
| `cudaLaunchKernel` | 12810 | 12738 | 53.07 ms | 50.37 ms |

模型主路径基本没变：

| Event | Count before | Count after | Self CUDA before | Self CUDA after |
| --- | ---: | ---: | ---: | ---: |
| `aten::addmm` | 3357 | 3357 | 54.53 ms | 54.48 ms |
| `aten::mm` | 258 | 258 | 18.24 ms | 18.24 ms |
| `aten::_flash_attention_forward` | 396 | 396 | 3.50 ms | 3.48 ms |
| `aten::native_layer_norm` | 987 | 987 | 2.68 ms | 2.64 ms |

解释：

- obs/static cache 确实减少了一批输入相关 copy / to / memcpy / launch。
- 但 Linear/GEMM、attention、layer norm 的调用数和 CUDA 时间基本不变。
- 所以它是低风险 hygiene 优化，不是主加速引擎。

## 4. 还能做什么无损加速

从 Phase19 + Phase21 + Phase23 的 profiler 看，剩余空间主要在这几类：

1. CUDA Graph / fixed-shape replay  
   Phase22 已经证明 `prepare-only graph` fixed replay drift 为 0，p50 约 `1.06x`，p90 约 `1.5x`。下一步应接入 inference service，但要避开 action-head static cache。

2. 图像 processor 路径拆分  
   当前仍然每步把两路图像转 PIL，再走 Eagle image processor。图像本身不能缓存，但固定 shape 下可以研究是否减少 PIL 往返和 Python wrapper 开销。这个需要非常小心，因为 image processor 输出必须 bitwise 对齐。

3. `aten::pow` outlier 复查  
   Phase19 中 `aten::pow` 出现过 63.7 ms CPU self，但 Phase23 cached profiler 只有 1.13 ms。这个更像 profiler/run-level outlier，不能当作稳定瓶颈。若要继续，需要单独 repeat profiler 验证。

4. Linear/GEMM 之外的小 op fusion  
   `add / layer_norm / transpose / reshape / view / cat` 数量很多。纯 eager 下难以无损融合；更现实的路径是 CUDA Graph 压 launch，或者非常局部地 `torch.compile`，但 compile 已有闭环行为风险。

5. 真正大加速仍要回到 kernel / quant 路线  
   `aten::addmm/mm` 的 CUDA 时间没有被这些 cache 改变。如果目标是显著降低 p50，仍然需要实际低比特 GEMM、CUTLASS backend、或更稳的 graph/fusion 路线。

## 5. 当前建议

短期可落地的无损组合：

```text
eager server:
  --lossless-cache-eagle-tokenizer
  --lossless-cache-prepare-input-pruning
  --lossless-cache-static-normalized-input
  --lossless-cache-action-head-static

cuda graph server:
  --lossless-cache-eagle-tokenizer
  --lossless-cache-prepare-input-pruning
  --lossless-cache-static-normalized-input
  --cuda-graph-dit-forward
```

其中 CUDA Graph server 的 `--cuda-graph-dit-forward` 还需要接入 inference service。

## 6. Artifacts

Code:

- `toy_quantvla/lossless_cache_patches.py`
- `toy_quantvla/phase20_lossless_cache_probe.py`
- `toy_quantvla/phase19_torch_profiler_probe.py`
- `toy_quantvla/phase10_fp16_linear_hotspot_profile.py`
- `toy_quantvla/timed_fp16_inference_service.py`

Results:

- `toy_quantvla/results/phase23_lossless_obs_static_cache_obs115_r100.json`
- `toy_quantvla/results/phase23_lossless_obs_static_plus_action_cache_obs115_r100.json`
- `toy_quantvla/results/phase23_torch_profiler_obs_static_full_cache_obs115_r3.json`
- `toy_quantvla/results/phase23_torch_profiler_obs_static_full_cache_obs115_r3_cpu_table.txt`
- `toy_quantvla/results/phase23_torch_profiler_obs_static_full_cache_obs115_r3_cuda_table.txt`

