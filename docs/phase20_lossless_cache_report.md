# Phase 20: Lossless Cache Probe

本阶段只做低风险无损优化，不启用 `torch.compile`，不改权重、不改 dtype、不改模型数学路径。

## 1. 做了什么

新增：

- `toy_quantvla/lossless_cache_patches.py`
- `toy_quantvla/phase20_lossless_cache_probe.py`

并给 `toy_quantvla/timed_fp16_inference_service.py` 增加两个 server 开关：

- `--lossless-cache-prepare-input-pruning`
- `--lossless-cache-action-head-static`

### prepare_input pruning

官方 `GR00T_N1_5.prepare_input` 会把完整 normalized input 同时传给 backbone 和 action head，然后两份都递归 `.to(cuda)`。

但在 inference `get_action` 中：

- backbone 实际只消费 `eagle_*` tensors
- action head 实际只消费 `state` 和 `embodiment_id`

所以 patch 后只移动各自需要的 key：

```text
backbone: eagle_input_ids, eagle_attention_mask, eagle_pixel_values, eagle_image_sizes
action:   state, embodiment_id
```

这是语义保持的，因为 action head 的 inference path 没有读取 `eagle_*`，backbone 也没有读取 state/action fields。

### action head static cache

官方 action head 在 8-step denoising loop 中每步重复构造：

- `timesteps_tensor = torch.full(...)`
- `pos_ids = torch.arange(...)`
- `position_embedding(pos_ids).unsqueeze(0)`
- `future_tokens.weight.unsqueeze(0).expand(...)`

这些在固定 batch size、固定 action horizon、固定 denoising steps 下都是静态的。patch 后缓存：

```text
timesteps: 8 tensors
position_embeddings: 1 tensor
future_tokens view: 1 tensor
```

## 2. Fixed-observation A/B

设置：

- real observation index: 115
- denoising steps: 8
- warmup repeats: 5
- compare repeats: 3
- profile repeats: 100
- no `torch.compile`

### Correctness

所有缓存版本都通过动作一致性检查：

| Variant | Bitwise equal after unapply | Max abs diff |
| --- | ---: | ---: |
| prepare pruning only | true | 0.0 |
| prepare pruning + static cache | true | 0.0 |

这说明当前 patch 没有引入 open-loop action drift。

### Latency

| Variant | Baseline p50 | Cached p50 | p50 speedup | Baseline mean | Cached mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| prepare pruning only | 103.48 ms | 103.60 ms | 0.999x | 114.83 ms | 109.90 ms |
| prepare pruning + static cache | 106.74 ms | 105.21 ms | 1.015x | 118.94 ms | 114.22 ms |

解释：

- p50 收益很小，约 0-1.5%。
- mean 有约 4% 左右改善，说明缓存/剪枝减少了一些长尾开销，但不是主瓶颈。
- static cache 的命中正常：100-repeat probe 中只有 8 个 timestep miss、1 个 position embedding miss、1 个 future token miss，之后均为 hit。

## 3. 2-case closed-loop smoke

设置：

```text
case-list: 4:9,6:8
server: FP16 + prepare pruning + action head static cache
```

结果：

| Metric | Value |
| --- | ---: |
| Success | 2 / 2 |
| Client policy p50 | 157.69 ms |
| Client policy p90 | 165.78 ms |
| Client policy p99 | 184.66 ms |
| Server get_action p50 | 152.51 ms |
| Server get_action p90 | 160.54 ms |
| Server get_action p99 | 178.73 ms |
| Server reserved memory | 5.78 GB |

参考 Phase19：

- 15-case FP16 server p50: 154.82 ms
- 15-case FP16 client p50: 159.99 ms
- earlier 2-case FP16 client p50: about 159.7 ms

所以 closed-loop smoke 也显示同方向的小幅收益，约 1-2%。但因为 workload 不完全相同，这不能当作严格 speedup claim。

## 4. 结论

这轮缓存是正确的，但不是大收益来源。

可以保留：

- 它通过 bitwise equality。
- server rollout 没有破坏成功样本。
- 代码开关可控，不影响默认 FP16 baseline。
- 对 mean / p90 有一些改善迹象。

但不要把它包装成主要加速结果：

- p50 speedup 只有约 1% 量级。
- Phase19 profiler 的主瓶颈仍然是大量 GEMM / linear / kernel launch。
- 当前 patch 只减少少量 `.to/copy` 和固定 tensor 构造，碰不到主要 CUDA compute。

## 5. 下一步

下一步建议继续无损路线，但目标要转向更大的固定 shape 开销：

1. transform/tokenization cache census  
   观察同一 task 内 `eagle_input_ids`、`attention_mask`、`image_sizes` 是否固定；如果固定，可以尝试缓存语言 tokenization 和 chat template，而不是缓存完整图像输入。

2. module-level timing with cache enabled  
   用 `--profile-linear-modules` 对 cached server 再拆一次，确认缓存没有改变主要热点结构。

3. CUDA Graph / graph replay prototype  
   不直接进 rollout。先做 same-observation equality gate，再看 fixed-shape replay 的 latency。这里才可能吃到 kernel launch 数量过高带来的收益。

4. shape bucket design  
   按 task language length / image token length 做 bucket。每个 bucket 预热和缓存固定 tensors，为后续 CUDA Graph 做准备。

## 6. Artifacts

Code:

- `toy_quantvla/lossless_cache_patches.py`
- `toy_quantvla/phase20_lossless_cache_probe.py`
- `toy_quantvla/timed_fp16_inference_service.py`

Results:

- `toy_quantvla/results/phase20_lossless_cache_probe_prune_only_obs115_r100.json`
- `toy_quantvla/results/phase20_lossless_cache_probe_static_prune_obs115_r100.json`
- `toy_quantvla/results/phase20_cached_2case_client_latency.json`
- `toy_quantvla/results/phase20_cached_2case_server_latency.json`
- `toy_quantvla/results/phase20_cached_2case_server_prepare.json`
- `toy_quantvla/results/phase20_cached_2case_eval.log`

