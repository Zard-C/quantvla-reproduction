# Phase 21: Cached Module-Level Linear Hotspot

本阶段对应 Phase20 下一步 2：在启用 lossless cache 后重新做模块级 Linear timing，确认 cache 是否改变主要热点结构。

## 1. 实验设置

远端机器：

- GPU: RTX 5090
- observation: LIBERO real obs index 115
- denoising steps: 8
- warmup repeats: 3
- profile repeats: 10

对比：

| Scope | Variant |
| --- | --- |
| `llm_dit_mlp` | FP16 baseline |
| `llm_dit_mlp` | FP16 + prepare-input pruning + action-head static cache |
| `dit_mlp_only` | FP16 baseline |
| `dit_mlp_only` | FP16 + prepare-input pruning + action-head static cache |

注意：这个 profiler 会在每个 wrapped `nn.Linear` 后执行 `torch.cuda.synchronize()`，所以绝对 latency 不是部署 latency。它适合用来观察热点结构，而不是直接当线上时延。

## 2. Profiler 更新

`toy_quantvla/phase10_fp16_linear_hotspot_profile.py` 新增：

- `--lossless-cache-prepare-input-pruning`
- `--lossless-cache-action-head-static`

并在输出中记录：

- `lossless_cache`
- `lossless_cache_stats`
- `p50_times_calls_seconds`
- `p90_times_calls_seconds`

原因是同步 profiling 容易出现少数长尾 outlier。`mean * calls` 是实际记录的总和，但很容易被少数异常同步点污染；`p50 * calls` 更适合用来判断稳定热点结构。

## 3. 请求级 timing

单位：ms/request。

| Scope | Variant | p50 | mean | p90 | p99 |
| --- | --- | ---: | ---: | ---: | ---: |
| `llm_dit_mlp` | baseline | 110.27 | 126.42 | 185.36 | 193.40 |
| `llm_dit_mlp` | cached | 102.61 | 112.00 | 132.44 | 179.58 |
| `dit_mlp_only` | baseline | 104.19 | 119.11 | 185.63 | 187.57 |
| `dit_mlp_only` | cached | 99.39 | 112.05 | 155.01 | 184.46 |

请求级结果显示 cached 版本有一些改善：

- `llm_dit_mlp`: p50 speedup about `1.075x`
- `dit_mlp_only`: p50 speedup about `1.048x`

但这个结果要保守解释，因为 profiler 本身引入了大量同步点。它说明 cache 可能减少了一些非 Linear 开销和长尾，但不能说明主 GEMM 被加速了。

## 4. Linear 稳健总量

这里使用 `sum(module_p50 * module_calls)`，更适合看热点结构。

| Scope | Baseline | Cached | Ratio |
| --- | ---: | ---: | ---: |
| `llm_dit_mlp` | 167.51 ms | 166.04 ms | `1.009x` |
| `dit_mlp_only` | 104.88 ms | 104.15 ms | `1.007x` |

结论：启用 cache 后，Linear 稳健总量只变化约 1%。这和 Phase20 的结论一致：lossless cache 是低风险小优化，不是主要加速来源。

## 5. Hotspot 结构

`llm_dit_mlp` scope 下，按 `p50 * calls` 计算的 family share：

| Family | Baseline share | Cached share |
| --- | ---: | ---: |
| `dit_mlp` | 62.79% | 62.85% |
| `llm_mlp` | 24.10% | 24.18% |
| `llm_attn` | 13.11% | 12.97% |

按 suffix 看也几乎不变：

| Suffix | Baseline share | Cached share |
| --- | ---: | ---: |
| DiT `proj` | 33.21% | 33.06% |
| DiT `2` | 29.58% | 29.78% |
| LLM `gate_proj` | 8.91% | 8.90% |
| LLM `up_proj` | 8.52% | 8.57% |
| LLM `down_proj` | 6.67% | 6.71% |
| LLM `o_proj` | 3.85% | 3.77% |
| LLM `q_proj` | 3.43% | 3.44% |
| LLM `k_proj` | 2.99% | 2.94% |
| LLM `v_proj` | 2.83% | 2.83% |

这说明 cache 没有改变主要计算热点：

1. DiT MLP 仍然是最大热点。
2. LLM MLP 次之。
3. LLM attention projection 更小，但仍然可见。
4. 后续真正想拿到大收益，仍然要碰 GEMM、kernel launch、graph replay 或实际低比特 kernel。

## 6. Cache 命中情况

cached r10 每次实验总共 13 次 `get_action`，即 3 次 warmup + 10 次 profile。

| Item | Hit | Miss | Cache size |
| --- | ---: | ---: | ---: |
| denoising timestep tensors | 96 | 8 | 8 |
| position embedding | 103 | 1 | 1 |
| future token view | 12 | 1 | 1 |

命中是正常的：

- 8 个 timestep miss 对应 8-step denoising 的第一次构造。
- position embedding 和 future token 各只有第一次 miss。
- 后续基本都是 cache hit。

## 7. 结论

Phase21 进一步确认：

> lossless cache 是正确且低风险的小优化，但它没有改变 Linear/GEMM 主热点结构。

它的收益主要来自：

- 减少不必要的 normalized input branch copy / `.to(cuda)`
- 减少 denoising loop 里的小 tensor 构造
- 减少少量 CPU/CUDA allocation 与同步长尾

它没有解决：

- 8-step denoising 导致的 DiT MLP 重复调用
- LLM/DiT Linear 的 GEMM 时间
- 大量小 kernel launch 的调度开销

因此当前路线应该这样定位：

1. 保留 lossless cache，作为默认可选安全优化。
2. 不把它包装成主要 speedup claim。
3. 下一步若继续无损加速，应优先看 CUDA Graph / fixed-shape replay。
4. 如果目标是显著加速 `llm_dit_mlp_only`，核心仍然是实际 FP4/W4A16 kernel 或更强的 graph/fusion 路线。

## 8. Artifacts

Code:

- `toy_quantvla/phase10_fp16_linear_hotspot_profile.py`
- `toy_quantvla/lossless_cache_patches.py`

Results:

- `toy_quantvla/results/phase21_fp16_linear_hotspot_llm_dit_mlp_obs115_r10_baseline.json`
- `toy_quantvla/results/phase21_fp16_linear_hotspot_llm_dit_mlp_obs115_r10_cached.json`
- `toy_quantvla/results/phase21_fp16_linear_hotspot_dit_mlp_only_obs115_r10_baseline.json`
- `toy_quantvla/results/phase21_fp16_linear_hotspot_dit_mlp_only_obs115_r10_cached.json`

