# Phase 8 CUTLASS FP4 Server Rollout Smoke

日期：2026-06-08

## 边界

本阶段把 cached Triton packer + CUTLASS blockscaled FP4 Linear 从 offline `get_action` smoke 接到了 inference server 路径，并在 LIBERO simulator 里跑了一个最小端到端 episode。

这不是最终性能结论：

- 只 patch 了 `dit_mlp_only` 的 1 个 Linear；
- 只跑了 `libero_10` 的 `task 6:init 1` 一个 episode；
- 速度统计是 eval client 侧 policy request latency，包含 ZMQ 往返、server policy 计算和 Python 调度；
- 还没有跑 full DiT MLP / `llm_mlp_dit_mlp` / `llm_dit_mlp` 的 rollout。

## 新增工程入口

新增：

- `toy_quantvla/cutlass_fp4_inference_service.py`

修改：

- `toy_quantvla/libero_eval_init_range.py`

server wrapper 做的事情：

1. 加载官方 `Gr00tPolicy`；
2. patch 指定 scope 的 `nn.Linear` 为 `CutlassBlockscaledFP4Linear`；
3. 可选读取真实 LeRobot observation 做 prewarm；
4. prewarm 会触发 CUTLASS compile，并记录 module stats；
5. 启动官方 `RobotInferenceServer`。

eval wrapper 新增 policy request latency：

```text
policy_latency_seconds: count / mean / min / max / p50 / p90 / p99
```

每个 trace step 也会写入 `policy_latency_seconds`。

## Prepare-only 验证

命令边界：

```text
scope: dit_mlp_only
max_modules: 1
denoising_steps: 1
pack_backend: triton
prewarm_indices: 115
prepare_only: true
```

结果文件：

- `toy_quantvla/results/phase8_cutlass_fp4_server_prepare_dit_mlp_1mod_prewarm.json`

结果：

| metric | value |
| --- | ---: |
| model load | 11.90s |
| patch | 1.04s |
| prewarm | 25.06s |
| prepare total | 38.11s |
| prewarm peak allocated | 5.675 GB |

这个测试证明 server wrapper 能完成真实 policy prewarm，且 1 个 DiT MLP module 的 CUTLASS compile 约 24.4s。

## Rollout Smoke

FP4 server：

```text
scope: dit_mlp_only
max_modules: 1
denoising_steps: 8
pack_backend: triton
prewarm_indices: 115
port: 5560
```

FP16 server：

```text
official Gr00tPolicy
denoising_steps: 8
port: 5561
```

Eval case：

```text
task_suite: libero_10
case_list: 6:1
task: put the white mug on the plate and put the chocolate pudding to the right of the plate
```

结果文件：

- `toy_quantvla/results/phase8_cutlass_fp4_server_prepare_dit_mlp_1mod_d8_rollout_smoke.json`
- `toy_quantvla/results/phase8_cutlass_fp4_rollout_dit1_task6_init1_latency.json`
- `toy_quantvla/results/phase8_cutlass_fp4_rollout_dit1_task6_init1.log`
- `toy_quantvla/results/phase8_fp16_rollout_task6_init1_latency_probe.json`
- `toy_quantvla/results/phase8_fp16_rollout_task6_init1_latency_probe.log`

成功率和 latency：

| config | success | steps / policy calls | mean | p50 | p90 | p99 | max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| FP16 official | 1/1 | 230 | 0.147s | 0.160s | 0.168s | 0.186s | 0.653s |
| CUTLASS FP4 DiT MLP 1 module | 1/1 | 231 | 0.160s | 0.162s | 0.176s | 0.223s | 0.291s |

FP4 server prepare/prewarm：

| metric | value |
| --- | ---: |
| model load | 11.17s |
| patch | 1.04s |
| prewarm | 25.37s |
| prepare total | 37.68s |
| prewarm peak allocated | 5.675 GB |

Patched module stats after d8 prewarm:

| module | calls | compile | activation pack mean | activation pack min | GEMM mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| `action_head.model.transformer_blocks.0.ff.net.0.proj` | 8 | 24.70s | 14.38 ms | 0.105 ms | 2.07 ms |

这里的 activation pack mean 被第一次 allocation/cache fill 拉高；同一个 module 后续 cached pack 最小值已经到约 0.105 ms。

## Full DiT MLP Prepare-only

随后跑了 `dit_mlp_only --max-modules 0`，也就是 action head 里全部 32 个 DiT MLP Linear，只做 server prepare/prewarm，不进入 simulator rollout。

命令边界：

```text
scope: dit_mlp_only
max_modules: 0
denoising_steps: 8
pack_backend: triton
prewarm_indices: 115
prepare_only: true
```

结果文件：

- `toy_quantvla/results/phase8_cutlass_fp4_server_prepare_dit_mlp_full_d8_prewarm.json`

总体结果：

| metric | value |
| --- | ---: |
| patched modules | 32 |
| model load | 11.14s |
| patch | 4.56s |
| prewarm | 778.37s |
| prepare total | 794.20s |
| post-patch current allocated | 5.185 GB |
| prewarm current allocated | 5.238 GB |
| prewarm peak allocated | 5.401 GB |
| prewarm peak reserved | 6.143 GB |

32 个 module 的聚合 stats：

| stat | count | mean | min | max | sum |
| --- | ---: | ---: | ---: | ---: | ---: |
| compile seconds | 32 | 24.12s | 23.84s | 24.60s | 771.93s |
| activation pack mean | 32 | 14.43 ms | 14.16 ms | 15.53 ms | 461.77 ms |
| activation pack min | 32 | 0.065 ms | 0.057 ms | 0.099 ms | 2.09 ms |
| GEMM mean | 32 | 1.153 ms | 1.024 ms | 1.263 ms | 36.90 ms |
| GEMM min | 32 | 0.047 ms | 0.043 ms | 0.056 ms | 1.51 ms |

所有 module 都是：

```text
compiled_m_values: [49]
calls: 8
compile count: 1
```

这个结果说明：

- d8 prewarm 成功覆盖全部 full DiT MLP modules；
- 启动预热代价约 13.24 分钟，其中约 97.2% 是 CUTLASS/CuTe compile；
- 热路径本身不慢，cached activation pack 最小值约 0.06 ms/module，GEMM 最小值约 0.05 ms/module；
- 现在最大工程债不是 FP4 arithmetic，而是 compile cache 粒度太细。

## Shared Compile Cache

随后把 `CutlassBlockscaledFP4Linear` 的 compile cache 从 per-module 扩展为 process-level per-shape cache。

cache key 包含：

```text
cutlass_root
sf_dtype
sf_vec_size
tile_shape_mnk
epi_tile
runtime M
in_features
out_features
output dtype/layout class
```

这样 full DiT MLP 32 个 Linear 在 `M=49` 下只需要编译两种 shape：

```text
49 x 1536 -> 6144
49 x 6144 -> 1536
```

新增/修改：

- `toy_quantvla/cutlass_blockscaled_fp4_linear.py`
- `toy_quantvla/phase8_cutlass_blockscaled_fp4_forward_smoke.py`
- `toy_quantvla/cutlass_fp4_inference_service.py`

新增参数：

```text
--no-share-compile-cache
```

默认开启 shared compile cache。

结果文件：

- `toy_quantvla/results/phase8_cutlass_fp4_server_prepare_dit_mlp_full_d8_prewarm_shared_compile.json`

对比：

| config | compiled kernels | shared hits | local hits | prewarm | prepare total |
| --- | ---: | ---: | ---: | ---: | ---: |
| per-module compile | 32 | 0 | 224 | 778.37s | 794.20s |
| shared compile cache | 2 | 30 | 224 | 53.48s | 69.99s |

加速：

| metric | old | shared cache | improvement |
| --- | ---: | ---: | ---: |
| prewarm | 778.37s | 53.48s | 14.55x |
| prepare total | 794.20s | 69.99s | 11.35x |
| compile sum | 771.93s | 48.75s | 15.84x |

显存基本不变：

| config | prewarm current allocated | prewarm peak allocated | prewarm peak reserved |
| --- | ---: | ---: | ---: |
| per-module compile | 5.238 GB | 5.401 GB | 6.143 GB |
| shared compile cache | 5.238 GB | 5.401 GB | 6.143 GB |

这说明 compile cache 共享解决的是启动时间，不是显存占用。

## Full DiT MLP Shared-cache Rollout

在 shared compile cache 生效后，启动 full DiT MLP FP4 server 并跑同一个 `task 6:init 1`。

结果文件：

- `toy_quantvla/results/phase8_cutlass_fp4_server_prepare_dit_mlp_full_d8_rollout_shared_compile.json`
- `toy_quantvla/results/phase8_cutlass_fp4_rollout_dit_full_task6_init1_shared_latency.json`
- `toy_quantvla/results/phase8_cutlass_fp4_rollout_dit_full_task6_init1_shared.log`
- `toy_quantvla/results/phase8_cutlass_fp4_server_dit_full_d8_shared.log`

server prepare:

| metric | value |
| --- | ---: |
| patched modules | 32 |
| compiled kernels | 2 |
| shared hits | 30 |
| local hits | 224 |
| patch | 4.49s |
| prewarm | 53.14s |
| prepare total | 68.69s |
| prewarm peak allocated | 5.401 GB |

同 case rollout latency：

| config | success | steps / policy calls | mean | p50 | p90 | p99 | max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| FP16 official | 1/1 | 230 | 0.147s | 0.160s | 0.168s | 0.186s | 0.653s |
| CUTLASS FP4 DiT MLP 1 module | 1/1 | 231 | 0.160s | 0.162s | 0.176s | 0.223s | 0.291s |
| CUTLASS FP4 DiT MLP full shared-cache | 1/1 | 420 | 0.163s | 0.160s | 0.169s | 0.196s | 0.234s |

解读：

- full DiT MLP FP4 可以跑通 simulator，并在该 case 上成功；
- 单步 latency 和 FP16 基本同量级，没有出现期望中的端到端加速；
- full DiT FP4 改变了闭环轨迹，成功但走了 420 policy calls，比 FP16 的 230 calls 明显更长；
- 因此端到端 episode 时间变长，主要不是 server 单步 latency，而是策略轨迹发生了重分配。

## Client/Server Latency Breakdown

为了定位单步 latency 没有明显变快的原因，又加了两层 timing：

1. eval client 侧拆分：

```text
preprocess_seconds
remote_get_action_seconds
postprocess_seconds
policy_total_seconds
```

2. server 侧用 `TimedPolicyWrapper` 包住 `policy.get_action`，在 server 退出时写：

```text
server get_action_seconds
```

新增/修改：

- `toy_quantvla/timing_utils.py`
- `toy_quantvla/timed_fp16_inference_service.py`
- `toy_quantvla/libero_eval_init_range.py`
- `toy_quantvla/cutlass_fp4_inference_service.py`

结果文件：

- `toy_quantvla/results/phase8_fp16_timed_task6_init1_client_latency.json`
- `toy_quantvla/results/phase8_fp16_timed_server_task6_init1_server_latency.json`
- `toy_quantvla/results/phase8_cutlass_fp4_timed_dit_full_task6_init1_client_latency.json`
- `toy_quantvla/results/phase8_cutlass_fp4_timed_server_dit_full_task6_init1_server_latency.json`

同 case timing：

| config | calls | client total mean | client remote mean | server get_action mean | client-server overhead | preprocess mean | postprocess mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| FP16 official | 243 | 0.1561s | 0.1559s | 0.1511s | 4.76 ms | 0.118 ms | 0.115 ms |
| FP4 DiT full shared-cache | 420 | 0.1624s | 0.1621s | 0.1572s | 4.88 ms | 0.118 ms | 0.114 ms |

p50 / p90:

| config | client total p50 | client total p90 | server get_action p50 | server get_action p90 |
| --- | ---: | ---: | ---: | ---: |
| FP16 official | 0.1612s | 0.1687s | 0.1559s | 0.1639s |
| FP4 DiT full shared-cache | 0.1613s | 0.1691s | 0.1562s | 0.1640s |

解读：

- eval client 的 image/state preprocessing 和 action postprocess 都是 `~0.1 ms`，可以忽略；
- ZMQ/serialization/client-server overhead 稳定在 `~4.8 ms`，不是 FP4 比 FP16 慢的主要原因；
- server-side `policy.get_action` 才是主体；
- FP4 DiT full 的 server-side mean `0.1572s`，FP16 mean `0.1511s`，FP4 慢约 `6.1 ms/step`；
- p50/p90 几乎持平，说明均值差异可能来自少量尾部波动和 wrapper 开销；
- 当前瓶颈更可能在 `CutlassBlockscaledFP4Linear.forward` 的 Python/CuTe tensor/output wrapper 构造、per-layer pack bookkeeping、以及非量化路径占比，而不是 client eval wrapper。

## Module-Level Runtime Breakdown

为了进一步确认 `policy.get_action` 里的时间到底花在哪里，给 `CutlassBlockscaledFP4Linear.forward` 加了 profile-only 阶段计时：

```text
input_prepare_seconds
activation_pack_seconds
output_prepare_seconds
compile_lookup_seconds
gemm_seconds
finalize_seconds
forward_seconds
```

注意：profile 模式会在阶段边界增加 CUDA synchronize，所以 profile rollout 的绝对 latency 会比普通 rollout 更慢；这里主要看比例和归因，不直接作为部署速度。

新增/修改：

- `toy_quantvla/cutlass_blockscaled_fp4_linear.py`
- `toy_quantvla/phase8_cutlass_blockscaled_fp4_forward_smoke.py`
- `toy_quantvla/cutlass_fp4_inference_service.py`
- `toy_quantvla/timing_utils.py`

### Offline Clean Profile

命令边界：

```text
scope=dit_mlp_only
max_modules=0
denoising_steps=8
pack_backend=triton
profile_modules=true
num_observations=1
indices=115
```

结果文件：

- `toy_quantvla/results/phase8_cutlass_blockscaled_fp4_forward_profile_dit_full_d8_1obs_v2.json`

top-level timing：

| item | seconds |
| --- | ---: |
| FP16 teacher get_action | 0.4765 |
| patch | 4.2922 |
| FP4 cold student | 54.2758 |
| FP4 warm student | 0.1637 |
| warm student / teacher | 0.3434 |

warm module totals over 32 DiT MLP modules:

| phase | total | count | mean |
| --- | ---: | ---: | ---: |
| forward | 0.0636s | 256 | 0.248 ms |
| input prepare | 0.0041s | 256 | 0.016 ms |
| activation pack | 0.0167s | 256 | 0.065 ms |
| output prepare | 0.0164s | 256 | 0.064 ms |
| compile lookup | 0.0015s | 256 | 0.006 ms |
| GEMM | 0.0128s | 256 | 0.050 ms |
| finalize | 0.0099s | 256 | 0.039 ms |

phase share inside FP4 module forward:

| phase | share |
| --- | ---: |
| activation pack | 26.22% |
| output prepare | 25.77% |
| GEMM | 20.05% |
| finalize | 15.57% |
| input prepare | 6.52% |
| compile lookup | 2.42% |

这说明 steady-state 下，真正的 CUTLASS GEMM 不是唯一大头；activation pack 和 output tensor/wrapper preparation 合起来约占 FP4 module forward 的一半。

### Profiled Rollout

同一个真实 simulator case：

```text
task=6
init=1
success=true
policy calls=420
```

结果文件：

- `toy_quantvla/results/phase8_cutlass_fp4_profiled_dit_full_task6_init1_client_latency.json`
- `toy_quantvla/results/phase8_cutlass_fp4_profiled_server_dit_full_task6_init1_latency.json`
- `toy_quantvla/results/phase8_cutlass_fp4_profiled_server_prepare_dit_full_task6_init1.json`

profiled server/client timing：

| item | value |
| --- | ---: |
| client policy total mean | 0.1818s |
| client remote mean | 0.1816s |
| server get_action mean | 0.1757s |
| server get_action p50 | 0.1611s |
| server get_action p90 | 0.2341s |

rollout module totals over 420 requests:

| phase | total | count | mean |
| --- | ---: | ---: | ---: |
| FP4 module forward | 27.1072s | 107520 | 0.252 ms |
| input prepare | 2.0742s | 107520 | 0.019 ms |
| activation pack | 6.9338s | 107520 | 0.064 ms |
| output prepare | 6.9262s | 107520 | 0.064 ms |
| compile lookup | 0.6675s | 107520 | 0.006 ms |
| GEMM | 5.1028s | 107520 | 0.047 ms |
| finalize | 4.4661s | 107520 | 0.042 ms |

server-side `get_action` total over the rollout is about `73.78s`; profiled FP4 DiT MLP modules account for `27.11s`, or `36.74%`.

phase share:

| phase | share of FP4 module | share of server get_action |
| --- | ---: | ---: |
| activation pack | 25.58% | 9.40% |
| output prepare | 25.55% | 9.39% |
| GEMM | 18.82% | 6.92% |
| finalize | 16.48% | 6.05% |
| input prepare | 7.65% | 2.81% |
| compile lookup | 2.46% | 0.90% |

关键结论：

- FP4 DiT MLP 路径在真实 rollout 中稳定复现了 offline warm profile，每个 `get_action` 约 `64.5 ms`；
- 但它只覆盖 profiled server-side `get_action` 的约 `36.7%`，剩余约 `63.3%` 来自未量化路径、diffusion/control flow、attention、projector、vision/LLM/request handling 等；
- 在 FP4 module 内部，GEMM 只占约 `18.8%`，pack 与 output wrapper preparation 合计约 `51.1%`；
- 因此只优化 CUTLASS GEMM 本身，理论收益上限很低；要看到 rollout 级加速，必须同时压低 activation pack / output wrapper 开销，或者扩大可加速覆盖面。

### Output Tensor Cache Ablation

基于上面的 profile，先尝试了一个低风险工程优化：按 module/device/runtime `M` 缓存 CUTLASS output tensor/storage，减少每次 forward 里的 `cute_tensor_like` 和 wrapper preparation。

实现位置：

- `toy_quantvla/cutlass_blockscaled_fp4_linear.py`
- `toy_quantvla/phase8_cutlass_blockscaled_fp4_forward_smoke.py`
- `toy_quantvla/cutlass_fp4_inference_service.py`

注意：这个 cache 现在是显式开关 `--cache-output-tensor`，不是默认行为。原因是它会复用 output storage，虽然当前单 case rollout 成功，但跨更多任务前不应该默认改变 storage lifetime。

offline profile 对比：

| config | warm get_action | module forward total | activation pack | output prepare | GEMM | finalize |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| no output cache | 0.1637s | 0.0636s | 0.0167s | 0.0164s | 0.0128s | 0.0099s |
| output cache | 0.1357s | 0.0422s | 0.0147s | 0.0019s | 0.0094s | 0.0093s |

结果文件：

- `toy_quantvla/results/phase8_cutlass_blockscaled_fp4_forward_profile_dit_full_d8_1obs_v3_cache_output.json`

offline profile 中，output cache 把 `output_prepare` 从 `16.4ms/get_action` 降到 `1.9ms/get_action`，profiled warm get_action 从 `0.1637s` 降到 `0.1357s`。

但真实非 profile rollout 没有同步收益：

| config | success | calls | client mean | client p50 | client p90 | server mean | server p50 | server p90 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| FP4 no output cache | 1/1 | 420 | 0.1624s | 0.1613s | 0.1691s | 0.1572s | 0.1562s | 0.1640s |
| FP4 output cache | 1/1 | 420 | 0.1635s | 0.1614s | 0.1724s | 0.1579s | 0.1561s | 0.1664s |

结果文件：

- `toy_quantvla/results/phase8_cutlass_fp4_output_cache_dit_full_task6_init1_client_latency.json`
- `toy_quantvla/results/phase8_cutlass_fp4_server_dit_full_d8_output_cache_task6_init1_latency.json`
- `toy_quantvla/results/phase8_cutlass_fp4_server_prepare_dit_full_d8_output_cache_task6_init1.json`

解读：

- output cache 确实降低了 profile 模式下可见的 `output_prepare`；
- 但非 profile rollout 的 server mean 基本持平，说明这个开销在真实异步执行路径里不是当前主瓶颈，或者被其他未覆盖路径和 tail latency 淹没；
- 因此 output cache 暂时保留为实验开关，不作为默认优化；
- 下一步不应继续只盯 output wrapper，而应该补 FP16 matched module hooks，直接估算原始 DiT MLP 在 FP16 server 里的占比，并用更低扰动的 profiler 确认真实 CUDA timeline。

### FP16 Matched DiT MLP Profile

为了判断 FP4 module 本身是否真的比原始 FP16 DiT MLP 快，又给 FP16 official server 加了 matched module hook。这个 hook 使用和 FP4 一样的 `dit_mlp_only` scope，包住原始 `nn.Linear`，只记录 forward latency，不改变权重和计算。

新增：

- `toy_quantvla/fp16_linear_profiler.py`
- `toy_quantvla/timed_fp16_inference_service.py`

命令边界：

```text
server=timed_fp16_inference_service.py
denoising_steps=8
profile_linear_modules=true
profile_scope=dit_mlp_only
task=6
init=1
```

结果文件：

- `toy_quantvla/results/phase8_fp16_profiled_dit_mlp_task6_init1_client_latency.json`
- `toy_quantvla/results/phase8_fp16_profiled_server_dit_mlp_task6_init1_latency.json`
- `toy_quantvla/results/phase8_fp16_profiled_server_prepare_dit_mlp_task6_init1.json`

FP16 DiT MLP profile：

| item | value |
| --- | ---: |
| success | 1/1 |
| policy calls | 243 |
| server get_action mean | 0.1586s |
| server get_action p50 | 0.1573s |
| server get_action p90 | 0.1672s |
| DiT MLP Linear forward total | 3.2362s |
| DiT MLP Linear calls | 62208 |
| DiT MLP per get_action | 13.32 ms |
| DiT MLP share of server get_action | 8.40% |

对比 FP4 profiled rollout：

| config | policy calls | server mean | DiT MLP per get_action | DiT MLP share |
| --- | ---: | ---: | ---: | ---: |
| FP16 matched profile | 243 | 0.1586s | 13.32 ms | 8.40% |
| FP4 profiled | 420 | 0.1757s | 64.54 ms | 36.74% |

这个对比需要小心解释：FP4 profile 在每个阶段有更多 CUDA synchronize，因此不能把 `64.54 / 13.32` 直接当成真实部署中的严格倍率。但它足以说明两件事：

- 在 FP16 official server 里，原始 DiT MLP Linear 只占约 `8.4%` 的 `get_action`；
- 即使把 DiT MLP Linear 完美加速到 0，`dit_mlp_only` 对端到端 `get_action` 的理论上限也只有约 8-9%；
- 这解释了为什么 full DiT MLP FP4 rollout 很难显著超过 FP16：覆盖面太小，且当前 FP4 path 自身还有 pack/wrapper 开销。

### No-Sync Runtime Ablation

检查 FP4 wrapper 时发现早期实现里即使非 profile，也会在 activation pack 后和 GEMM 后执行 `torch.cuda.synchronize()`。这会打断 CUDA 异步执行，因此做了一个修复：这些同步只保留在 `--profile-modules` 模式，普通 server path 不再每层同步。

结果文件：

- `toy_quantvla/results/phase8_cutlass_fp4_no_sync_dit_full_task6_init1_client_latency.json`
- `toy_quantvla/results/phase8_cutlass_fp4_server_dit_full_d8_no_sync_task6_init1_latency.json`
- `toy_quantvla/results/phase8_cutlass_fp4_server_prepare_dit_full_d8_no_sync_task6_init1.json`

非 profile rollout 对比：

| config | success | calls | client mean | client p50 | client p90 | server mean | server p50 | server p90 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| FP16 official | 1/1 | 243 | 0.1561s | 0.1612s | 0.1687s | 0.1511s | 0.1559s | 0.1639s |
| FP4 original | 1/1 | 420 | 0.1624s | 0.1613s | 0.1691s | 0.1572s | 0.1562s | 0.1640s |
| FP4 no-sync | 1/1 | 420 | 0.1656s | 0.1614s | 0.1795s | 0.1606s | 0.1563s | 0.1742s |
| FP4 output cache | 1/1 | 420 | 0.1635s | 0.1614s | 0.1724s | 0.1579s | 0.1561s | 0.1664s |

结论：

- 去掉非 profile 的显式 synchronize 没有改善这个 rollout；
- 可能原因是 Triton pack / CUTLASS bridge 内部仍有同步或 CPU-side 阻塞，也可能是该 case 的 tail latency 波动淹没了小收益；
- 这个修复仍然保留，因为普通 runtime path 不应该为了 timing 强制同步，但它不是当前端到端加速的突破口。

### Larger Scope Offline Smoke

由于 FP16 matched profile 显示 `dit_mlp_only` 覆盖面太小，又测试了更大的 scope。

offline command boundary:

```text
denoising_steps=8
pack_backend=triton
num_observations=1
indices=115
```

结果文件：

- `toy_quantvla/results/phase8_cutlass_blockscaled_fp4_forward_smoke_llm_mlp_full_d8_tritonpack_shared.json`
- `toy_quantvla/results/phase8_cutlass_blockscaled_fp4_forward_smoke_llm_mlp_dit_mlp_full_d8_tritonpack_shared.json`

offline 对比：

| scope | patched modules | teacher | warm student | warm/teacher | relative RMSE | cosine | max abs diff | patch | cold |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `dit_mlp_only` | 32 | 0.4765s | 0.1637s | 0.3434 | 0.00354 | 0.9999937 | 0.00504 | - | - |
| `llm_mlp_only` | 36 | 0.5533s | 0.0927s | 0.1674 | 0.00389 | 0.9999925 | 0.00616 | 4.84s | 55.05s |
| `llm_mlp_dit_mlp` | 68 | 0.4905s | 0.1509s | 0.3076 | 0.00496 | 0.9999877 | 0.00732 | 8.59s | 106.26s |

offline 观察：

- `llm_mlp_only` 是目前 offline 速度最好的 scope；
- `llm_mlp_dit_mlp` 覆盖更大，但 warm latency 反而比纯 `llm_mlp_only` 慢；
- 两者单步 action error 都仍然很小，说明 offline action-vector 误差不足以判断闭环成功率。

### LLM MLP Rollout Smoke

基于 offline 结果，先跑 `llm_mlp_only` 的真实 rollout smoke。

结果文件：

- `toy_quantvla/results/phase8_cutlass_fp4_llm_mlp_full_task6_init1_client_latency.json`
- `toy_quantvla/results/phase8_cutlass_fp4_server_llm_mlp_full_d8_task6_init1_latency.json`
- `toy_quantvla/results/phase8_cutlass_fp4_server_prepare_llm_mlp_full_d8_task6_init1.json`

case:

```text
task=6
init=1
scope=llm_mlp_only
denoising_steps=8
```

rollout result:

| config | success | calls | client mean | client p50 | client p90 | server mean | server p50 | server p90 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| FP16 official | 1/1 | 243 | 0.1561s | 0.1612s | 0.1687s | 0.1511s | 0.1559s | 0.1639s |
| FP4 `dit_mlp_only` | 1/1 | 420 | 0.1624s | 0.1613s | 0.1691s | 0.1572s | 0.1562s | 0.1640s |
| FP4 `llm_mlp_only` | 0/1 | 991 | 0.2096s | 0.1610s | 0.1712s | 0.2050s | 0.1560s | 0.1674s |

解释：

- `llm_mlp_only` 的 p50/p90 latency 仍接近 FP16/DiT-only；
- mean 被一个约 `54s` 的 server/client outlier 拉高，发生在失败长 horizon/terminated path 附近；
- 主要问题是行为失败：991 calls 跑到 horizon，最后 `executing action in terminated episode`；
- 这说明 LLM MLP 的小 offline action error 可以累积成闭环病态，不能因为 offline warm latency 好就直接推进 rollout 主线。

### LLM MLP Subset FP4 Sweep

按照“以真实 FP4 结果为导向”的原则，继续把 LLM MLP 拆成更小的 FP4 子集。这里不使用 fake quant 作为结论，只用 CUTLASS FP4 offline/server 结果。

offline boundary:

```text
scope=llm_mlp_only
denoising_steps=8
pack_backend=triton
num_observations=1
indices=115
patched_modules=12 each subset
```

结果文件：

- `toy_quantvla/results/phase8_cutlass_blockscaled_fp4_forward_smoke_llm_mlp_gate_proj_d8_tritonpack_shared.json`
- `toy_quantvla/results/phase8_cutlass_blockscaled_fp4_forward_smoke_llm_mlp_up_proj_d8_tritonpack_shared.json`
- `toy_quantvla/results/phase8_cutlass_blockscaled_fp4_forward_smoke_llm_mlp_down_proj_d8_tritonpack_shared.json`
- `toy_quantvla/results/phase8_cutlass_blockscaled_fp4_forward_smoke_llm_mlp_layers_early_d8_tritonpack_shared.json`
- `toy_quantvla/results/phase8_cutlass_blockscaled_fp4_forward_smoke_llm_mlp_layers_mid_d8_tritonpack_shared.json`
- `toy_quantvla/results/phase8_cutlass_blockscaled_fp4_forward_smoke_llm_mlp_layers_late_d8_tritonpack_shared.json`

offline subset results:

| subset | warm | warm/teacher | relative RMSE | cosine | max abs diff | patch+cold |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `gate_proj` | 0.0927s | 0.1879 | 0.00201 | 0.9999980 | 0.00366 | 27.94s |
| `up_proj` | 0.0934s | 0.1387 | 0.00178 | 0.9999985 | 0.00298 | 28.05s |
| `down_proj` | 0.0943s | 0.1792 | 0.00262 | 0.9999966 | 0.00431 | 28.11s |
| `layers_0_3` | 0.0909s | 0.1925 | 0.00371 | 0.9999933 | 0.00641 | 52.06s |
| `layers_4_7` | 0.0912s | 0.1455 | 0.00186 | 0.9999984 | 0.00343 | 52.61s |
| `layers_8_11` | 0.1606s | 0.3328 | 0.00183 | 0.9999984 | 0.00240 | 56.40s |

offline observation:

- projection-wise, `up_proj` has the lowest action error among `gate/up/down`;
- layer-wise, middle and late layers have lower action error than early layers, but late layers show worse warm latency in this single-observation run;
- `up_proj` is the cleanest first rollout candidate: low error, small scope, lower prepare cost than layer bins.

### LLM MLP `up_proj` Rollout Smoke

Based on the subset sweep, `up_proj` only was tested in the same real simulator case.

结果文件：

- `toy_quantvla/results/phase8_cutlass_fp4_llm_mlp_up_proj_task6_init1_client_latency.json`
- `toy_quantvla/results/phase8_cutlass_fp4_server_llm_mlp_up_proj_d8_task6_init1_latency.json`
- `toy_quantvla/results/phase8_cutlass_fp4_server_prepare_llm_mlp_up_proj_d8_task6_init1.json`

case:

```text
task=6
init=1
scope=llm_mlp_only
name_contains=up_proj
denoising_steps=8
```

rollout comparison:

| config | success | calls | client mean | client p50 | client p90 | server mean | server p50 | server p90 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| FP16 official | 1/1 | 243 | 0.1561s | 0.1612s | 0.1687s | 0.1511s | 0.1559s | 0.1639s |
| FP4 `llm_mlp_only` full | 0/1 | 991 | 0.2096s | 0.1610s | 0.1712s | 0.2050s | 0.1560s | 0.1674s |
| FP4 `llm_mlp_up_proj` | 1/1 | 226 | 0.2769s | 0.1612s | 0.1687s | 0.2716s | 0.1558s | 0.1633s |

Interpretation:

- `up_proj` only succeeds on the same case where full `llm_mlp_only` fails;
- calls drop to `226`, compared with FP16's `243` and DiT-only's `420`;
- p50/p90 latency is essentially FP16-like;
- mean latency is dominated by one large `~26.6s` outlier, so mean should not be used as the main speed signal for this run;
- this is currently the best FP4 rollout candidate: behavior survived the smoke, scope is small, and offline error is lowest among projection slices.

Remaining concern:

- the large outlier must be reproduced or ruled out across more init states;
- success on one case is not enough to call `up_proj` safe;
- next test should be matched multi-init, not broader scope.

### LLM MLP `up_proj` Multi-init Smoke

To separate one-init luck from a real candidate, `up_proj` only was rerun on the same task with three init states.

结果文件：

- `toy_quantvla/results/phase8_cutlass_fp4_llm_mlp_up_proj_task6_init0_2_client_latency.json`
- `toy_quantvla/results/phase8_cutlass_fp4_server_llm_mlp_up_proj_d8_task6_init0_2_latency.json`
- `toy_quantvla/results/phase8_cutlass_fp4_server_prepare_llm_mlp_up_proj_d8_task6_init0_2.json`
- `toy_quantvla/results/phase8_cutlass_fp4_llm_mlp_up_proj_task6_init0_2_client.log`
- `toy_quantvla/results/phase8_cutlass_fp4_llm_mlp_up_proj_task6_init0_2_eval.log`
- `toy_quantvla/results/phase8_cutlass_fp4_server_llm_mlp_up_proj_d8_task6_init0_2.log`

case:

```text
task=6
init=0,1,2
scope=llm_mlp_only
name_contains=up_proj
denoising_steps=8
patched_modules=12
pack_backend=triton
```

client-side rollout result:

| init | success | calls | client mean | client p50 | client p90 | client p99 | client max |
| ---: | :---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | false | 991 | 0.1868s | 0.1625s | 0.1691s | 0.1956s | 26.2857s |
| 1 | true | 236 | 0.1600s | 0.1615s | 0.1674s | 0.1746s | 0.1752s |
| 2 | true | 220 | 0.1589s | 0.1620s | 0.1693s | 0.1812s | 0.1901s |
| overall | 2/3 | 1447 | 0.1782s | 0.1623s | 0.1691s | 0.1871s | 26.2857s |

server-side `get_action` overall:

| count | mean | p50 | p90 | p99 | max |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1447 | 0.1732s | 0.1572s | 0.1640s | 0.1816s | 26.2788s |

prepare/profile note:

- model load: `11.29s`;
- patch: `2.30s`;
- prewarm total: `27.12s`;
- prewarm memory: `5.413 GB` allocated, `5.937 GB` reserved.

Interpretation:

- `init1` and `init2` are clean successes, with no long latency tail and with calls close to or lower than the FP16 `task6:init1` smoke;
- `init0` fails at horizon with `991` calls and also contains the same kind of `~26s` long-tail request;
- the multi-init result is therefore promising but not sufficient alone: `up_proj` is much safer than full `llm_mlp_only`, but task6 init0 needs an exact FP16 timed baseline before attributing the failure to FP4;
- this run strengthens `up_proj` as the first real FP4 candidate; the exact FP16 control is reported below.

### FP16 Exact Matched Baseline

The exact matched FP16 timed baseline was then run on the same task and init states. This is the key control for interpreting the `up_proj` multi-init smoke.

结果文件：

- `toy_quantvla/results/phase8_fp16_timed_task6_init0_2_client_latency.json`
- `toy_quantvla/results/phase8_fp16_timed_server_task6_init0_2_server_latency.json`
- `toy_quantvla/results/phase8_fp16_timed_server_prepare_task6_init0_2.json`
- `toy_quantvla/results/phase8_fp16_timed_task6_init0_2_client.log`
- `toy_quantvla/results/phase8_fp16_timed_task6_init0_2_eval.log`
- `toy_quantvla/results/phase8_fp16_timed_server_task6_init0_2.log`

matched result:

| init | FP16 success | FP16 calls | FP16 p50 | FP16 p90 | FP16 max | FP4 `up_proj` success | FP4 calls | FP4 p50 | FP4 p90 | FP4 max |
| ---: | :---: | ---: | ---: | ---: | ---: | :---: | ---: | ---: | ---: | ---: |
| 0 | false | 991 | 0.1616s | 0.1707s | 0.8095s | false | 991 | 0.1625s | 0.1691s | 26.2857s |
| 1 | true | 236 | 0.1618s | 0.1697s | 0.5874s | true | 236 | 0.1615s | 0.1674s | 0.1752s |
| 2 | true | 247 | 0.1603s | 0.1673s | 0.1962s | true | 220 | 0.1620s | 0.1693s | 0.1901s |
| overall | 2/3 | 1474 | 0.1612s | 0.1701s | 0.8095s | 2/3 | 1447 | 0.1623s | 0.1691s | 26.2857s |

server-side `get_action` overall:

| config | count | mean | p50 | p90 | p99 | max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| FP16 timed | 1474 | 0.1548s | 0.1560s | 0.1651s | 0.1909s | 0.8039s |
| FP4 `llm_mlp_up_proj` | 1447 | 0.1732s | 0.1572s | 0.1640s | 0.1816s | 26.2788s |

Interpretation:

- success/failure pattern is identical: both FP16 and FP4 `up_proj` fail at `init0` and succeed at `init1/init2`;
- `init0` should no longer be treated as an FP4-specific behavior regression in this small matched set;
- `init1` has exactly matched calls (`236` vs `236`), while `init2` is shorter under FP4 (`220` vs `247`), but this is too small to call a systematic improvement;
- p50/p90 latency is essentially matched, so there is no real speedup yet;
- FP4 still has a single large `~26s` latency tail, absent from this FP16 baseline, so the next step is to instrument whether it is a real runtime tail or a compile-cache miss.

### FP4 Request Trace and Shape Warmup

To locate the `~26s` FP4 tail, the server was extended with optional per-request JSONL tracing:

```text
--server-request-trace-jsonl
--server-request-trace-min-seconds
--server-request-trace-module-deltas
--server-request-trace-cuda-sync
```

For FP4 modules, the trace can record request-level module call deltas, compile-cache hit deltas, and compile events. The default server behavior is unchanged when these flags are not passed.

Trace rerun result files:

- `toy_quantvla/results/phase8_cutlass_fp4_llm_mlp_up_proj_task6_init0_2_trace_client_latency.json`
- `toy_quantvla/results/phase8_cutlass_fp4_server_llm_mlp_up_proj_d8_task6_init0_2_trace_latency.json`
- `toy_quantvla/results/phase8_cutlass_fp4_server_llm_mlp_up_proj_d8_task6_init0_2_request_trace.jsonl`
- `toy_quantvla/results/phase8_cutlass_fp4_server_prepare_llm_mlp_up_proj_d8_task6_init0_2_trace.json`
- `toy_quantvla/results/phase8_cutlass_fp4_llm_mlp_up_proj_task6_init0_2_trace_client.log`
- `toy_quantvla/results/phase8_cutlass_fp4_llm_mlp_up_proj_task6_init0_2_trace_eval.log`
- `toy_quantvla/results/phase8_cutlass_fp4_server_llm_mlp_up_proj_d8_task6_init0_2_trace.log`

The exact trace rerun preserved the behavior pattern:

| init | success | calls | client mean | client p50 | client p90 | client max |
| ---: | :---: | ---: | ---: | ---: | ---: | ---: |
| 0 | false | 991 | 0.1881s | 0.1624s | 0.1706s | 26.9273s |
| 1 | true | 236 | 0.1603s | 0.1622s | 0.1697s | 0.1822s |
| 2 | true | 220 | 0.1616s | 0.1618s | 0.1725s | 0.1980s |
| overall | 2/3 | 1447 | 0.1795s | 0.1623s | 0.1705s | 26.9273s |

Trace diagnosis:

| item | value |
| --- | ---: |
| request trace rows | 1447 |
| rows with FP4 compile delta | 1 |
| long-tail request index | 1 |
| long-tail server time | 26.9210s |
| long-tail FP4 module calls | 12 |
| long-tail compile count delta | 1 |
| long-tail compile seconds delta | 25.2818s |
| module with new compile | `backbone.eagle_model.language_model.model.layers.0.mlp.up_proj` |
| compiled M values after event | `[565, 566]` |

Interpretation:

- the `~26s` outlier is not random server jitter and not steady-state GEMM latency;
- it is a CUTLASS compile-cache miss on the first rollout request;
- the original prewarm compiled `M=566`, while the first real task6 request required `M=565`;
- because shared compile cache is enabled, only the first module pays the compile cost; the other `up_proj` modules hit the shared cache for the new shape.

To move this cost out of rollout, the FP4 server now accepts:

```text
--prewarm-task-description "..."
```

This reuses a real prewarm observation but overrides the task text, allowing the server to precompile LLM sequence lengths induced by the actual eval task description.

Warmdesc prepare-only result:

- `toy_quantvla/results/phase8_cutlass_fp4_server_prepare_llm_mlp_up_proj_d8_task6_prewarm_taskdesc.json`

| item | value |
| --- | ---: |
| prewarm observations | 2 |
| prewarm seconds | 52.0920s |
| prewarm total seconds | 52.1997s |
| prepare seconds | 65.5828s |
| compiled M values | `[565, 566]` |

Warmdesc rollout smoke:

- `toy_quantvla/results/phase8_cutlass_fp4_llm_mlp_up_proj_task6_init1_warmdesc_client_latency.json`
- `toy_quantvla/results/phase8_cutlass_fp4_server_llm_mlp_up_proj_d8_task6_init1_warmdesc_latency.json`
- `toy_quantvla/results/phase8_cutlass_fp4_server_llm_mlp_up_proj_d8_task6_init1_warmdesc_request_trace.jsonl`
- `toy_quantvla/results/phase8_cutlass_fp4_server_prepare_llm_mlp_up_proj_d8_task6_init1_warmdesc.json`
- `toy_quantvla/results/phase8_cutlass_fp4_llm_mlp_up_proj_task6_init1_warmdesc_client.log`
- `toy_quantvla/results/phase8_cutlass_fp4_llm_mlp_up_proj_task6_init1_warmdesc_eval.log`
- `toy_quantvla/results/phase8_cutlass_fp4_server_llm_mlp_up_proj_d8_task6_init1_warmdesc.log`

| item | value |
| --- | ---: |
| success | 1/1 |
| calls | 226 |
| client mean | 0.1515s |
| client p50 | 0.1606s |
| client p90 | 0.1729s |
| client max | 0.5197s |
| server mean | 0.1466s |
| server p50 | 0.1553s |
| server p90 | 0.1677s |
| server max | 0.5166s |
| request trace rows | 226 |
| rows with FP4 compile delta | 0 |

Interpretation:

- shape-aware prewarm removes the rollout-time `~26s` cold compile;
- the cost is not eliminated, but moved into startup/prewarm where it belongs;
- steady-state FP4 `up_proj` remains close to FP16 in p50/p90 and still does not demonstrate a real speedup;
- for deployment, the server needs a warmup contract that covers expected LLM sequence lengths for the task set before accepting eval/production requests.

## 解读

可以确认：

1. cached Triton/CUTLASS FP4 已经能接入 server 并跑通 LIBERO simulator。
2. prewarm 能把 episode 内首次 CUTLASS compile 移到 server 启动阶段。
3. `libero_10 task 6:init 1` 在 FP4 1-module 和 FP16 下都成功。
4. eval wrapper 现在能记录端到端 policy request latency。
5. full DiT MLP 32 modules 的 d8 server prewarm 可以跑通，但启动预热约 13.24 分钟。
6. shared compile cache 把 full DiT MLP d8 prepare total 从 794.20s 降到 69.99s。
7. full DiT MLP shared-cache rollout 在 `task 6:init 1` 上成功。
8. latency breakdown 证明 client preprocess/postprocess 不是瓶颈，主要时间在 server-side `policy.get_action`。
9. module-level profiling 证明当前 FP4 DiT MLP 只覆盖 profiled server-side `get_action` 的约 36.7%，且 module 内部 pack/output wrapper 比 GEMM 更值得优化。
10. output tensor cache 能改善 profiled offline path，但没有改善非 profile rollout latency，因此不是当前真实端到端瓶颈。
11. FP16 matched module profile 显示原始 DiT MLP Linear 只占 FP16 server-side `get_action` 的约 8.4%，所以 `dit_mlp_only` 的端到端加速上限天然很低。
12. 去掉非 profile 每层同步没有改善 rollout latency，但作为正确 runtime hygiene 应保留。
13. `llm_mlp_only` offline 速度最好，但真实 rollout 在 `task6:init1` 失败，说明更大 scope 的闭环风险明显更高。
14. LLM MLP `up_proj` only 是目前最好的 FP4 子集候选：offline error 最低，真实 rollout 成功，并且 single-init calls 降到 226。
15. `up_proj` multi-init smoke 在 task6:init0-2 上达到 2/3，init1 和 init2 都是干净成功，说明它不是纯 one-init 偶然结果。
16. exact matched FP16 baseline 在 task6:init0-2 上同样是 2/3，且失败点同为 init0；因此 task6:init0 的失败不能归因于 FP4 `up_proj`。
17. FP4 `up_proj` 在这组三个 init 上行为层面基本追平 FP16 baseline，但性能层面还没有体现加速。
18. request trace 证明 `~26s` FP4 长尾来自 first-request CUTLASS compile-cache miss，不是稳态 kernel latency。
19. `--prewarm-task-description` 可以提前覆盖 task-induced LLM sequence length；task6 warmdesc smoke 中 compile delta rows 降到 0，rollout max 降到约 0.52s。

还不能确认：

1. 1-module FP4 不会带来端到端加速，当前 latency 与 FP16 同量级且均值略慢。
2. full DiT MLP FP4 也没有在该 case 上显示单步 latency 优势，mean latency 为 0.163s，FP16 为 0.147s。
3. 当前显存峰值仍接近 FP16，full DiT MLP 的 prewarm current allocated 约 5.238 GB。
4. 需要确认更多 task/init 的成功率是否保持，以及轨迹变长是否普遍存在。
5. 需要用低扰动 profiler 确认非 profile server 的 CUDA timeline，因为 Python-side synchronize profile 会放大部分阶段开销。
6. 如果扩大 scope 到 attention/projector，需要重新做成功率风险评估，不能只按速度推进。
7. `llm_mlp_only` 的失败是否是单 init 脆弱性，还是该 scope 普遍破坏语言/动作条件，需要更多 matched init 才能下结论。
8. `up_proj` 的 task6:init2 calls 变少是否代表量化扰动改善轨迹，还只是 simulator 随机性或单点偶然，需要更大 matched set。
9. warmdesc 目前只验证了 task6:init1；更大 matched set 需要对所有 task descriptions 做 shape-aware prewarm。

## 下一步

建议继续按两级推进：

1. 为 exact matched case set 收集 task descriptions，并在 FP4 server 启动时做 shape-aware prewarm；
2. 扩展 `llm_mlp_up_proj` 到更多 exact matched cases，例如 task6:init3-4 或之前权重挑出的中等难度任务；
3. 同时保留 FP16 timed baseline，对每个 case 比 success/calls/p50/p90/max，不再只看单点成功率；
4. 用低扰动 profiler 或 CUDA events 复核 activation pack / GEMM / non-quantized path 的真实 timeline；
5. output tensor cache 仅作为实验开关保留，更多 task 验证前不默认启用；
6. 暂缓 full `llm_mlp_only` 和 `llm_mlp_dit_mlp` rollout 主线；offline 虽然可行，但行为风险已经暴露。
