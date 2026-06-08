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

还不能确认：

1. 1-module FP4 不会带来端到端加速，当前 latency 与 FP16 同量级且均值略慢。
2. full DiT MLP FP4 也没有在该 case 上显示单步 latency 优势，mean latency 为 0.163s，FP16 为 0.147s。
3. 当前显存峰值仍接近 FP16，full DiT MLP 的 prewarm current allocated 约 5.238 GB。
4. 需要确认更多 task/init 的成功率是否保持，以及轨迹变长是否普遍存在。
5. 需要测 FP16 DiT MLP 的同口径 module-level runtime，才能直接判断 FP4 module 本身相对 FP16 module 是否更快。

## 下一步

建议继续按两级推进：

1. 跑 3-5 个 matched short cases，确认 full DiT FP4 的 trajectory-length 变化是否普遍；
2. 优先优化 `CutlassBlockscaledFP4Linear` 的 activation pack 和 output tensor/wrapper preparation；
3. 增加 FP16 matched module-level hooks，估算原始 DiT MLP 在 FP16 server 中的真实占比；
4. 如果 FP4 module 优化后仍只覆盖三分之一左右的 server time，再考虑扩大到 attention/projector 或更大 scope；
5. 暂缓扩大到 `llm_mlp_dit_mlp`，先把 full DiT MLP 的速度账算清楚。
