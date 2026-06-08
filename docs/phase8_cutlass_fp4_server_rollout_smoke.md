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

## 解读

可以确认：

1. cached Triton/CUTLASS FP4 已经能接入 server 并跑通 LIBERO simulator。
2. prewarm 能把 episode 内首次 CUTLASS compile 移到 server 启动阶段。
3. `libero_10 task 6:init 1` 在 FP4 1-module 和 FP16 下都成功。
4. eval wrapper 现在能记录端到端 policy request latency。

还不能确认：

1. 1-module FP4 不会带来端到端加速，当前 latency 与 FP16 同量级且均值略慢。
2. 当前显存峰值仍接近 FP16，因为只替换了 1 个 Linear。
3. 大 scope rollout 仍会面对长 prewarm：按 offline full DiT MLP 32 modules 估计，cold compile 可到 13 分钟量级。
4. 需要确认 full DiT MLP / `llm_mlp_dit_mlp` 的成功率是否保持，以及 server latency 是否低于 FP16。

## 下一步

建议继续按两级推进：

1. `dit_mlp_only --max-modules 0` 做 server prepare-only，确认 full DiT MLP 的 d8 prewarm stats、显存和 prepare 时间；
2. 选 1-2 个短 case 做 full DiT MLP rollout smoke，记录 success 和 latency；
3. 若 full DiT MLP 仍成功且 latency 有改善，再跑小规模 matched case set；
4. 最后再考虑 `llm_mlp_dit_mlp` 或 Phase 5 行为上最强的 `llm_dit_mlp` 路径。
