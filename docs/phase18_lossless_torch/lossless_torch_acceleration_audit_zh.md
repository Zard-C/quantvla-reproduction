# Phase 18 无损 Torch 加速审计

## 边界

这一阶段只讨论量化之前的无损加速：不改权重、不改低比特 kernel、不接受同观测 action drift。当前产物基于已有 5090 结果和本地代码静态审计，没有重新跑 GPU。

## 当前 FP16 基线

| item | value |
| --- | --- |
| 30-case success | 16/30 |
| server get_action p50 | 156.2 ms |
| server get_action p90 | 161.5 ms |
| client policy p50 | 161.1 ms |
| client preprocess p50 | 0.113 ms |
| client postprocess p50 | 0.101 ms |
| client-server overhead p50 | 4.674 ms |
| reserved memory | 5512.0 MiB |

直接结论：client 侧 preprocess/postprocess 是 0.1 ms 量级，通信/序列化差额约 4.7 ms，主瓶颈仍是 server `policy.get_action`。

## 覆盖面事实

| profile scope | per get_action | share |
| --- | --- | --- |
| DiT MLP Linear only | 13.318 ms | 8.4% of profiled server mean |
| LLM attn + LLM MLP + DiT MLP selected linears | 17.556 ms | 17.12% of offline profile p50 |

这解释了之前 `dit_mlp_only` 很难端到端提速：它在 FP16 server 里的覆盖面只有约 8.4%。如果不改变更大范围的算子执行，单独优化这块的理论上限很低。

## 已验证路线对比

| route | speed | behavior | judgement |
| --- | --- | --- | --- |
| CUDA graph DiT forward | p50 1.018x, p90 1.504x | fixed replay max diff 0 | clean but small |
| torch.compile block-island | server p50 2.04x | success 16/30 -> 13/30 | fast but not lossless |

CUDA Graph 是目前唯一已经证明同观测 bit-exact 的加速尝试，但 p50 基本不动，主要改善 p90。`torch.compile` 速度强，但已经不是无损替换。

## 静态审计结论

- `client preprocessing/postprocessing`: LIBERO client 图像/状态预处理和 action 后处理的 p50 都约为 0.1 ms。 判断：`not_a_primary_bottleneck`。
- `env.step(action.tolist())`: 这里的 action 已经是 simulator client 侧的 NumPy 数组，不是 server GPU tensor 的 D2H。 判断：`do_not_optimize_first`。
- `server final action copy`: N1.5 官方 `policy.py` 在 unapply transform 前会执行 `normalized_action.cpu()`。这是返回 CPU/NumPy action 前的必要 copy，但它会同步 GPU 队列；需要在 5090 profiler 里单独量化。 判断：`measure_before_optimizing`。
- `RoPE / position compute`: N1.5 action head 内没有 RoPE，重复位置计算主要是 timestep sinusoidal embedding、DiT sinusoidal position embedding 和 action learned position embedding；真正 RoPE 在 Eagle 的 Qwen3 language model 内。 判断：`inspect_installed_transformers_qwen3`。
- `request trace and summary flushing`: per-request JSONL trace 每行都会 flush；latency summary flush 也可能在 rollout 中写文件。 判断：`disable_for_deployment_or_buffer`。
- `explicit CUDA synchronize`: server request CUDA sync 是显式 opt-in，应继续只用于 profiling；module profiler 的 synchronize 也是诊断用途。 判断：`safe_when_disabled`。
- `fixed shapes`: task description 长度会改变 encoder_hidden_states shape；已有 warmdesc prewarm，应在 rollout 前覆盖所有 eval task description。 判断：`required_for_compile_or_kernel_cache_paths`。
- `hidden D2H / scalar sync inside GR00T policy`: 当前 wrapper 层审计不能证明 Isaac-GR00T 内部没有隐藏 D2H/scalar sync，必须在 5090 上用 torch.profiler 验证。 判断：`needs_gpu_profile`。

更细的 RoPE/固定 shape 静态审计见 `docs/phase18_lossless_torch/static_code_audit_rope_fixed_shape_zh.md`。

## 优先级

| rank | candidate | lossless | expected | gate |
| --- | --- | --- | --- | --- |
| 1 | 低扰动 torch.profiler 审计 | yes | 先测量 | 找到每 request 反复出现且超过 1 ms 的 sync/D2H，或确认这类问题不存在。 |
| 2 | 部署态 timing hygiene | yes | 小 | 同一 30-case set 下 server p50/p90 不变差，成功率不变。 |
| 3 | shape-aware prewarm contract | yes | 只改善冷启动/长尾 | request-time 没有 cold spike，也没有新的 graph/compile/cache miss 记录。 |
| 4 | eager DiT forward 的 CUDA graph replay | yes | p50 ~1.02x, p90 better | matched rollout 的 success/action trace 与 FP16 对齐，同时 server p90 改善。 |
| 5 | torch.compile block-island | no | p50 ~2.04x | 不是无损路径；只能配合 behavior guard 或闭环修正使用。 |

当前建议不是先写 kernel，而是先在 5090 上做低扰动 profiler：如果没有隐藏 D2H/同步，Torch 层无损空间大概率只剩 hygiene、prewarm、CUDA Graph p90 优化；主 p50 加速仍要回到 kernel/compile/闭环修正路线。

## 5090 下一步命令草案

```bash
# 1. 纯 FP16 hygiene run：不打开 request trace、不打开 server cuda sync、不打开 module profiler。
python toy_quantvla/timed_fp16_inference_service.py --port 5555 --prewarm-observations 1 --prewarm-indices 115 --prewarm-task-description 'put the white mug on the left plate and put the yellow and white mug on the right plate' --prewarm-task-description 'put the white mug on the plate and put the chocolate pudding to the right of the plate' --prewarm-task-description 'put both moka pots on the stove' --server-latency-json toy_quantvla/results/phase18_fp16_hygiene_server_latency.json
python toy_quantvla/libero_eval_init_range.py --task-suite-name libero_10 --case-list 4:6,4:7,4:8,4:9,4:10,6:0,6:1,6:2,6:3,6:4,8:6,8:7,8:8,8:9,8:10 --headless --port 5555 --deterministic-policy-seeds --policy-seed-base 20260613 --latency-json toy_quantvla/results/phase18_fp16_hygiene_client_latency.json
# 2. profiler probe：包住短 fixed-observation 或短 rollout，重点 grep cudaMemcpyDtoH、aten::item、aten::_local_scalar_dense、cudaDeviceSynchronize。
# 3. 如果 profiler 没有发现隐藏 sync/D2H，就不要继续在 D2H 上耗时间；改测 CUDA graph DiT 的 p90-only clean optimization。
```

## 产物

- JSON: `toy_quantvla/results/phase18_lossless_torch_acceleration_audit.json`
- Report: `docs/phase18_lossless_torch/lossless_torch_acceleration_audit_zh.md`
