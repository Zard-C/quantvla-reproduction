# Phase 8 Real Activation Speed and Memory Bench

日期：2026-06-08

## 目标

这一步把 Phase 8 blockscaled FP4 bridge 从 synthetic activation 推到真实 GR00T forward activation：

1. 用 LeRobot LIBERO 真实样本跑 GR00T `get_action`；
2. forward hook 捕获 selected `nn.Linear` 的真实输入 activation；
3. 对同一个 `activation + real weight` 跑：
   - torch fp16 GEMM；
   - CUTLASS SM120 blockscaled FP4 GEMM；
4. 统计速度、量化误差、kernel reference 误差、pack/compile 开销、显存和理论 footprint。

新增脚本：

- `toy_quantvla/phase8_cutlass_blockscaled_fp4_real_activation_bench.py`

边界：

- 当前 CUTLASS FP4 pack 仍使用 CuTe helper conversion，不是最终 fused runtime packer。
- 当前统计的是单层 GEMM backend，不是已经 patch 到全模型的 end-to-end FP4 inference。
- 显存统计同时给出 full `get_action` CUDA peak 和单层理论 footprint。由于整模型常驻显存约 5.5GB，单层 kernel 的 CUDA peak 差异会被模型本体淹没。

## 配置

两轮小规模实验都使用：

```bash
--observation-source real
--indices 115
--num-observations 1
--denoising-steps 1
--sf-vec-size 16
--sf-dtype Float8E4M3FN
--tile-shape-mnk 128,128,128
--epi-tile 64,32
```

结果文件：

- `toy_quantvla/results/phase8_cutlass_blockscaled_fp4_real_activation_bench_small_e4m3.json`
- `toy_quantvla/results/phase8_cutlass_blockscaled_fp4_real_activation_bench_dit_mlp_e4m3.json`

## Full get_action capture memory

真实 observation `dataset index = 115`，捕获 activation 时的 FP16 teacher `get_action`：

| scope | capture seconds | allocated | peak allocated | reserved |
| --- | ---: | ---: | ---: | ---: |
| `llm_dit_mlp` first 4 cases | 0.812s | 5.48GB | 5.68GB | 5.78GB |
| `dit_mlp_only` | 0.547s | 5.48GB | 5.68GB | 5.78GB |

解读：

- 当前完整 FP16 GR00T 推理常驻/峰值显存大约在 5.5GB 到 5.7GB。
- 单层 FP4 GEMM 的显存收益不会明显反映在这个 peak 里，因为模型主体仍是 FP16/bf16 常驻。
- 真正要看到 end-to-end 显存下降，需要把 selected weights 以 FP4 packed buffer 常驻，而不是只在单层 bench 中临时 pack。

## GEMM speed and error

### Summary

| group | cases | CUTLASS FP4 mean ms | torch fp16 mean ms | speedup | quant RMSE vs fp32 | kernel RMSE vs decoded ref |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `llm_attn` | 2 | 0.0101 | 0.0253 | 2.49x | 0.0692 | 2.06e-4 |
| `llm_mlp` | 2 | 0.0211 | 0.0754 | 3.62x | 0.1050 | 2.07e-4 |
| `dit_mlp` | 2 | 0.0157 | 0.0239 | 1.74x | 0.0348 | 2.05e-4 |

这组结果比上一轮 real-weight/synthetic-activation bench 更重要，因为 activation 来自真实 GR00T forward。

### Per-case

| group | shape `(M,K,N)` | CUTLASS FP4 ms | torch fp16 ms | speedup | quant RMSE |
| --- | --- | ---: | ---: | ---: | ---: |
| `llm_attn` | `(566,2048,1024)` | 0.00989 | 0.02283 | 2.31x | 0.0658 |
| `llm_attn` | `(566,2048,2048)` | 0.01039 | 0.02777 | 2.67x | 0.0726 |
| `llm_mlp` | `(566,2048,6144)` | 0.01898 | 0.07723 | 4.07x | 0.1076 |
| `llm_mlp` | `(566,6144,2048)` | 0.02320 | 0.07364 | 3.17x | 0.1024 |
| `dit_mlp` | `(49,1536,6144)` | 0.00981 | 0.02264 | 2.31x | 0.0333 |
| `dit_mlp` | `(49,6144,1536)` | 0.02166 | 0.02526 | 1.17x | 0.0362 |

解读：

- LLM MLP 是最有工程收益的区域：真实 activation 下仍有 3x 到 4x GEMM speedup。
- DiT MLP 的第一个 projection 有 2.3x speedup，第二个 projection 只有 1.17x，说明小 M + skinny output shape 的 launch/tiling overhead 更敏感。
- 量化误差上，DiT MLP 反而最稳，真实 activation 下 mean relative RMSE 只有约 0.035。
- kernel RMSE 一直在 `~2e-4`，说明 CUTLASS output 与 decoded FP4 reference 对齐，误差主要来自 FP4 量化本身，不是 layout/launch 错误。

## Theoretical activation + weight footprint

当前使用 `sf_vec_size=16`，理论上每 16 个 FP4 value 有 1 个 FP8 scale。单层 activation+weight footprint：

| group / shape | FP16 act+weight | theoretical FP4+scale | ratio |
| --- | ---: | ---: | ---: |
| `llm_attn (566,2048,1024)` | 6.51MB | 1.83MB | 28.1% |
| `llm_attn (566,2048,2048)` | 10.71MB | 3.01MB | 28.1% |
| `llm_mlp (566,2048,6144)` | 27.48MB | 7.73MB | 28.1% |
| `llm_mlp (566,6144,2048)` | 32.12MB | 9.03MB | 28.1% |
| `dit_mlp (49,1536,6144)` | 19.02MB | 5.35MB | 28.1% |
| `dit_mlp (49,6144,1536)` | 19.48MB | 5.48MB | 28.1% |

这个 28.1% 是理想 packed representation：

```text
FP4 values: 0.5 byte/value
FP8 scale: 1 byte / 16 values
total = 0.5 + 1/16 = 0.5625 byte/value
FP16 = 2 bytes/value
ratio = 0.5625 / 2 = 28.125%
```

## Current helper pack overhead

当前 helper pack 很慢：

| group | helper pack mean | compile mean |
| --- | ---: | ---: |
| `llm_attn` | 5.18s | 24.21s |
| `llm_mlp` | 14.98s | 24.34s |
| `dit_mlp` | 10.34s | 25.48s |

这不是最终推理性能，只说明当前 bridge 使用的 CuTe helper conversion 不能作为 runtime packer。后续必须实现真正的 runtime activation pack kernel，并把 weight offline pack 常驻化。

当前 helper storage 也不是理想 packed storage。因为 helper path 的 FP4 backing tensor 使用 int8 torch storage 承载 CUTLASS dtype view，所以实际 helper storage 约为理论 packed 的 1.9x 左右。最终 backend 需要真正 two-values-per-byte 的 packed representation，才能兑现理论显存。

## 当前结论

可以确认：

1. 真实 GR00T activation 下，CUTLASS blockscaled FP4 GEMM 仍然有明显速度收益。
2. 真实 activation 的量化误差比 synthetic random activation 更温和：
   - DiT MLP 约 0.035；
   - LLM attention 约 0.069；
   - LLM MLP 约 0.105。
3. 单层理论 activation+weight footprint 可降到 FP16 的 28.1%。
4. 当前整模型 FP16 `get_action` 峰值约 5.68GB；要看到 end-to-end 显存下降，必须做 packed weight 常驻和 module wrapper。

还不能确认：

1. 当前 helper pack 不能用于真实 runtime。
2. 还没有 `CutlassBlockscaledFP4Linear` module。
3. 还没有 end-to-end `get_action` FP4 latency / peak memory。
4. 还没有 closed-loop LIBERO success rate。

## 下一步

下一步应进入 module wrapper：

1. `CutlassBlockscaledFP4Linear.from_linear(...)`
   - weight offline pack once；
   - weight FP4 + scale registered as buffers；
   - shape-level compiled GEMM cache。
2. runtime activation pack kernel：
   - 先用 helper path 验证功能；
   - 再替换为 fused CUDA/CuTe/Triton packer。
3. end-to-end smoke：
   - `dit_mlp_only` get_action drift + speed + peak memory；
   - `llm_mlp_only`；
   - `llm_dit_mlp`。
4. 小规模 LIBERO rollout：
   - 复用 Phase 5/7 matched case list；
   - 报告 success rate、mean step latency、server peak memory。
