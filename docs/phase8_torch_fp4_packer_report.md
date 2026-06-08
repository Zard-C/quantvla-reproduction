# Phase 8 Torch-side CUTLASS FP4 Packer 报告

日期：2026-06-08

## 目标

上一轮 `CutlassBlockscaledFP4Linear` 已经能跑通 GR00T `get_action`，但 activation pack 仍然走 CuTe helper conversion，导致 runtime 很慢。本阶段目标是写一个 torch-side packer，先把 helper conversion 从 runtime 路径里拿掉：

- FP4 data：手写 Float4E2M1FN quant + two-nibbles-per-byte pack；
- scale：用 `torch.float8_e4m3fn` 量化，再写成 CUTLASS SM120 blockscaled GEMM 需要的 MMA scale layout；
- wrapper：增加 `--pack-backend helper|torch`，用于 A/B 测试；
- 验证：对齐 helper 字节、CUTLASS decode、GEMM 输出、真实 `get_action` latency 和 action drift。

新增代码：

- `toy_quantvla/blockscaled_fp4_packer.py`
- `toy_quantvla/phase8_blockscaled_fp4_packer_smoke.py`

修改代码：

- `toy_quantvla/cutlass_blockscaled_fp4_linear.py`
- `toy_quantvla/phase8_cutlass_blockscaled_fp4_forward_smoke.py`

## 字节格式结论

FP4 storage 不是每个 logical element 一个有效 byte，而是两个 FP4 code 打进一个 int8 byte：

```text
packed_byte = low_nibble(first_value_code) | high_nibble(second_value_code)
```

其中 Float4E2M1FN 的 magnitude code 对应：

| code | value |
| ---: | ---: |
| 0 | 0 |
| 1 | 0.5 |
| 2 | 1 |
| 3 | 1.5 |
| 4 | 2 |
| 5 | 3 |
| 6 | 4 |
| 7 | 6 |

负数通过 sign bit 加 `8`。midpoint 使用 round-to-nearest-even，例如 `0.75 -> 1.0`、`1.25 -> 1.0`、`2.5 -> 2.0`。

scale storage 使用 `Float8E4M3FN` 字节，但必须重排到 CUTLASS 的 MMA layout：

```text
logical shape: (32, 4, rest_m, 4, rest_k, l)
stride example for M=49,K=1536,sf_vec=16:
(16, 4, 12288, 1, 512, 12288)
```

这个 stride/layout 是关键；scale 数值本身和 `torch.float8_e4m3fn` 的字节一致。

## Packer Correctness

结果文件：

- `toy_quantvla/results/phase8_blockscaled_fp4_packer_smoke_small.json`
- `toy_quantvla/results/phase8_blockscaled_fp4_packer_smoke_small_gemm.json`
- `toy_quantvla/results/phase8_blockscaled_fp4_packer_smoke_dit_mlp.json`

| case | shape `(M,K,N)` | FP4 meaningful bytes | scale storage | CUTLASS decode diff | GEMM diff |
| --- | ---: | --- | --- | ---: | ---: |
| small | `(9,64,17)` | equal | equal | 0 | not run |
| small + GEMM | `(9,64,17)` | equal | equal | 0 | 0 |
| DiT MLP | `(49,1536,6144)` | equal | equal | 0 | not run |

这里的 `equal` 是 byte-level equality，不是只比较 dequant 后误差。也就是说，torch packer 生成的有效 FP4 bytes 和 scale bytes 与 CUTLASS helper 完全一致。

## Packer Speed

| case | helper activation pack | torch activation pack | speedup | helper weight pack | torch weight pack | speedup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| small `(9,64,17)` | 367.9 ms | 117.8 ms | 3.12x | 371.6 ms | 115.0 ms | 3.23x |
| small + GEMM `(9,64,17)` | 361.5 ms | 112.4 ms | 3.22x | 360.7 ms | 113.2 ms | 3.19x |
| DiT MLP `(49,1536,6144)` | 438.5 ms | 114.6 ms | 3.83x | 9211.9 ms | 118.8 ms | 77.55x |

解读：

- activation pack 已经从 helper 的 0.4s 级别降到 0.11s 级别。
- weight pack 收益非常大，主要因为 helper 对大 weight 的 conversion 非常慢；torch packer 后 patch 阶段不再被 9-12s/module 卡住。
- 但 0.11s/module 对 runtime 来说仍然偏慢。它证明了格式和路径，但还不是最终性能形态。

## get_action Smoke

真实 LeRobot observation：

```text
dataset index: 115
denoising_steps: 1
sf_vec_size: 16
sf_dtype: Float8E4M3FN
pack_backend: torch
```

结果文件：

- `toy_quantvla/results/phase8_cutlass_blockscaled_fp4_forward_smoke_dit_mlp_1mod_torchpack.json`
- `toy_quantvla/results/phase8_cutlass_blockscaled_fp4_forward_smoke_dit_mlp_2mod_torchpack.json`
- `toy_quantvla/results/phase8_cutlass_blockscaled_fp4_forward_smoke_llm_mlp_1mod_torchpack.json`
- `toy_quantvla/results/phase8_cutlass_blockscaled_fp4_forward_smoke_dit_mlp_8mod_torchpack.json`

| config | patched modules | teacher | patch | cold student | warm student | warm / teacher | action rel RMSE | cosine |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| DiT MLP | 1 | 0.663s | 0.772s | 25.29s | 0.163s | 0.25x | 0.00391 | 0.999992 |
| DiT MLP | 2 | 0.464s | 0.868s | 48.65s | 0.268s | 0.58x | 0.00747 | 0.999974 |
| LLM MLP | 1 | 0.457s | 0.691s | 25.56s | 0.182s | 0.40x | 0.00416 | 0.999991 |
| DiT MLP | 8 | 0.428s | 1.482s | 203.41s | 0.953s | 2.23x | 0.00773 | 0.999973 |

对比上一版 helper wrapper：

| config | helper warm | torch-pack warm | improvement |
| --- | ---: | ---: | ---: |
| DiT MLP 1 module | 0.588s | 0.163s | 3.61x |
| DiT MLP 2 modules | 1.269s | 0.268s | 4.74x |
| LLM MLP 1 module | 1.345s | 0.182s | 7.38x |

动作误差基本和 helper wrapper 版本一致，因为有效 FP4/scale bytes 完全一致。

## 当前结论

这一步是实质性推进：

1. 已经确认 CUTLASS helper 的 FP4/scale storage contract。
2. torch-side packer 可以 byte-exact 替代 helper conversion。
3. 单模块和双模块 `get_action` smoke 已经出现真实 latency 改善。
4. weight pack 从十秒级降到亚秒级，patch 大 scope 不再被 helper weight conversion 卡死。

但还不能说最终性能路线已经走通：

1. torch packer 每个 module 仍约 115 ms，8 modules 时 warm `get_action` 已经慢于 teacher。
2. cold student 仍然被 per-module CUTLASS compile 主导，8 modules 约 203s。
3. 当前 wrapper 每个 module 都独立 allocate/pack/compile cache，没有跨模块复用。
4. storage backing 仍通过 CUTLASS DSL tensor wrapper 走 int8 storage，端到端显存收益还没有完全兑现。

## 下一步

优先级建议：

1. 写真正的 Triton/CUDA activation pack kernel，把 `amax + FP8 scale + normalized FP4 pack + MMA scale reorder` 融成 1-2 个 kernel。
2. 做 packed buffer 复用，避免每个 forward 创建新的 CUTLASS tensor/storage。
3. 研究 compile cache 共享：同 shape 的 DiT MLP projection 不应该每个 module 各编译一次。
4. 在 pack kernel 完成后，再跑 `dit_mlp_only` 更大 scope 和 LIBERO rollout。

当前不建议直接跑大规模 rollout。8-module smoke 已经显示 runtime pack/allocation 仍是瓶颈；先把 pack kernel 化，性价比更高。

## Update: Triton cached packer

后续已经实现 Triton pack kernel，并在 `CutlassBlockscaledFP4Linear` 中缓存 activation operand storage：

- 详见 `docs/phase8_triton_fp4_packer_cached_report.md`
- Triton packer 对 small 和 DiT MLP 真实 shape 都达到 byte-exact；
- DiT MLP 8 modules warm `get_action` 从 torchpack 版 0.953s 降到 0.048s；
- 下一步瓶颈转向 CUTLASS per-module cold compile 和更大 scope 的 action drift/rollout 验证。
