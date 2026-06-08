# Phase 8 Triton FP4 Packer + Cached Operand 报告

日期：2026-06-08

## 目标

上一阶段 torch-side packer 已经 byte-exact 替代 CUTLASS helper conversion，但 runtime 仍有约 115 ms/module 的 activation pack 成本。本阶段继续推进性能路线：

- 用 Triton kernel 实现 blockscaled FP4 activation/weight pack；
- 保持 CUTLASS SM120 blockscaled GEMM 的 operand contract 不变；
- 在 `CutlassBlockscaledFP4Linear` 里缓存 activation operand storage，避免每次 forward 重新创建 CUTLASS tensor/storage；
- 用真实 GR00T `get_action` smoke 验证速度、显存和 action drift。

新增/修改代码：

- `toy_quantvla/blockscaled_fp4_packer.py`
- `toy_quantvla/cutlass_blockscaled_fp4_linear.py`
- `toy_quantvla/phase8_blockscaled_fp4_packer_smoke.py`
- `toy_quantvla/phase8_cutlass_blockscaled_fp4_forward_smoke.py`
- `toy_quantvla/phase8_triton_real_operand_bytecheck.py`

## Triton Packer 实现边界

当前 Triton packer 支持我们真实目标路径所需的配置：

```text
l = 1
sf_vec_size = 16
sf_dtype = Float8E4M3FN
K must be even
```

GR00T 的 DiT/LLM MLP Linear shape 都满足这些条件。

Triton kernel 做的事情：

```text
1. 每个 (row, scale-block) 读取 16 个元素
2. 计算 raw_scale = max(abs(x)) / 6
3. 编码 Float8E4M3FN scale byte
4. 用 decoded scale 做 normalized = x / decoded_scale
5. 编码 Float4E2M1FN code
6. 两个 FP4 code 打包进一个 int8 byte
7. scale byte 写入 CUTLASS SM120 MMA scale layout
```

一个关键修正：

- 真实 DiT MLP shape 初版在 E4M3 subnormal 到 min-normal 边界有 23/4704 个 scale byte 不一致；
- 原因是 `~0.01465` 以上的值应从 subnormal `0x07` round up 到 min-normal `0x08`；
- 修正后真实 shape 也达到 byte-exact。

## Byte-exact Correctness

结果文件：

- `toy_quantvla/results/phase8_blockscaled_fp4_packer_smoke_small_triton.json`
- `toy_quantvla/results/phase8_blockscaled_fp4_packer_smoke_small_triton_gemm.json`
- `toy_quantvla/results/phase8_blockscaled_fp4_packer_smoke_dit_mlp_triton.json`

| case | shape `(M,K,N)` | FP4 meaningful bytes | scale storage | CUTLASS decode diff | GEMM diff |
| --- | ---: | --- | --- | ---: | ---: |
| small | `(9,64,17)` | equal | equal | 0 | not run |
| small + GEMM | `(9,64,17)` | equal | equal | 0 | 0 |
| DiT MLP | `(49,1536,6144)` | equal | equal | 0 | not run |

这里的 `equal` 是 byte-level equality，对齐的是 CUTLASS helper conversion，不是只看数值误差。

## Real Operand Byte-check

为了避免只在随机张量上对齐，又补了真实 GR00T operand byte-check：

结果文件：

- `toy_quantvla/results/phase8_triton_real_operand_bytecheck_dit_mlp_2cases.json`

真实 observation `dataset index=115`，捕获 DiT MLP 前两个 Linear 的 runtime activation 和真实 weight：

| module | shape `(M,K,N)` | activation FP4 bytes | activation scale | weight FP4 bytes | weight scale | decoded diff |
| --- | ---: | --- | --- | --- | --- | ---: |
| `transformer_blocks.0.ff.net.0.proj` | `(49,1536,6144)` | equal | equal | equal | equal | 0 |
| `transformer_blocks.0.ff.net.2` | `(49,6144,1536)` | equal | equal | equal | equal | 0 |

这里修掉了两个真实数据才暴露的边界：

- E4M3 subnormal round up 到 min-normal `0x08`；
- FP4 tie 和 negative zero 的 byte-level 处理。
- scale byte 量化成 0 时，要匹配 helper 的除零语义：负数饱和到 `-6`，`0/0` 的 NaN 经 FP4 convert 走到 `+6`。

## 非缓存 Packer Speed

这些数字仍包含每次创建 CUTLASS tensor/storage 的固定开销，所以不能代表 kernel-only 时间：

| case | helper activation pack | Triton activation pack | speedup | helper weight pack | Triton weight pack | speedup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| small `(9,64,17)` | 364.1 ms | 111.8 ms | 3.26x | 369.7 ms | 113.3 ms | 3.26x |
| small + GEMM `(9,64,17)` | 369.1 ms | 113.8 ms | 3.24x | 366.0 ms | 113.0 ms | 3.24x |
| DiT MLP `(49,1536,6144)` | 433.9 ms | 115.8 ms | 3.75x | 8999.5 ms | 117.6 ms | 76.52x |

结论：

- 单纯把 torch operations 换成 Triton kernel 后，pack time 仍约 110-120 ms；
- 主要瓶颈已经不是 pack arithmetic，而是每次重新构造 CUTLASS operand storage；
- 因此必须做 cached operand。

## Cached Operand get_action Smoke

真实 LeRobot observation：

```text
dataset index: 115
denoising_steps: 1
pack_backend: triton
cached activation operand: enabled
```

结果文件：

- `toy_quantvla/results/phase8_cutlass_blockscaled_fp4_forward_smoke_dit_mlp_1mod_tritonpack_cached.json`
- `toy_quantvla/results/phase8_cutlass_blockscaled_fp4_forward_smoke_dit_mlp_2mod_tritonpack_cached.json`
- `toy_quantvla/results/phase8_cutlass_blockscaled_fp4_forward_smoke_llm_mlp_1mod_tritonpack_cached.json`
- `toy_quantvla/results/phase8_cutlass_blockscaled_fp4_forward_smoke_dit_mlp_8mod_tritonpack_cached.json`

| config | patched modules | teacher | patch | cold student | warm student | warm / teacher | action rel RMSE | cosine |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| DiT MLP | 1 | 0.422s | 0.800s | 24.79s | 0.044s | 0.10x | 0.00391 | 0.999992 |
| DiT MLP | 2 | 0.462s | 0.941s | 49.20s | 0.045s | 0.10x | 0.00747 | 0.999974 |
| LLM MLP | 1 | 0.489s | 1.040s | 24.98s | 0.041s | 0.08x | 0.00416 | 0.999991 |
| DiT MLP | 8 | 0.567s | 1.607s | 195.37s | 0.048s | 0.09x | 0.00773 | 0.999973 |

对比上一阶段 torch-side packer：

| config | torch-pack warm | cached Triton warm | improvement |
| --- | ---: | ---: | ---: |
| DiT MLP 1 module | 0.163s | 0.044s | 3.69x |
| DiT MLP 2 modules | 0.268s | 0.045s | 6.01x |
| LLM MLP 1 module | 0.182s | 0.041s | 4.44x |
| DiT MLP 8 modules | 0.953s | 0.048s | 19.75x |

这说明 `llm_dit_mlp_only` 这条工程路线开始真正出现希望：至少在 small offline `get_action` smoke 中，大 scope 不再被 runtime pack 拖死。

## 显存观察

8-module DiT MLP：

```text
teacher peak allocated: 5.683 GB
cached Triton warm peak allocated: 5.622 GB
```

显存有小幅下降，但还没有兑现理论 FP4+scale 的完整收益。原因仍然是：

- 大多数模型权重仍保留原始 dtype；
- 只 patch 了部分模块；
- CUTLASS DSL wrapper 和 compile/runtime cache 仍有额外 reserved memory；
- 还没有做整 scope 的原权重释放策略。

## 当前结论

可以确认：

1. Triton packer 已经 byte-exact 对齐 CUTLASS helper conversion。
2. 缓存 activation operand 后，runtime pack 固定开销基本被消掉。
3. DiT MLP 8 modules 的 warm `get_action` 从 torchpack 的 0.953s 降到 0.048s。
4. action drift 仍然很小，8 modules 的 cosine 约 0.999971。

仍然不能忽略：

1. cold student 仍被 CUTLASS per-module compile 主导，8 modules 约 200s。
2. 目前只测了 `denoising_steps=1` 的 offline `get_action` smoke，还不是 LIBERO rollout。
3. full scope 可能出现更多 unique M/shape，需要 compile/cache 策略。
4. 需要做真实 rollout 前，最好先跑更大 patch scope 的 offline action drift。

## 下一步

建议下一步按这个顺序：

1. 跑 `dit_mlp_only` 更大 scope，例如 16 modules / full DiT MLP，先只做 `get_action` smoke。
2. 跑 `llm_mlp_only` 或 `llm_dit_mlp` 的更大 scope offline action drift。
3. 研究 CUTLASS compile cache 共享，同 shape 的 modules 不应每个都 cold compile 25s。
4. 在 offline action drift 可控后，再进入小规模 LIBERO rollout，对比 FP16 baseline、fake quant、cached Triton FP4。
