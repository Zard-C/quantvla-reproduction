# Phase 8 CUTLASS FP4 Module Wrapper Smoke

日期：2026-06-08

## 目标

这一步把上一轮 single-layer CUTLASS blockscaled FP4 GEMM bridge 接成真正的 `nn.Linear` replacement：

- `CutlassBlockscaledFP4Linear.from_linear(...)`
- weight 在 patch 时离线 pack 一次；
- activation 在 forward 时 runtime pack；
- CUTLASS GEMM 按 runtime `M` 缓存 compile；
- 用 GR00T `get_action` 做 teacher/student action drift、latency、显存 smoke。

新增代码：

- `toy_quantvla/cutlass_blockscaled_fp4_linear.py`
- `toy_quantvla/phase8_cutlass_blockscaled_fp4_forward_smoke.py`

边界：

- 当前 activation pack 仍然使用 CuTe helper conversion，不是最终 fused runtime packer。
- 因此这一步验证的是 module replacement 可行性，不代表最终 inference latency。
- 为避免 helper pack 过慢，smoke 只 patch 1-2 个模块。

## 运行配置

真实 LeRobot observation：

```text
dataset: /root/autodl-tmp/datasets/libero_10_subset
dataset index: 115
denoising_steps: 1
sf_vec_size: 16
sf_dtype: Float8E4M3FN
tile_shape_mnk: 128,128,128
epi_tile: 64,32
```

结果文件：

- `toy_quantvla/results/phase8_cutlass_blockscaled_fp4_forward_smoke_dit_mlp_1mod.json`
- `toy_quantvla/results/phase8_cutlass_blockscaled_fp4_forward_smoke_dit_mlp_2mod.json`
- `toy_quantvla/results/phase8_cutlass_blockscaled_fp4_forward_smoke_llm_mlp_1mod.json`

## End-to-end get_action smoke

| config | patched modules | teacher | cold student | warm student | warm / teacher | action rel RMSE | cosine |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| DiT MLP first projection | 1 | 0.574s | 25.01s | 0.588s | 1.02x | 0.00391 | 0.999992 |
| DiT MLP first two projections | 2 | 0.607s | 51.20s | 1.269s | 2.09x | 0.00747 | 0.999974 |
| LLM MLP gate projection | 1 | 0.497s | 25.90s | 1.345s | 2.70x | 0.00416 | 0.999991 |

解读：

- module replacement 已经真实穿过 GR00T `get_action`。
- action drift 很小，1-2 个模块 patch 后 action cosine 仍接近 1。
- cold student 包含首次 CUTLASS compile，约 25s/shape。
- warm student 仍变慢，是因为 helper activation pack 在 forward 里执行。

## Module internal stats

### DiT MLP first projection

`action_head.model.transformer_blocks.0.ff.net.0.proj`

```text
shape: in=1536, out=6144, runtime M=49
weight pack: 9.48s
activation pack: ~0.54s / call
compile: 24.31s
GEMM: ~0.0041s measured inside wrapper
```

注意：wrapper 内 GEMM timing 包含 Python call/synchronize 影响；真正可信的 GEMM microbench 仍以 real activation bench 为准：

```text
M=49,K=1536,N=6144
CUTLASS FP4: 0.0098ms
torch fp16: 0.0226ms
speedup: 2.31x
```

### DiT MLP second projection

`action_head.model.transformer_blocks.0.ff.net.2`

```text
shape: in=6144, out=1536, runtime M=49
weight pack: 9.37s
activation pack: ~0.70s / call
compile: 24.65s
```

### LLM MLP gate projection

`backbone.eagle_model.language_model.model.layers.0.mlp.gate_proj`

```text
shape: in=2048, out=6144, runtime M=566
weight pack: 12.44s
activation pack: ~1.31s / call
compile: 24.47s
```

## Memory

Full `get_action` peak allocated:

| config | teacher peak | warm student peak |
| --- | ---: | ---: |
| DiT MLP 1 module | 5.683GB | 5.674GB |
| DiT MLP 2 modules | 5.683GB | 5.665GB |
| LLM MLP 1 module | 5.683GB | 5.672GB |

这一步还看不到明显 end-to-end 显存下降，原因是：

1. 绝大多数模型权重仍然以原格式常驻；
2. 当前 helper FP4 storage 用 int8 backing tensor，不是 two-values-per-byte 的最终 packed tensor；
3. 只 patch 了 1-2 个模块，收益被整模型常驻显存淹没。

不过从单层 footprint 估算，理论 FP4+scale activation/weight 是 FP16 的 28.1%。真正要兑现这个数字，需要：

- selected weight 全部 offline pack 后常驻；
- 删除/释放原始 FP16 selected weights；
- activation packer 输出真正 packed FP4 buffer。

## 当前结论

可以确认：

1. `CutlassBlockscaledFP4Linear` 已能作为 `nn.Linear` replacement 跑通 GR00T `get_action`。
2. weight offline pack + runtime activation pack + cached CUTLASS GEMM 的工程结构可行。
3. 小范围 patch 的 action drift 很小。
4. 当前 end-to-end 变慢不是 GEMM 问题，而是 helper activation pack 问题。

还不能确认：

1. 当前 wrapper 不适合大规模 patch 或 rollout。
2. helper pack 仍然是最大 blocker。
3. 当前 storage 还不是理想 packed memory layout。
4. 还没有 true end-to-end FP4 latency / memory benefit。

## 下一步

下一步应该集中写 runtime activation packer，而不是继续扩大 helper wrapper 的 rollout：

1. 先实现一个 dedicated activation pack op：
   - input `torch.Tensor[M,K]`;
   - output FP4 packed storage + scale storage；
   - scale layout 直接写成 CUTLASS SM120 需要的 MMA layout；
   - 使用 decoded scale 归一化，保持当前 bridge 的数值语义。
2. weight packer 可以先继续用 helper path offline pack，后续再替换。
3. activation packer 完成后，再重跑：
   - DiT MLP 2 modules；
   - LLM MLP 1 module；
   - `dit_mlp_only` 全 32 modules；
   - end-to-end `get_action` latency and peak memory。

## Update: torch-side packer

后续已经实现 torch-side blockscaled FP4 packer，能 byte-exact 替代 CuTe helper conversion：

- 详见 `docs/phase8_torch_fp4_packer_report.md`
- 关键结论：FP4 meaningful bytes、scale storage、CUTLASS decode 和 GEMM 输出都与 helper 对齐；
- DiT MLP 1 module warm `get_action` 从 helper 版 0.588s 降到 0.163s；
- DiT MLP 2 modules warm `get_action` 从 helper 版 1.269s 降到 0.268s；
- LLM MLP 1 module warm `get_action` 从 helper 版 1.345s 降到 0.182s；
- DiT MLP 8 modules 仍慢于 teacher，说明下一步瓶颈已经转向 pack kernel 化、buffer 复用和 compile cache 共享。

## Update: cached Triton packer

再后续已经实现 cached Triton FP4 packer：

- 详见 `docs/phase8_triton_fp4_packer_cached_report.md`
- DiT MLP 8 modules warm `get_action` 已从 torchpack 版 0.953s 降到 0.048s；
- 当前 blocker 从 runtime pack 转向 CUTLASS per-module cold compile，以及更大 scope / rollout 行为验证。
