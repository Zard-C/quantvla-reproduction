# Phase 8 CUTLASS Blockscaled FP4 Bridge

日期：2026-06-06

## 目标

Phase 8 dense backend 已经证明 `nn.Linear -> CUTLASS callable op` 这条链路可行，但 dense fp16 不是最终目标。这一轮继续向真实量化 backend 推进，目标是验证：

1. 从普通 torch activation / weight 出发，能否构造 CUTLASS SM120 blockscaled FP4 operand；
2. scale factor tensor 的 MMA layout 是否能正确生成；
3. CUTLASS FP4 GEMM 输出是否匹配硬件语义参考；
4. 在真实 GR00T Linear 权重上，速度和量化误差大概处在什么区间。

新增脚本：

- `toy_quantvla/phase8_cutlass_blockscaled_fp4_smoke.py`
- `toy_quantvla/phase8_cutlass_blockscaled_fp4_real_weight_bench.py`

## 关键实现点

CUTLASS blockscaled GEMM 计算的是：

```text
C = (A_fp4 * SFA) @ (B_fp4 * SFB)^T
```

这里 `A_fp4/B_fp4` 是 `Float4E2M1FN`，`SFA/SFB` 是 FP8 scale factor。scale factor 不是普通 `(M, ceil(K/block), L)` contiguous tensor，而是要转换到 SM120 MMA 指定 layout：

```text
M(32 x 4 x rest_m) x K(4 x rest_k) x L
```

这次踩到的两个关键坑：

1. scale MMA tensor 不能 `.contiguous()`。
   - 官方示例用的是 `permute(...)` 后的非 contiguous stride。
   - `cute_tensor_like` 会读取 stride/leading dimension，拍平后 TMA 读到的 scale layout 就错了。
2. operand 归一化必须使用 decoded scale。
   - 流程不能是 `scale_raw = absmax / 6`，然后 `x / scale_raw`。
   - 因为 kernel 实际使用的是被 `Float8E4M3FN` 或 `Float8E8M0FNU` 量化后的 scale。
   - 正确流程是：

```text
scale_raw = absmax(x_block) / 6
scale_fp8 = quantize(scale_raw)
scale_decoded = dequantize(scale_fp8)
x_fp4 = quantize(x / scale_decoded)
output = x_fp4_decoded * scale_decoded
```

修正后，kernel 输出和 decoded FP4 + decoded scale 参考可以对齐到 `~1e-4` relative RMSE。

## 单层 Synthetic 结果

### 128 x 512 x 512

| config | CUTLASS FP4 ms | torch fp16 ms | speedup | kernel rel RMSE | quant rel RMSE vs fp32 |
| --- | ---: | ---: | ---: | ---: | ---: |
| sf32 + E8M0 | 0.00923 | 0.03863 | 4.18x | 6.42e-5 | 0.167 |
| sf16 + E4M3 | 0.00945 | 0.03713 | 3.93x | 2.08e-4 | 0.135 |

### GR00T DiT MLP shape, random weight

Shape: `M=49, K=1536, N=6144`

| config | CUTLASS FP4 ms | torch fp16 ms | speedup | kernel rel RMSE | quant rel RMSE vs fp32 |
| --- | ---: | ---: | ---: | ---: | ---: |
| sf32 + E8M0 | 0.00961 | 0.03726 | 3.88x | 1.59e-4 | 0.164 |
| sf16 + E4M3 | 0.00964 | 0.03950 | 4.10x | 2.08e-4 | 0.141 |

解读：

- E4M3/sf16 在这两个 smoke 上数值略好。
- E8M0/sf32 与 Phase 7 官方 example benchmark 一致，是更标准的 MXFP4 路线。
- 两者速度在这些 shape 上接近，都明显快于 torch fp16 GEMM。

## 真实 GR00T 权重结果

命令边界：

```bash
python toy_quantvla/phase8_cutlass_blockscaled_fp4_real_weight_bench.py \
  --scope llm_dit_mlp \
  --denoising-steps 1 \
  --num-observations 1 \
  --max-cases 3 \
  --sf-vec-size 16 \
  --sf-dtype Float8E4M3FN \
  --repeats 20 \
  --warmup 3 \
  --output-json toy_quantvla/results/phase8_cutlass_blockscaled_fp4_real_weight_bench_small_e4m3.json
```

结果：

| group | cases | CUTLASS FP4 mean ms | torch fp16 mean ms | speedup | quant rel RMSE vs fp32 | kernel rel RMSE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `dit_mlp` | 2 | 0.01582 | 0.03909 | 2.80x | 0.1468 | 2.09e-4 |
| `llm_attn` | 1 | 0.01083 | 0.03715 | 3.43x | 0.1403 | 2.08e-4 |

这组结果比纯随机权重更有价值，因为权重来自真实 GR00T checkpoint。activation 仍是 synthetic random input；下一步应该接真实 captured activation。

## 当前结论

可以确认：

1. 真实 tensor 到 CUTLASS blockscaled FP4 operand 的转换路径已经跑通。
2. scale factor MMA layout 的关键细节已明确。
3. kernel 输出与硬件语义参考高度一致，说明 pack/layout/launch 基本正确。
4. 在真实 GR00T 权重上，小规模 representative case 有 `2.8x-3.4x` GEMM speedup。

还不能确认：

1. 当前 pack path 使用 CUTLASS helper conversion，不是最终高性能 runtime packer。
2. activation 还不是真实 rollout/captured activation。
3. 还没有把 `CutlassBlockscaledFP4Linear` 接进 GR00T `get_action`。
4. 还没有做 closed-loop simulator success rate。

## 下一步

建议按这个顺序推进：

1. real activation bench：
   - 复用 Phase 6 hook，捕获真实 `x`；
   - 对真实 `x` + 真实 weight 跑 blockscaled FP4 GEMM；
   - 记录 per-layer quant RMSE 和速度。
2. module wrapper：
   - 实现 `CutlassBlockscaledFP4Linear`；
   - weight offline pack once；
   - activation runtime pack；
   - shape-level compiled GEMM cache。
3. `get_action` drift：
   - 先 `dit_mlp_only`；
   - 再 `llm_mlp_only`；
   - 最后 `llm_dit_mlp`。
4. 小规模 LIBERO rollout：
   - 用 Phase 5/7 的 matched case list；
   - 对比 FP16、fake FP4、CUTLASS real FP4。

工程判断：这一步是从“CUTLASS 例子能跑”到“真实 QuantVLA backend 有落地可能”的关键转折。现在真正的难点不再是 GEMM kernel，而是高效、可缓存、可接入模型 forward 的 FP4 packing。
