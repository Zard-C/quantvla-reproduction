# Phase 6 Initial Report: Triton W4A16 Linear Kernel

## 1. Goal

Phase 6 starts the engineering path from fake quantization to real packed-weight
execution. The first target is W4A16 Linear:

```text
weight:
  signed int4, packed two values per byte

activation:
  fp16

scale:
  fp32 per output channel

output:
  fp16
```

This is intentionally narrower than the Phase 5 fake W4A8 path. We first prove
real packed int4 weight storage and fused dequantization matmul before adding
activation quantization.

## 2. Implemented Files

```text
toy_quantvla/triton_w4a16.py
  pack_weight_w4_per_row
  unpack_int4_nibbles
  dequantize_w4_weight
  w4a16_linear_reference
  w4a16_linear_triton
  PackedW4A16Linear

toy_quantvla/phase6_w4a16_smoke.py
  correctness smoke
  odd-K test
  wrapper reshape test
  lightweight CUDA microbenchmark
```

The packed format matches the Phase 5 fake quant rule:

```text
scale[out] = max(abs(W[out, :])) / 7
q[out, in] = round(W[out, in] / scale[out]).clamp(-7, 7)
```

Storage layout:

```text
qweight: uint8 [out_features, ceil(in_features / 2)]
scale: fp32 [out_features]
bias: fp16/fp32 unchanged from source Linear
```

Signed int4 values are stored as unsigned nibbles:

```text
stored_nibble = q + 8
packed_byte = low_nibble | (high_nibble << 4)
```

For odd `K`, the final high nibble is padded with `q = 0`, encoded as `8`.

## 3. CUDA Smoke Environment

Ran on the RTX 5090 instance:

```text
torch: 2.8.0+cu128
cuda: 12.8
triton: 3.4.0
gpu: NVIDIA GeForce RTX 5090
```

Result JSON:

```text
toy_quantvla/results/phase6_w4a16_smoke_cuda.json
```

## 4. Correctness Results

The Triton kernel was compared against a PyTorch dense-dequant reference:

```text
reference:
  W_dequant = unpack(qweight) * scale
  y_ref = F.linear(x, W_dequant, bias)

kernel:
  y = Triton matmul(x, packed_qweight, scale, bias)
```

| M | K | N | max abs diff | relative RMSE | cosine |
|---:|---:|---:|---:|---:|---:|
| 1 | 17 | 13 | 0.0 | 0.0 | 1.00000012 |
| 8 | 64 | 32 | 0.0 | 0.0 | 0.99999994 |
| 32 | 1024 | 1024 | 0.000122 | 0.000353 | 1.00000000 |

The wrapper test also passed:

```text
input shape: [3, 5, 257]
output shape: [3, 5, 123]
max abs diff: 7.63e-06
relative RMSE: 3.19e-07
cosine: 1.0
```

This verifies:

```text
1. packed int4 layout works
2. odd-K padding works
3. per-row scale indexing works
4. bias addition works
5. leading-dimension flatten/reshape works
```

## 5. Memory Result

For large Linear layers, packed weight plus fp32 scale is close to 25% of fp16
weight memory.

| M | K | N | fp16 weight bytes | packed bytes | scale bytes | ratio |
|---:|---:|---:|---:|---:|---:|---:|
| 32 | 1024 | 1024 | 2,097,152 | 524,288 | 4,096 | 25.20% |

For very small layers, scale overhead is visible:

```text
M=1, K=17, N=13:
  ratio = 38.24%
```

But the target GR00T MLP Linear layers are large enough that scale overhead is
negligible.

## 6. Latency Smoke

The first kernel is correctness-first and not yet optimized.

| M | K | N | torch fp16 Linear | dense dequant ref | W4A16 Triton |
|---:|---:|---:|---:|---:|---:|
| 1 | 17 | 13 | 0.0107 ms | 0.0104 ms | 0.0388 ms |
| 8 | 64 | 32 | 0.0113 ms | 0.0110 ms | 0.0362 ms |
| 32 | 1024 | 1024 | 0.0135 ms | 0.0138 ms | 0.0379 ms |

Interpretation:

```text
The current Triton kernel is numerically correct but slower than cuBLAS fp16.
This is expected for the first implementation:

1. no autotuning
2. conservative tile sizes
3. no persistent/cache-aware schedule
4. dequantized weight tile is cast to fp16 before tl.dot
5. no specialized small-M path yet
```

So the current claim is:

```text
real packed W4A16 execution path exists and passes CUDA correctness smoke
```

The current claim is not:

```text
we already beat cuBLAS or have deployment speedup
```

## 7. Engineering Meaning

This is still a useful milestone because it crosses the boundary from fake quant
to real packed execution:

```text
Phase 5:
  fake W4A8 behavior validation

Phase 6 initial:
  packed W4A16 kernel correctness validation
```

The next engineering question is performance, not behavior.

## 8. Next Steps

Recommended next work:

```text
1. Add autotune configs:
   BLOCK_M in [16, 32, 64]
   BLOCK_N in [32, 64, 128]
   BLOCK_K in [64, 128, 256]
   num_warps in [4, 8]

2. Benchmark real selected GR00T Linear shapes.

3. Add a model patcher:
   replace selected llm_dit_mlp nn.Linear modules with PackedW4A16Linear.

4. Run one offline GR00T get_action smoke:
   FP16 vs W4A16 output drift.

5. Only after that, run a tiny LIBERO rollout smoke.
```

The immediate priority is shape-aware performance tuning. Simulator rollout is
not yet the bottleneck to test; the kernel still needs to become competitive on
the actual model shapes.

