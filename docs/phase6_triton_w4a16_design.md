# Phase 6 Triton W4A16 Linear Design

Boundary: this is an engineering design note for turning the current fake
quantized `llm_dit_mlp` Linear path into a real packed-weight kernel path. It is
not yet an implementation or benchmark result.

Status update: the first correctness-oriented Triton implementation now exists.
See `docs/phase6_w4a16_triton_initial_report.md` for CUDA smoke results and
current limitations.

## Goal

Replace selected `nn.Linear` modules with a real packed W4A16 Linear:

```text
current fake quant:
  x -> fake A8 -> dequant fp -> torch F.linear(fake W4 dequant fp)

first real kernel:
  x fp16/bf16 -> Triton matmul(packed int4 W, scale) -> y fp16/bf16
```

The first target should be W4A16, not W4A8. This keeps activation floating point
and only packs weights. It is the smallest step that can prove real memory
savings and fused dequantization without changing the closed-loop behavior too
much.

## Match The Current Fake Quant Format

The current fake quant path in `toy_quantvla/phase3_fake_quant_forward.py`
quantizes Linear weights with:

```python
w_q, _ = fake_quant_symmetric(weight, weight_bits, scale_dim=1)
```

For a Linear weight shaped:

```text
W: [out_features, in_features]
```

`scale_dim=1` means one scale per output row:

```text
scale[out] = max(abs(W[out, :])) / 7
q[out, in] = round(W[out, in] / scale[out])
q range = [-7, 7]
```

The real W4A16 kernel should initially use the same rule:

```text
int4 weight: signed symmetric, values [-7, 7]
scale: fp16/fp32 per output channel
zero point: none
activation: fp16 or bf16
output: fp16 or bf16
```

This keeps the first packed kernel comparable to the behavior-level fake quant
experiments.

## Packed Layout

Use two int4 values per byte.

Recommended logical storage:

```text
qweight_packed: uint8 [out_features, ceil(in_features / 2)]
scale: fp16/fp32 [out_features]
bias: fp16/bf16 [out_features] or None
```

Packing:

```text
q0 = q[:, 0::2]
q1 = q[:, 1::2]

store unsigned nibble:
  uq = q + 8

packed = uq0 | (uq1 << 4)
```

Unpacking inside Triton:

```text
low  = packed & 0x0F
high = packed >> 4
q = unsigned_nibble - 8
w = q * scale[out]
```

For odd `in_features`, pad the final nibble with zero.

## Kernel Contract

The module should expose the same user-facing behavior as `nn.Linear`:

```text
input x: [..., K]
weight W: [N, K]
bias: [N] or None
output y: [..., N]
```

For the Triton kernel, flatten the leading dimensions:

```text
M = product(x.shape[:-1])
K = in_features
N = out_features

X2d: [M, K]
Y2d: [M, N]
```

The kernel computes:

```text
Y[m, n] = sum_k X[m, k] * dequant(W[n, k]) + bias[n]
```

Then reshape:

```text
Y -> x.shape[:-1] + [N]
```

## Minimal Triton Kernel Shape

First implementation:

```text
BLOCK_M = 16 or 32
BLOCK_N = 32 or 64
BLOCK_K = 64 or 128
```

Within each program:

```text
1. load X tile: [BLOCK_M, BLOCK_K]
2. load packed W tile for N rows and K columns
3. unpack int4 nibbles to int8/int16
4. multiply by per-output scale
5. tl.dot(X_tile, W_dequant_tile)
6. accumulate in fp32
7. cast/store fp16/bf16
```

Because weight is stored as `[N, K/2]`, loading W for a `[K, N]` matmul requires
careful address math. A simple first version can load a dequantized tile shaped:

```text
W_tile: [BLOCK_K, BLOCK_N]
```

by indexing:

```text
out_offsets = n0 + arange(0, BLOCK_N)
k_offsets = k0 + arange(0, BLOCK_K)
packed_col = k_offsets // 2
nibble_sel = k_offsets % 2
```

Then:

```text
packed = qweight_packed[out_offsets, packed_col]
nibble = low_or_high(packed, nibble_sel)
q = nibble - 8
w = q * scale[out_offsets]
```

## PyTorch Module Wrapper

Add a new module, conceptually:

```python
class PackedW4A16Linear(nn.Module):
    def __init__(self, base: nn.Linear):
        self.in_features = base.in_features
        self.out_features = base.out_features
        self.register_buffer("qweight", pack_int4(base.weight))
        self.register_buffer("scale", row_scales)
        self.register_buffer("bias", base.bias.detach().clone() if base.bias else None)

    def forward(self, x):
        return triton_w4a16_linear(x, self.qweight, self.scale, self.bias)
```

Patch selected modules using the same `group_for_module` and `config_groups`
logic currently used by `patch_modules`.

## Correctness Tests

Start outside GR00T with synthetic matrices:

```text
shapes:
  M in [1, 8, 64, 256]
  K/N from real selected Linear modules
  dtype fp16 first, bf16 later
```

Compare against fake quant reference:

```python
w_q = fake_quant_symmetric(weight, 4, scale_dim=1)
ref = F.linear(x, w_q, bias)
out = PackedW4A16Linear(weight)(x)
```

Metrics:

```text
max_abs_diff
mean_abs_diff
relative_rmse
cosine similarity
```

Expected tolerance:

```text
fp16 output vs fake-quant reference:
  small numerical differences from accumulation/order are fine
  but should be much smaller than quantization error itself
```

## Benchmark Tests

Use three benchmarks:

```text
1. microbenchmark selected Linear shapes
2. one GR00T policy forward without simulator
3. inference service throughput / latency smoke
```

Report separately:

```text
weight memory:
  fp16 weight bytes vs packed int4 + scales

latency:
  F.linear fp16
  fake quant Linear
  packed W4A16 Triton Linear
```

Do not claim simulator speedup until the service path is measured end to end.

## Expected Memory Saving

For one Linear:

```text
fp16 weight memory:
  2 * out_features * in_features bytes

packed int4 weight memory:
  0.5 * out_features * in_features bytes

scale memory:
  2 or 4 * out_features bytes
```

For large Linear layers, scale overhead is tiny, so weight memory is roughly:

```text
packed W4 ~= 25% of fp16 weight memory
```

Bias remains unchanged.

## Why Triton First

Triton is the right first implementation layer because:

```text
1. It can fuse int4 unpack + dequant + matmul in one kernel.
2. It avoids TensorRT graph export complexity.
3. It is much faster to iterate than CUDA C++.
4. It is enough to prove memory and latency direction.
```

CUDA C++ can come later if Triton becomes the bottleneck.

## Implementation Stages

Recommended stages:

```text
Stage 1:
  implement pack_int4_symmetric_per_row
  implement unpack/dequant reference
  test exact roundtrip

Stage 2:
  implement Triton W4A16 matmul kernel
  test against fake quant F.linear

Stage 3:
  implement PackedW4A16Linear wrapper
  patch toy configs with same selection logic

Stage 4:
  benchmark selected Linear shapes
  benchmark offline GR00T forward

Stage 5:
  run small LIBERO smoke
  compare behavior against fake quant and FP16
```

## Open Design Choices

Important choices to settle before implementation:

```text
1. per-row scale vs groupwise scale
   start with per-row to match current fake quant

2. symmetric [-7, 7] vs full int4 range [-8, 7]
   start with current fake quant behavior, qmax=7

3. output dtype
   start with fp16, add bf16 once basic path works

4. K padding
   pad odd K to even for packing; mask in kernel

5. backward
   not needed for inference; forward-only is enough
```

## References

- Triton official matrix multiplication tutorial:
  https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html
- Triton official tutorials index:
  https://triton-lang.org/main/getting-started/tutorials/index.html
- PyTorch blog on Triton GPTQ-style fused dequantization:
  https://docs.pytorch.org/blog/accelerating-triton/
