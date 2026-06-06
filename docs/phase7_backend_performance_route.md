# Phase 7 Report: Backend Performance Route for Real `llm_dit_mlp`

## 1. Why Phase 7 Exists

Phase 6 proved that our packed W4A16 path can run, but it did not deliver speed:

```text
custom Triton W4A16:
  correct
  memory-saving
  slower than torch fp16 Linear

full llm_dit_mlp W4A16:
  large offline action drift
```

So Phase 7 moves away from the naive Triton kernel and starts evaluating mature
backend choices. The target expectation remains:

```text
Make the llm_dit_mlp route viable, eventually.
```

But we should separate two problems:

```text
1. performance backend:
   can a real INT4/FP4 backend beat fp16 on our shapes?

2. accuracy/scope:
   can the full llm_dit_mlp scope keep action drift low enough?
```

The current step focuses on performance backend selection.

## 2. Backend Environment Check

Before installing anything, the RTX 5090 GR00T environment had:

```text
torch: 2.8.0+cu128
cuda capability: sm_120
triton: 3.4.0

missing:
  torchao
  bitsandbytes
  TensorRT
  TensorRT-LLM
  CUTLASS Python package
  vLLM
  flashinfer
```

We installed:

```text
bitsandbytes==0.49.2
```

Reason: `bitsandbytes.nn.Linear4bit` is the lowest-effort mature 4-bit Linear
backend to test. It supports FP4 and NF4 weight formats through `Linear4bit` /
`Params4bit`.

Reference:

```text
https://huggingface.co/docs/bitsandbytes/reference/nn/linear4bit
```

## 3. bitsandbytes Real-Shape Benchmark

New script:

```text
toy_quantvla/phase7_bnb4_real_shape_bench.py
```

Result:

```text
toy_quantvla/results/phase7_bnb4_real_shape_bench.json
```

Settings:

```text
scope: llm_dit_mlp
real GR00T runtime shapes: 6 representative cases
quant types: fp4, nf4
compute dtype: float16
backend: bitsandbytes Linear4bit
```

### 3.1 Per-shape result

| family | M | K | N | torch fp16 | bnb FP4 | FP4 speed | FP4 rel RMSE | bnb NF4 | NF4 speed | NF4 rel RMSE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| DiT MLP | 49 | 1536 | 6144 | 0.0153 ms | 0.1073 ms | 0.143x | 0.0410 | 0.1070 ms | 0.143x | 0.0306 |
| DiT MLP | 49 | 6144 | 1536 | 0.0313 ms | 0.1072 ms | 0.292x | 0.0500 | 0.1052 ms | 0.298x | 0.0359 |
| LLM attn | 551 | 2048 | 1024 | 0.0167 ms | 0.0994 ms | 0.168x | 0.1247 | 0.0985 ms | 0.169x | 0.0927 |
| LLM attn | 551 | 2048 | 2048 | 0.0270 ms | 0.0987 ms | 0.274x | 0.1271 | 0.0996 ms | 0.271x | 0.0938 |
| LLM MLP | 551 | 2048 | 6144 | 0.0762 ms | 0.0995 ms | 0.765x | 0.1247 | 0.1016 ms | 0.750x | 0.0930 |
| LLM MLP | 551 | 6144 | 2048 | 0.0722 ms | 0.1005 ms | 0.718x | 0.1253 | 0.1003 ms | 0.719x | 0.0931 |

### 3.2 Summary

| family | cases | best speed vs torch fp16 | best rel RMSE |
|---|---:|---:|---:|
| DiT MLP | 2 | 0.220x | 0.0332 |
| LLM attention | 2 | 0.221x | 0.1099 |
| LLM MLP | 2 | 0.742x | 0.1089 |

Conclusion:

```text
bitsandbytes Linear4bit is not the performance route for our GR00T shapes.
```

It is slower than torch fp16 in every representative case. NF4 is more accurate
than FP4 here, but neither is fast enough to justify integration.

## 4. Backend Route Assessment

### 4.1 bitsandbytes

Status:

```text
tested, rejected for performance
```

Pros:

```text
easy to install
easy to wrap Linear modules
NF4 accuracy is reasonable for weight-only tests
```

Cons:

```text
slower than fp16 on all real GR00T shapes
does not match our symmetric W4 format
unlikely to make llm_dit_mlp viable
```

### 4.2 torchao

Status:

```text
not tested in current env
```

Official torchao inference docs list `Int4WeightOnlyConfig` as groupwise int4
weight-only quantization, and also list newer Blackwell-relevant formats such as
NVFP4/MXFP4. However, the same docs benchmark newer stacks than our current
environment.

Current blocker:

```text
our remote env:
  torch 2.8.0+cu128

torchao docs benchmark stack:
  torch 2.9+ or newer in several sections
  B200/Blackwell focused recipes for NVFP4/MXFP4
```

Reference:

```text
https://docs.pytorch.org/ao/stable/workflows/inference.html
```

Recommendation:

```text
Do not mutate the current GR00T env for torchao yet.
Create a separate backend env if we test torchao.
```

### 4.3 TensorRT Weight-Only Quantization

Status:

```text
promising but integration-heavy
```

TensorRT supports INT4 weight-only quantization for GEMM-style layers: high
precision activations, INT4 weights, dequantization inside the GEMM path.

References:

```text
https://docs.nvidia.com/deeplearning/tensorrt/latest/
https://docs.nvidia.com/deeplearning/tensorrt/10.13.2/inference-library/work-quantized-types.html
```

Pros:

```text
closest conceptual match to W4A16/W4BF16
NVIDIA-supported deployment backend
likely better kernels than our Python Triton prototype
```

Cons:

```text
full GR00T export is hard
partial per-Linear TensorRT engines would have too much call overhead
needs graph/subgraph capture, ONNX/Torch-TensorRT, or plugin integration
```

Recommendation:

```text
TensorRT WoQ is a serious backend candidate,
but only if we can compile whole subgraphs, not individual Linear calls.
```

### 4.4 CUTLASS / CuTe

Status:

```text
best serious custom-kernel route
```

CUTLASS officially targets high-performance GEMM, supports narrow integer
types, block-scaled FP4 formats, and Blackwell-specific GEMM paths.

Reference:

```text
https://docs.nvidia.com/cutlass/4.2.1/index.html
```

Pros:

```text
designed for peak GEMM performance
supports narrow integer data types
Blackwell support is explicit
lets us specialize exactly for our six real GR00T shapes
```

Cons:

```text
requires C++/CUDA or CuTe DSL integration
more engineering than bitsandbytes/torchao
needs packaging as a PyTorch extension or standalone op
```

Recommendation:

```text
If we want llm_dit_mlp to actually become fast,
CUTLASS/CuTe is the most credible next implementation route.
```

## 5. Updated Performance-Roadmap Decision

The mature-backend quick test gives a clean answer:

```text
bitsandbytes does not solve performance.
naive Triton does not solve performance.
```

The remaining credible routes are:

```text
Route A: CUTLASS/CuTe custom op
  specialize for the six real GR00T shapes
  target W4A16 or Blackwell-native FP4/NVFP4
  highest chance of real speedup

Route B: TensorRT WoQ subgraph export
  compile larger GR00T subgraphs with INT4 weight-only GEMM
  likely harder because GR00T is not a clean LLM-only graph

Route C: torchao in a new PyTorch backend env
  try Int4WeightOnly/NVFP4/MXFP4 if dependency stack supports 5090 well
  useful as a sanity check, but not guaranteed
```

## 6. Recommendation

Next concrete task:

```text
Build a CUTLASS/CuTe prototype benchmark outside GR00T first.
```

Minimum target:

```text
one or two real shapes:
  DiT MLP: M=49, K=1536, N=6144
  LLM MLP: M=551, K=2048, N=6144

compare:
  torch fp16
  current Triton W4A16
  bitsandbytes NF4
  CUTLASS/CuTe candidate
```

Gate:

```text
If CUTLASS cannot beat torch fp16 on these shapes,
then W4A16 is mainly a memory-saving route, not a speed route.

If CUTLASS beats torch fp16,
then we can build a PyTorch extension and revisit llm_dit_mlp integration.
```

We should not run actual rollouts for full `llm_dit_mlp` until a backend is both:

```text
1. faster than fp16 on real shapes
2. accurate enough in offline forward drift
```

