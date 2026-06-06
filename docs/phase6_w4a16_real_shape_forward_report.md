# Phase 6 Report: Real GR00T Shapes and W4A16 Forward Smoke

## 1. What This Step Adds

The previous Phase 6 smoke proved that the Triton W4A16 kernel was correct on
synthetic matrix shapes. This step connects it more tightly to GR00T:

```text
1. capture real runtime Linear shapes from the GR00T get_action path
2. benchmark W4A16 on representative selected-module shapes
3. replace selected GR00T Linear modules with PackedW4A16Linear
4. run offline get_action and measure action drift
```

This is still offline engineering validation, not simulator rollout.

Follow-up status: patch-scope contraction and expanded tile tuning are reported
in `docs/phase6_w4a16_scope_perf_report.md`.

## 2. New Files

```text
toy_quantvla/phase6_w4a16_real_shape_bench.py
  Loads GR00T, hooks selected Linear inputs, captures runtime M/K/N,
  and benchmarks real module weights over a small tile-config sweep.

toy_quantvla/phase6_w4a16_forward_smoke.py
  Replaces selected nn.Linear modules with PackedW4A16Linear and compares
  teacher/student get_action outputs under matched seeds.
```

The Triton kernel now supports mixed activation dtypes seen in GR00T:

```text
LLM path:
  bf16 activations

DiT path:
  fp32 activations in some feed-forward modules

microbenchmark path:
  fp16 activations
```

## 3. Real Shape Capture

Run:

```text
toy_quantvla/results/phase6_w4a16_real_shape_bench.json
```

The script selected and captured all expected modules:

```text
selected modules: 116
captured modules: 116
representative runtime shapes: 6
```

The 6 representative shapes are:

| group | runtime M | K | N | module family |
|---|---:|---:|---:|---|
| dit_mlp_selected | 49 | 1536 | 6144 | DiT FF up/proj |
| dit_mlp_selected | 49 | 6144 | 1536 | DiT FF down |
| llm_selected | 551 | 2048 | 1024 | LLM attention k/v |
| llm_selected | 551 | 2048 | 2048 | LLM attention q/o |
| llm_selected | 551 | 2048 | 6144 | LLM MLP gate/up |
| llm_selected | 551 | 6144 | 2048 | LLM MLP down |

This confirms that `llm_dit_mlp` is not only DiT MLP. It includes the existing
`llm_selected` regex, which covers LLM attention and MLP Linear layers, plus
DiT feed-forward Linear layers.

## 4. Real Shape Benchmark

Each case uses a representative real GR00T weight tensor and random activation
with the captured runtime shape. The script sweeps a small set of Triton tile
configs and reports the best one.

| group | M | K | N | torch fp16 | best W4A16 | speed vs torch | best config | relative RMSE |
|---|---:|---:|---:|---:|---:|---:|---|---:|
| dit_mlp_selected | 49 | 1536 | 6144 | 0.0146 ms | 0.0411 ms | 0.356x | bm64_bn64_bk64_w4 | 1.76e-4 |
| dit_mlp_selected | 49 | 6144 | 1536 | 0.0311 ms | 0.0502 ms | 0.619x | bm16_bn32_bk64_w4 | 3.08e-4 |
| llm_selected | 551 | 2048 | 1024 | 0.0200 ms | 0.0404 ms | 0.496x | bm32_bn32_bk64_w4 | 1.15e-5 |
| llm_selected | 551 | 2048 | 2048 | 0.0271 ms | 0.0708 ms | 0.382x | bm64_bn64_bk128_w4 | 1.19e-5 |
| llm_selected | 551 | 2048 | 6144 | 0.0762 ms | 0.1220 ms | 0.624x | bm64_bn64_bk64_w4 | 1.21e-5 |
| llm_selected | 551 | 6144 | 2048 | 0.0722 ms | 0.1232 ms | 0.586x | bm64_bn64_bk64_w4 | 1.50e-5 |

Interpretation:

```text
The current W4A16 kernel is correct on real GR00T shapes,
but it is still slower than torch fp16 Linear.
```

The kernel is now an engineering foothold, not yet a performance win.

## 5. Memory Result

Replacing the 116 selected modules gives:

| scope | modules | fp16 weight bytes | packed bytes | scale bytes | bias bytes | packed+scale ratio |
|---|---:|---:|---:|---:|---:|---:|
| all selected | 116 | 1,811,939,328 | 452,984,832 | 1,474,560 | 245,760 | 25.08% |
| dit_mlp_selected | 32 | 603,979,776 | 150,994,944 | 491,520 | 245,760 | 25.08% |
| llm_selected | 84 | 1,207,959,552 | 301,989,888 | 983,040 | 0 | 25.08% |

So the selected Linear weight payload drops from about:

```text
1.81 GB fp16 weight
```

to:

```text
452.98 MB packed int4 weight
+ 1.47 MB fp32 scale
```

This confirms the expected memory direction. Bias is tiny here.

## 6. Offline Forward Smoke

Run:

```text
toy_quantvla/results/phase6_w4a16_forward_smoke.json
```

Settings:

```text
config: llm_dit_mlp
patched modules: 116
denoising steps: 1
synthetic observations: 3
prefer_triton: true
```

Aggregate action drift:

| metric | mean | max | min |
|---|---:|---:|---:|
| NMSE | 0.1716 | 0.3805 | 0.0595 |
| relative RMSE | 0.3781 | 0.6169 | 0.2440 |
| cosine | 0.9258 | 0.9843 | 0.8310 |
| max abs diff | 0.5034 | 0.9570 | 0.2329 |

This proves:

```text
PackedW4A16Linear can be installed into GR00T and complete get_action.
```

It does not yet prove:

```text
The packed W4A16 path is close enough for simulator rollout.
```

The drift is non-trivial, especially on the worst synthetic observation.

## 7. Current Interpretation

Phase 6 has crossed a real boundary:

```text
Before:
  fake quant only

Now:
  real packed int4 weights
  fused dequant matmul
  selected GR00T modules replaceable
  offline get_action runs end to end
```

But the current implementation has two open problems:

```text
1. Performance:
   W4A16 is still slower than torch fp16 on real GR00T shapes.

2. Behavior drift:
   W4-only packed replacement over the full llm_dit_mlp selected set has
   noticeable action drift on synthetic get_action.
```

So the next step should not be LIBERO rollout yet. The next step should be a
more surgical engineering loop.

## 8. Recommended Next Step

Recommended order:

```text
1. Optimize kernel performance:
   - better tile configs
   - specialize for M=49 DiT and M=551 LLM
   - try larger BLOCK_K and BLOCK_N
   - consider separate kernels for fp16/bf16/fp32 paths

2. Compare real W4A16 vs fake W4 weight-only reference:
   - isolate kernel numerical error from quantization error
   - do this per module group

3. Add selective patch scopes:
   - dit_mlp_only W4A16
   - llm_mlp_only if we split LLM attention away
   - llm_selected full scope

4. Only after forward drift is acceptable:
   - one tiny LIBERO smoke
```

The most important engineering lesson from this step is that the deployment
problem is now concrete: we have a working path, and the remaining work is
performance tuning plus scope selection.
