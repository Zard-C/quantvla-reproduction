# Phase 6 Report: Kernel Performance and Patch Scope Contraction

## 1. Purpose

After the initial W4A16 kernel and full `llm_dit_mlp` offline forward smoke, we
needed to answer two engineering questions before considering rollout:

```text
1. Can simple Triton tile tuning make the packed W4A16 kernel competitive?
2. Can a narrower patch scope reduce action drift enough to become a rollout candidate?
```

This report answers both.

## 2. Added Scope Controls

Phase 6 now has a dedicated scope helper:

```text
toy_quantvla/phase6_w4a16_scopes.py
```

Supported scopes:

| scope | meaning |
|---|---|
| `dit_mlp_only` | DiT feed-forward Linear modules only |
| `llm_attn_only` | LLM self-attention Linear modules only |
| `llm_mlp_only` | LLM MLP Linear modules only |
| `llm_selected` | LLM attention plus LLM MLP |
| `llm_mlp_dit_mlp` | LLM MLP plus DiT MLP, excluding LLM attention |
| `llm_dit_mlp` | Phase 5-compatible full selected scope |

This keeps Phase 6 deployment experiments separate from the older Phase 5
`config_groups` semantics.

## 3. Scope Drift Sweep

Settings:

```text
denoising steps: 1
synthetic observations: 3
kernel path: PackedW4A16Linear with Triton
```

Result files:

```text
toy_quantvla/results/phase6_w4a16_forward_dit_mlp_only.json
toy_quantvla/results/phase6_w4a16_forward_llm_attn_only.json
toy_quantvla/results/phase6_w4a16_forward_llm_mlp_only.json
toy_quantvla/results/phase6_w4a16_forward_llm_mlp_dit_mlp.json
toy_quantvla/results/phase6_w4a16_forward_llm_dit_mlp.json
```

| scope | modules | fp16 weight | packed+scale ratio | mean cosine | mean rel RMSE | max abs diff |
|---|---:|---:|---:|---:|---:|---:|
| `dit_mlp_only` | 32 | 0.604 GB | 25.08% | 0.9967 | 0.0808 | 0.1509 |
| `llm_attn_only` | 48 | 0.302 GB | 25.10% | 0.9612 | 0.2740 | 0.8340 |
| `llm_mlp_only` | 36 | 0.906 GB | 25.08% | 0.9232 | 0.3593 | 0.9316 |
| `llm_mlp_dit_mlp` | 68 | 1.510 GB | 25.08% | 0.9362 | 0.3410 | 0.9238 |
| `llm_dit_mlp` | 116 | 1.812 GB | 25.08% | 0.9258 | 0.3781 | 0.9570 |

The result is decisive:

```text
dit_mlp_only is the only W4A16 scope with small offline action drift.

LLM MLP is the largest drift source.
LLM attention is also risky, though less severe than LLM MLP.
Full llm_dit_mlp is not a good rollout candidate yet.
```

## 4. Kernel Tuning Sweep

The real-shape benchmark was rerun with a broader tile list:

```text
BLOCK_M in {16, 32, 64}
BLOCK_N in {32, 64, 128, 256}
BLOCK_K in {64, 128, 256}
num_warps in {4, 8} for selected larger tiles
```

Result file:

```text
toy_quantvla/results/phase6_w4a16_real_shape_bench_tuned.json
```

Best result per real GR00T shape:

| family | M | K | N | torch fp16 | best W4A16 | speed vs torch | best config |
|---|---:|---:|---:|---:|---:|---:|---|
| DiT MLP | 49 | 1536 | 6144 | 0.0197 ms | 0.0412 ms | 0.477x | bm16_bn64_bk64_w4 |
| DiT MLP | 49 | 6144 | 1536 | 0.0313 ms | 0.0784 ms | 0.399x | bm32_bn64_bk64_w4 |
| LLM attn | 551 | 2048 | 1024 | 0.0202 ms | 0.0396 ms | 0.511x | bm64_bn64_bk64_w4 |
| LLM attn | 551 | 2048 | 2048 | 0.0269 ms | 0.0562 ms | 0.480x | bm64_bn128_bk64_w8 |
| LLM MLP | 551 | 2048 | 6144 | 0.0779 ms | 0.1220 ms | 0.638x | bm64_bn64_bk64_w4 |
| LLM MLP | 551 | 6144 | 2048 | 0.0731 ms | 0.1223 ms | 0.598x | bm64_bn64_bk64_w4 |

Conclusion:

```text
Simple tile tuning did not make this kernel faster than torch fp16 Linear.
The current packed W4A16 kernel remains correctness-first, not performance-ready.
```

The likely reason is that the current layout and kernel still pay a substantial
unpack/dequant cost before tensor-core dot. cuBLAS fp16 is very hard to beat on
these shapes unless the int4 path has a more optimized schedule or uses a more
specialized layout.

## 5. Rollout Gate

Current gate decision:

```text
Do not run full llm_dit_mlp W4A16 rollout.
Do not run llm_mlp_dit_mlp rollout yet.
Do not use kernel speed as the justification for rollout yet.
```

The only reasonable rollout candidate from this sweep is:

```text
dit_mlp_only W4A16
```

Why:

```text
1. It patches 32 modules.
2. It compresses about 0.604 GB of fp16 weights down to about 25%.
3. It has low offline action drift:
   mean cosine 0.9967
   mean relative RMSE 0.0808
4. It avoids LLM-side drift sources.
```

But even for `dit_mlp_only`, rollout should be treated as behavior smoke, not as
an acceleration claim.

## 6. Recommended Next Step

The next step should be:

```text
1. Add a W4A16 inference-service path with --scope dit_mlp_only.
2. Run an offline real-data get_action comparison first if convenient.
3. Then run a tiny LIBERO smoke, e.g. 5-10 targeted episodes.
4. Keep FP16 and fake-quant baselines separate from packed-W4A16 results.
```

Kernel performance work should continue in parallel, but the current simple
Triton kernel is not enough to claim speedup.

The engineering conclusion is now clean:

```text
Patch-scope contraction succeeded.
Kernel performance tuning did not yet succeed.
Rollout should only proceed with dit_mlp_only and only as a behavior smoke.
```

