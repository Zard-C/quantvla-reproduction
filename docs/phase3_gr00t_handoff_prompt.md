# QuantVLA Phase 3 GR00T Handoff Prompt

You are working in `/root/autodl-tmp/quantvla-reproduction`.

Do not commit model weights, datasets, generated checkpoints, or large tensor dumps.

Read first:

```bash
sed -n '1,260p' docs/phase1_local_analysis.md
sed -n '1,260p' docs/phase2_toy_results.md
```

Phase 2 toy validation passed on 2026-06-04. The important decisions are:

- `alpha = std_teacher / std_student` is applied by multiplying logits or `Q`.
- `beta = rms_teacher / rms_student` is applied by multiplying the post-output attention projection.
- Use log-domain clamp `0.30` and neutral band `0.03` unless a GR00T-specific calibration sweep proves otherwise.
- Quantize all selected LLM linear layers and only DiT MLP linear layers.
- Leave DiT attention `Q/K/V/O` projections floating point in the main layout.

## Objective

Move from toy validation to a clean GR00T N1.5 implementation path for QuantVLA-style post-training quantization. Start with weight-only inspection and fake-quant forward validation before any LIBERO evaluation.

## Required Local Discovery

1. Locate the GR00T N1.5 model code, checkpoint path, and any existing official QuantVLA or GR00T checkout.
2. Confirm whether LIBERO assets and task configs are already available locally.
3. Record exact paths in a new doc, but do not copy weights into this repo.
4. Dump small JSON weight statistics only, using the schema in `toy_quantvla/results/weight_stats_schema.json`.

Suggested first command after locating a checkpoint:

```bash
python toy_quantvla/extract_weight_stats.py --help
```

## Implementation Targets

Layer selector:

```text
include:
  backbone.eagle_model.language_model.*.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)
  action_head.model.transformer_blocks.*.ff.net.(0.proj|2)

exclude:
  vision, radio, norm, ln, layernorm, embed, lm_head
  action_head.model.transformer_blocks.*.attn1.*
```

Quantization:

- Start with fake W4A8 quantization for correctness.
- Weights: signed symmetric 4-bit, per-output-channel.
- Activations: signed symmetric 8-bit, percentile clipping at `99.9`.
- SmoothQuant-style migration strength: `0.15`.
- Calibration batches: `32`.
- Keep fake quantization and real low-bit kernels clearly separated in reporting.

ATM:

- Collect teacher and student DiT attention logits per layer/head on unlabeled calibration observations.
- Compute `alpha = std_teacher / std_student`.
- Clamp in log domain with `log_clamp = 0.30`.
- Neutralize if `abs(log_alpha) < 0.03`.
- Apply by multiplying logits or `Q`.

OHB:

- Collect teacher and student post-output attention tensors per DiT attention layer.
- Compute `beta = rms_teacher / rms_student`.
- Clamp in log domain with `log_clamp = 0.30`.
- Neutralize if `abs(log_beta) < 0.03`.
- Apply by multiplying post-output tensors before residual addition.

## Execution Order

1. Weight-stat dump only.
2. Build and unit-test the GR00T module selector.
3. Add fake-quant wrappers for selected linears.
4. Run a no-server forward smoke test on a tiny local batch.
5. Collect calibration tensors for 4, 8, 32, and 128 samples if available.
6. Save `alpha` and `beta` calibration JSON with layer names, shapes, means, min/max, neutral counts, and clamp-hit counts.
7. Compare FP16 teacher, naive W4A8, selective W4A8, selective W4A8+ATM, and selective W4A8+ATM+OHB on offline forward metrics.
8. Only after the above is stable, decide whether to start LIBERO evaluation.

## Required Outputs

Create a Phase 3 report under `docs/` with:

- Exact commands run.
- Python, PyTorch, CUDA, GPU, model path, checkpoint identifier.
- Layer selector counts for LLM, full DiT, DiT MLP, and excluded DiT attention projections.
- Weight-stat JSON path and summary.
- Calibration JSON path and summary.
- Offline metric tables.
- Any expectation failures or ambiguous layer names.

Do not start the GR00T inference server or LIBERO evaluation until the offline forward and calibration checks are documented.

