# QuantVLA Phase 2.5 GR00T Weight Statistics

Date: 2026-06-04

Model path:

```text
/root/autodl-tmp/models/gr00t-n1.5-libero-long-posttrain
```

Input files:

```text
model-00001-of-00002.safetensors 4999367032 bytes
model-00002-of-00002.safetensors 2586705312 bytes
model.safetensors.index.json 104606 bytes
```

Boundary: this phase only reads safetensors weights and metadata. It does not start the GR00T inference server and does not run LIBERO.

## Selector Counts

Using the Phase 1 selector:

```text
LLM selected:
  backbone.eagle_model.language_model.model.layers.*.{self_attn q/k/v/o, mlp gate/up/down}.weight

DiT MLP selected:
  action_head.model.transformer_blocks.*.ff.net.{0.proj,2}.weight

DiT attention excluded:
  action_head.model.transformer_blocks.*.attn1.{to_q,to_k,to_v,to_out.0}.weight
```

Observed counts:

| group | count |
|---|---:|
| llm_selected | 84 |
| dit_mlp_selected | 32 |
| dit_attention_excluded | 64 |

Interpretation: this GR00T N1.5 LIBERO-long checkpoint has 12 LLM layers and 16 DiT transformer blocks. The selected layout is therefore 84 LLM linear weights plus 32 DiT MLP weights, while 64 DiT attention projection weights remain floating point.

## Group Summary

| group | std mean | max_abs mean | max_abs max | p99.9 mean | tail ratio mean | tail ratio max | per-output p99 mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| dit_attention_excluded | 0.02945 | 0.25428 | 0.85369 | 0.10454 | 2.40119 | 7.17244 | 0.16036 |
| dit_mlp_selected | 0.02951 | 0.58427 | 1.56504 | 0.11195 | 5.30308 | 13.25232 | 0.26307 |
| llm_selected | 0.03370 | 0.53112 | 0.97266 | 0.13498 | 4.01146 | 8.62338 | 0.28381 |

Definitions:

- `tail ratio = max_abs / p99.9_abs` per weight tensor.
- `per-output p99` is the p99 of per-output-channel max absolute values.

## Highest Tail-Ratio Layers

| group | layer | shape | max_abs | p99.9_abs | tail ratio |
|---|---|---:|---:|---:|---:|
| dit_mlp_selected | action_head.model.transformer_blocks.0.ff.net.2.weight | [1536, 6144] | 1.5000 | 0.1132 | 13.252 |
| dit_mlp_selected | action_head.model.transformer_blocks.1.ff.net.2.weight | [1536, 6144] | 1.2154 | 0.0925 | 13.146 |
| dit_mlp_selected | action_head.model.transformer_blocks.2.ff.net.2.weight | [1536, 6144] | 1.0676 | 0.0945 | 11.299 |
| dit_mlp_selected | action_head.model.transformer_blocks.3.ff.net.2.weight | [1536, 6144] | 1.0505 | 0.0938 | 11.195 |
| dit_mlp_selected | action_head.model.transformer_blocks.15.ff.net.2.weight | [1536, 6144] | 1.5650 | 0.1544 | 10.133 |
| llm_selected | backbone.eagle_model.language_model.model.layers.7.mlp.down_proj.weight | [2048, 6144] | 0.9727 | 0.1128 | 8.623 |
| dit_mlp_selected | action_head.model.transformer_blocks.8.ff.net.2.weight | [1536, 6144] | 0.8624 | 0.1093 | 7.893 |
| llm_selected | backbone.eagle_model.language_model.model.layers.2.mlp.down_proj.weight | [2048, 6144] | 0.8516 | 0.1094 | 7.786 |
| dit_mlp_selected | action_head.model.transformer_blocks.15.ff.net.0.proj.weight | [6144, 1536] | 0.8012 | 0.1060 | 7.561 |
| dit_mlp_selected | action_head.model.transformer_blocks.0.ff.net.0.proj.weight | [6144, 1536] | 0.7076 | 0.0950 | 7.445 |
| llm_selected | backbone.eagle_model.language_model.model.layers.6.mlp.down_proj.weight | [2048, 6144] | 0.8594 | 0.1157 | 7.426 |
| dit_attention_excluded | action_head.model.transformer_blocks.1.attn1.to_out.0.weight | [1536, 1536] | 0.8537 | 0.1190 | 7.172 |

## Interpretation

The real GR00T weights support the outlier-channel premise, especially in DiT MLP and LLM MLP down-projection weights. The DiT MLP group has the strongest tail behavior among the three groups: tail-ratio mean `5.30` and max `13.25`.

However, the first synthetic VLA-like toy distribution was much harsher in absolute weight scale. Its sampled `vla_like_dit_mlp` weight max_abs was about `26.2`, while the real checkpoint max is about `1.56`. Therefore, large absolute toy MSE values should not be interpreted literally. Normalized metrics remain the correct primary signal.

The real weights also clarify the selective layout: DiT MLP weights are actually more outlier-heavy than DiT attention weights, so smoothing and W4A8 robustness matter there. DiT attention is still kept floating point in QuantVLA not because its weights have the worst static outliers, but because Q/K/V/O determine attention temperature and residual injection, making them sensitive to upstream activation drift.

## Status

`toy_quantvla/results/weight_stats.json` was generated from the downloaded checkpoint. After rerunning Phase 2 toy experiments, `empirical_weight_stats_available=True`, `normalized_metric_gate=True`, and `phase3_ready=True` in `toy_quantvla/results/summary.json`.

This readiness means: proceed to Phase 3 offline GR00T module inspection and fake-quant forward checks. It does not mean we should start LIBERO evaluation yet.
