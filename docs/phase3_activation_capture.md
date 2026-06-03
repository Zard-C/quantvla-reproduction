# QuantVLA Phase 3 Activation Capture

Boundary: this is a synthetic-input activation probe through the official GR00T policy path. It is not a LIBERO task evaluation.

## Run

- Model: `/root/autodl-tmp/models/gr00t-n1.5-libero-long-posttrain`
- Denoising steps: `1`
- Observations: `3`
- Synthetic variants: `zero, midgray, noise`
- Percentile for clipping probe: `0.999`
- Forward seconds: `1.526`

## Group Summary

| group | modules | called | input rms | input absmax | input p99.9 | input tail | W8 NMSE absmax | W8 NMSE p99.9 | p99.9 clip | output rms | output absmax |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| llm_selected | 84 | 84 | 0.532848 | 274.23 | 3.99343 | 320.959 | 0.028847 | 0.0524129 | 0.000991304 | 1.03361 | 237.341 |
| dit_mlp_selected | 32 | 32 | 0.648948 | 18.3208 | 7.85954 | 2.67036 | 0.00337728 | 0.0383361 | 0.00100044 | 3.17099 | 66.3561 |
| dit_attention_excluded | 64 | 64 | 0.780782 | 35.4735 | 5.27478 | 4.09744 | 0.00810065 | 0.0444129 | 0.0010033 | 1.33992 | 11.877 |

## Worst Input W8 NMSE With p99.9 Scale

| group | module | NMSE | max |
|---|---|---:|---:|
| llm_selected | `backbone.eagle_model.language_model.model.layers.2.mlp.down_proj` | 0.999824 | 0.999826 |
| dit_mlp_selected | `action_head.model.transformer_blocks.0.ff.net.2` | 0.291011 | 0.30103 |
| llm_selected | `backbone.eagle_model.language_model.model.layers.1.mlp.down_proj` | 0.250644 | 0.253811 |
| llm_selected | `backbone.eagle_model.language_model.model.layers.9.mlp.down_proj` | 0.243383 | 0.257348 |
| llm_selected | `backbone.eagle_model.language_model.model.layers.7.mlp.down_proj` | 0.216876 | 0.224972 |
| dit_mlp_selected | `action_head.model.transformer_blocks.15.ff.net.2` | 0.199207 | 0.215348 |
| llm_selected | `backbone.eagle_model.language_model.model.layers.10.mlp.down_proj` | 0.197822 | 0.209053 |
| dit_attention_excluded | `action_head.model.transformer_blocks.0.attn1.to_k` | 0.14958 | 0.165331 |
| dit_attention_excluded | `action_head.model.transformer_blocks.0.attn1.to_v` | 0.14958 | 0.165331 |
| dit_attention_excluded | `action_head.model.transformer_blocks.10.attn1.to_k` | 0.14958 | 0.165331 |
| dit_attention_excluded | `action_head.model.transformer_blocks.10.attn1.to_v` | 0.14958 | 0.165331 |
| dit_attention_excluded | `action_head.model.transformer_blocks.12.attn1.to_k` | 0.14958 | 0.165331 |
| dit_attention_excluded | `action_head.model.transformer_blocks.12.attn1.to_v` | 0.14958 | 0.165331 |
| dit_attention_excluded | `action_head.model.transformer_blocks.14.attn1.to_k` | 0.14958 | 0.165331 |
| dit_attention_excluded | `action_head.model.transformer_blocks.14.attn1.to_v` | 0.14958 | 0.165331 |
| dit_attention_excluded | `action_head.model.transformer_blocks.2.attn1.to_k` | 0.14958 | 0.165331 |
| dit_attention_excluded | `action_head.model.transformer_blocks.2.attn1.to_v` | 0.14958 | 0.165331 |
| dit_attention_excluded | `action_head.model.transformer_blocks.4.attn1.to_k` | 0.14958 | 0.165331 |
| dit_attention_excluded | `action_head.model.transformer_blocks.4.attn1.to_v` | 0.14958 | 0.165331 |
| dit_attention_excluded | `action_head.model.transformer_blocks.6.attn1.to_k` | 0.14958 | 0.165331 |

## Highest Input Tail Ratio

| group | module | tail ratio | max |
|---|---|---:|---:|
| llm_selected | `backbone.eagle_model.language_model.model.layers.2.mlp.down_proj` | 26396.1 | 26693.9 |
| llm_selected | `backbone.eagle_model.language_model.model.layers.1.mlp.down_proj` | 60.3507 | 63.426 |
| llm_selected | `backbone.eagle_model.language_model.model.layers.10.mlp.down_proj` | 48.8717 | 50.4041 |
| llm_selected | `backbone.eagle_model.language_model.model.layers.7.mlp.down_proj` | 45.9362 | 47.04 |
| llm_selected | `backbone.eagle_model.language_model.model.layers.9.mlp.down_proj` | 36.1691 | 37.0196 |
| llm_selected | `backbone.eagle_model.language_model.model.layers.4.mlp.down_proj` | 22.6563 | 27.0968 |
| llm_selected | `backbone.eagle_model.language_model.model.layers.11.mlp.down_proj` | 21.7069 | 21.8716 |
| llm_selected | `backbone.eagle_model.language_model.model.layers.5.mlp.down_proj` | 16.8983 | 21.8562 |
| llm_selected | `backbone.eagle_model.language_model.model.layers.3.mlp.down_proj` | 14.7016 | 15.9422 |
| llm_selected | `backbone.eagle_model.language_model.model.layers.0.mlp.down_proj` | 13.88 | 14.2785 |
| llm_selected | `backbone.eagle_model.language_model.model.layers.2.mlp.gate_proj` | 11.7173 | 12.1739 |
| llm_selected | `backbone.eagle_model.language_model.model.layers.2.mlp.up_proj` | 11.7173 | 12.1739 |
| llm_selected | `backbone.eagle_model.language_model.model.layers.6.mlp.down_proj` | 10.729 | 11.0769 |
| dit_attention_excluded | `action_head.model.transformer_blocks.0.attn1.to_k` | 10.4798 | 11.2456 |
| dit_attention_excluded | `action_head.model.transformer_blocks.0.attn1.to_v` | 10.4798 | 11.2456 |
| dit_attention_excluded | `action_head.model.transformer_blocks.10.attn1.to_k` | 10.4798 | 11.2456 |
| dit_attention_excluded | `action_head.model.transformer_blocks.10.attn1.to_v` | 10.4798 | 11.2456 |
| dit_attention_excluded | `action_head.model.transformer_blocks.12.attn1.to_k` | 10.4798 | 11.2456 |
| dit_attention_excluded | `action_head.model.transformer_blocks.12.attn1.to_v` | 10.4798 | 11.2456 |
| dit_attention_excluded | `action_head.model.transformer_blocks.14.attn1.to_k` | 10.4798 | 11.2456 |

## Interpretation

- All three selector groups are exercised by the synthetic GR00T forward path.
- `w8_nmse_absmax_scale` estimates conservative per-tensor int8 activation fake quantization error.
- `w8_nmse_p999_scale` estimates percentile-clipped int8 activation fake quantization error and reports the corresponding clip fraction.
- Very large LLM tail ratios are concentrated in MLP down-projection activations. This supports the need for smoothing/calibration, but these numbers are still from synthetic visual inputs and should be checked with real LIBERO observations before making benchmark claims.
- This establishes activation ranges for the teacher model. The next step is to insert fake-quantized selected linears and compare teacher/student outputs, then calibrate ATM/OHB on the same hook scaffold.
