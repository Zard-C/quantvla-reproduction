# QuantVLA Phase 3 Offline Weight Quantization

Boundary: this analysis reads safetensors weights only. It does not instantiate GR00T, start an inference server, or run LIBERO.

Model: `/root/autodl-tmp/models/gr00t-n1.5-libero-long-posttrain`

## Selector Counts

| group | layers | params | fp16 MiB | ideal W4 MiB | ideal saving | NMSE mean | NMSE max | cosine mean | tail ratio mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| llm_selected | 84 | 603979776 | 1152.00 | 288.00 | 75.0% | 0.0298665 | 0.0442779 | 0.985522 | 4.01146 |
| dit_mlp_selected | 32 | 301989888 | 576.00 | 144.00 | 75.0% | 0.0372728 | 0.18975 | 0.98229 | 5.30308 |
| dit_attention_excluded | 64 | 163577856 | 312.00 | 78.00 | 75.0% | 0.0235331 | 0.0276277 | 0.988488 | 2.40119 |

## Worst Selected Layers By W4 NMSE

| group | layer | shape | NMSE | cosine | max_abs | tail ratio |
|---|---|---:|---:|---:|---:|---:|
| dit_mlp_selected | `action_head.model.transformer_blocks.15.ff.net.2.weight` | [1536, 6144] | 0.18975 | 0.911597 | 1.56504 | 10.1333 |
| dit_mlp_selected | `action_head.model.transformer_blocks.0.ff.net.2.weight` | [1536, 6144] | 0.0722104 | 0.965583 | 1.49997 | 13.2523 |
| dit_mlp_selected | `action_head.model.transformer_blocks.14.ff.net.2.weight` | [1536, 6144] | 0.0498473 | 0.976426 | 0.473003 | 3.48348 |
| dit_mlp_selected | `action_head.model.transformer_blocks.1.ff.net.2.weight` | [1536, 6144] | 0.0483507 | 0.976886 | 1.21535 | 13.1456 |
| llm_selected | `backbone.eagle_model.language_model.model.layers.1.mlp.down_proj.weight` | [2048, 6144] | 0.0442779 | 0.979263 | 0.605469 | 5.36797 |
| llm_selected | `backbone.eagle_model.language_model.model.layers.10.mlp.down_proj.weight` | [2048, 6144] | 0.0436106 | 0.97922 | 0.695312 | 5.67331 |
| dit_mlp_selected | `action_head.model.transformer_blocks.5.ff.net.2.weight` | [1536, 6144] | 0.0416309 | 0.980174 | 0.591236 | 5.85248 |
| dit_mlp_selected | `action_head.model.transformer_blocks.2.ff.net.2.weight` | [1536, 6144] | 0.0415779 | 0.980133 | 1.06762 | 11.2988 |
| dit_mlp_selected | `action_head.model.transformer_blocks.3.ff.net.2.weight` | [1536, 6144] | 0.0414861 | 0.980118 | 1.05048 | 11.1953 |
| llm_selected | `backbone.eagle_model.language_model.model.layers.2.mlp.down_proj.weight` | [2048, 6144] | 0.0408578 | 0.980732 | 0.851562 | 7.78571 |
| llm_selected | `backbone.eagle_model.language_model.model.layers.11.self_attn.o_proj.weight` | [2048, 2048] | 0.0402699 | 0.98028 | 0.664062 | 3.95349 |
| dit_mlp_selected | `action_head.model.transformer_blocks.11.ff.net.2.weight` | [1536, 6144] | 0.0401477 | 0.98084 | 0.402507 | 3.23137 |
| llm_selected | `backbone.eagle_model.language_model.model.layers.9.self_attn.o_proj.weight` | [2048, 2048] | 0.0395313 | 0.9808 | 0.542969 | 3.63399 |
| dit_mlp_selected | `action_head.model.transformer_blocks.8.ff.net.2.weight` | [1536, 6144] | 0.0387575 | 0.981412 | 0.862357 | 7.8926 |
| llm_selected | `backbone.eagle_model.language_model.model.layers.6.mlp.down_proj.weight` | [2048, 6144] | 0.0387385 | 0.98147 | 0.859375 | 7.42616 |
| dit_mlp_selected | `action_head.model.transformer_blocks.13.ff.net.2.weight` | [1536, 6144] | 0.0384694 | 0.981699 | 0.588874 | 4.74815 |
| llm_selected | `backbone.eagle_model.language_model.model.layers.5.mlp.down_proj.weight` | [2048, 6144] | 0.0384086 | 0.981336 | 0.660156 | 5.77778 |
| llm_selected | `backbone.eagle_model.language_model.model.layers.9.mlp.down_proj.weight` | [2048, 6144] | 0.0378951 | 0.981924 | 0.789062 | 6.59592 |
| llm_selected | `backbone.eagle_model.language_model.model.layers.11.mlp.down_proj.weight` | [2048, 6144] | 0.0376573 | 0.98213 | 0.792969 | 6.36863 |
| llm_selected | `backbone.eagle_model.language_model.model.layers.8.self_attn.k_proj.weight` | [1024, 2048] | 0.0371807 | 0.981847 | 0.84375 | 5.72185 |

## Interpretation

- The Phase 1 selector exactly matches the checkpoint naming for LLM linears, DiT MLP linears, and excluded DiT attention projections.
- W4 per-output-channel weight-only fake quantization has low static weight NMSE for the selected groups. This does not prove activation quantization is safe, but it supports moving to offline activation/forward checks.
- DiT attention weights are listed for comparison only. They remain excluded in the main QuantVLA layout because their Q/K/V/O outputs control attention temperature and residual energy.
- The memory table is an ideal tensor-storage estimate. Runtime memory depends on activation buffers, scales, packing format, kernels, and model framework overhead.
