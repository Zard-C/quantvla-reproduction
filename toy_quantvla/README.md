# QuantVLA Toy Experiments

This workspace validates the QuantVLA mechanics before starting GR00T or LIBERO.

It is intentionally self-contained:

- `quant.py`: W4A8 fake quantization, SmoothQuant-style linear smoothing, metrics.
- `distributions.py`: standard-normal and VLA-like synthetic weight/activation presets.
- `toy_attention.py`: small attention and transformer block utilities.
- `run_toy_experiments.py`: runs the required Phase 2 experiments and writes reports.
- `extract_weight_stats.py`: optional checkpoint statistics dumper. It loads weights only.

## Quantization Config Glossary

- `llm_only`: quantizes selected LLM `nn.Linear` layers, including self-attention projections `q_proj`, `k_proj`, `v_proj`, `o_proj`, and MLP projections `gate_proj`, `up_proj`, `down_proj`.
- `dit_mlp_only`: quantizes only DiT action-head feed-forward/MLP linear layers matching `action_head.model.transformer_blocks.*.ff.net.(0.proj|2)`.
- `llm_dit_mlp`: union of `llm_selected` and `dit_mlp_selected`; on the current GR00T N1.5 LIBERO checkpoint this is `84 + 32 = 116` quantized modules.
- DiT attention projections matching `action_head.model.transformer_blocks.*.attn1.(to_q|to_k|to_v|to_out.0)` are intentionally excluded from quantization and remain floating point.
- ATM/OHB are calibration and balancing operations applied to DiT attention processors. They do not mean DiT attention weights are quantized.
- `identity` mode installs the same custom DiT attention processor with `alpha = 1` and `beta = 1`; it is only a processor-replacement control.

Run:

```bash
python toy_quantvla/run_toy_experiments.py
```

Required outputs are written under `toy_quantvla/results/`:

- `summary.json`
- `summary.md`
- `atm_direction.csv`
- `ohb_direction.csv`
- `distribution_sensitivity.csv`
- `weight_stats_schema.json`

The toy distributions include layerwise log-normal scales, channelwise log-normal scales,
sparse outlier rows/columns, heavy-tailed activations, and simulated upstream student drift.
