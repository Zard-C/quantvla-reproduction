# QuantVLA Toy Experiments

This workspace validates the QuantVLA mechanics before starting GR00T or LIBERO.

It is intentionally self-contained:

- `quant.py`: W4A8 fake quantization, SmoothQuant-style linear smoothing, metrics.
- `distributions.py`: standard-normal and VLA-like synthetic weight/activation presets.
- `toy_attention.py`: small attention and transformer block utilities.
- `run_toy_experiments.py`: runs the required Phase 2 experiments and writes reports.
- `extract_weight_stats.py`: optional checkpoint statistics dumper. It loads weights only.

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
