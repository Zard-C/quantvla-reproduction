# QuantVLA Phase 2 Rerun Prompt

Rerun Phase 2 before starting GR00T/LIBERO.

Reason: the first Phase 2 run produced very large absolute MSE values under the VLA-like preset. That is not automatically a failure, but it means absolute MSE is dominated by the synthetic tensor scale. The rerun must report normalized metrics and must calibrate the synthetic distribution against real VLA weight statistics when possible.

## Why The First MSE Was Huge

From `toy_quantvla/results/summary.json`:

- VLA-like activation `max_abs` reached about `182.7`.
- VLA-like DiT MLP weight `max_abs` reached about `26.2`.
- DiT attention logits std reached about `173.9`.
- Post-output RMS reached about `45.8`.

With this scale, absolute output MSE can be very large even when relative error or cosine similarity is the more meaningful signal. Keep absolute MSE, but do not use it as the primary decision metric.

## Required Rerun Changes

Pull latest code first:

```bash
git pull
```

The latest toy code adds:

- `output_nmse`
- `final_output_nmse`
- cosine metrics for final output
- relative RMS errors
- relative logits std MAE
- a stricter `phase3_ready` gate

Run:

```bash
python toy_quantvla/run_toy_experiments.py
```

## Real Weight Statistics

Before the rerun, if a GR00T or other VLA checkpoint is locally available, dump lightweight weight statistics only. Do not start the inference server and do not run LIBERO.

Example for a local checkpoint file:

```bash
python toy_quantvla/extract_weight_stats.py /path/to/checkpoint.pt \
  --output toy_quantvla/results/weight_stats.json \
  --max-layers 128
```

If the model is not a single checkpoint file, add a small loader that reads its state dict without inference. Save only JSON statistics:

- shape
- mean/std
- max_abs
- p99 / p99.9
- max_abs / p99.9 tail ratio
- per-output-channel max summaries
- top outlier channels

Do not commit model weights or tensor dumps.

## Metrics To Use For Decisions

Primary metrics:

- `output_nmse`
- `final_output_nmse`
- `final_output_cosine`
- `logits_std_relative_mae`
- `rms_relative_error`
- `post_o_nmse`

Secondary metrics:

- absolute `output_mse`
- absolute `final_output_mse`
- absolute `post_o_mse`

## Acceptance Criteria For Phase 2

Phase 2 supports moving toward Phase 3 only if:

- ATM multiply beats divide on relative logits std error.
- OHB multiply beats divide on relative RMS error.
- DiT attention quantization is worse than DiT MLP-only under normalized metrics, not only absolute MSE.
- Smoothing improves VLA-like normalized linear error more than standard-normal normalized error.
- Calibration with 32 or 128 samples improves relative metrics.
- `weight_stats.json` exists, or the report explicitly says no VLA checkpoint was available and marks the conclusion as synthetic-only.

## Report Back

Paste or commit:

- `toy_quantvla/results/summary.md`
- `toy_quantvla/results/summary.json`
- all CSVs under `toy_quantvla/results/`
- optional `toy_quantvla/results/weight_stats.json`

In the message, include:

- exact commands used
- whether real checkpoint stats were extracted
- whether `phase3_ready` is true or false
- a short note explaining any large absolute MSE using normalized metrics
