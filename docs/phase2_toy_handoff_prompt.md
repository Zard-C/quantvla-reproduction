# QuantVLA Phase 2 Toy Experiment Handoff Prompt

You are working on a 5090 machine. Do not start the GR00T inference server or LIBERO evaluation yet.

Read the local Phase 1 analysis first:

```bash
sed -n '1,260p' docs/phase1_local_analysis.md
```

If this is a rerun after the first toy results, read and follow `docs/phase2_rerun_prompt.md` as the controlling protocol.

Goal: validate the QuantVLA method mechanics with toy experiments before running GR00T/LIBERO.

Important: the toy model must not use only standard-normal random weights. It should include VLA-like weight and activation distributions, because QuantVLA is mainly about scale drift, outlier channels, and heavy-tailed activation behavior.

Create a small toy workspace:

```text
toy_quantvla/
  quant.py
  distributions.py
  toy_attention.py
  extract_weight_stats.py
  run_toy_experiments.py
  README.md
```

Required experiments:

1. W4A8 fake quantization sanity check.
2. Linear smoothing/outlier test under both standard-normal and VLA-like distributions.
3. Toy transformer selective quantization:
   - MLP only
   - attention only
   - both
4. ATM direction test:
   - compute `alpha = std_teacher / std_student`
   - compare `student_logits * alpha` vs `student_logits / alpha`
5. OHB direction test:
   - compute `beta = rms_teacher / rms_student`
   - compare `student_output * beta` vs `student_output / beta`
6. Calibration noise test:
   - sample counts 4, 8, 32, 128
   - log-domain clamp `0.30`
   - neutral band `0.03`
7. Distribution sensitivity test:
   - compare standard-normal toy vs VLA-like toy
   - report whether QuantVLA mechanisms matter more under VLA-like outlier/heavy-tail settings

VLA-like distribution requirements:

- Per-layer weight scale sampled from a log-normal distribution.
- Per-output-channel scale sampled from a wider log-normal distribution.
- Sparse outlier channels/columns with 4x to 12x larger magnitude.
- Activation distribution with Gaussian core plus Laplace or Student-t tail.
- Upstream student drift simulated with channel-wise scale shifts and quantization noise.
- Separate presets for LLM attention, LLM MLP, DiT attention, and DiT MLP.

Optional but preferred if a VLA checkpoint is already available locally:

- Do not run inference server or LIBERO.
- Load weights only and dump small JSON statistics for selected layers:
  - shape, mean, std, max_abs, p99, p99.9
  - per-channel max_abs summary
  - tail ratio `max_abs / p99.9`
  - top outlier channel magnitudes
- Use these statistics to parameterize the toy distributions.
- Do not commit model weights or large tensor dumps.

Expected conclusion to check:

- ATM multiply direction should reduce logits std error and softmax distribution drift.
- OHB multiply direction should reduce post-output RMS error.
- Quantizing DiT attention projections should be more fragile than quantizing only MLP, especially under VLA-like distributions.

Required outputs:

```text
toy_quantvla/results/summary.json
toy_quantvla/results/summary.md
toy_quantvla/results/atm_direction.csv
toy_quantvla/results/ohb_direction.csv
toy_quantvla/results/distribution_sensitivity.csv
toy_quantvla/results/weight_stats_schema.json
```

Report back:

- exact command used
- Python/PyTorch/CUDA versions
- tables from `summary.md`
- whether any expectation failed
