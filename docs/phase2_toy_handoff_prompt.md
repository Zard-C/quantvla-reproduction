# QuantVLA Phase 2 Toy Experiment Handoff Prompt

You are working on a 5090 machine. Do not start GR00T or LIBERO yet.

Read the local Phase 1 analysis first:

```bash
sed -n '1,260p' docs/phase1_local_analysis.md
```

Goal: validate the QuantVLA method mechanics with self-contained toy experiments before touching the official GR00T repo.

Create a small toy workspace:

```text
toy_quantvla/
  quant.py
  toy_attention.py
  run_toy_experiments.py
  README.md
```

Required experiments:

1. W4A8 fake quantization sanity check.
2. Linear smoothing/outlier test.
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

Expected conclusion to check:

- ATM multiply direction should reduce logits std error and softmax distribution drift.
- OHB multiply direction should reduce post-output RMS error.
- Quantizing DiT attention projections should be more fragile than quantizing only MLP in the toy setting.

Required outputs:

```text
toy_quantvla/results/summary.json
toy_quantvla/results/summary.md
toy_quantvla/results/atm_direction.csv
toy_quantvla/results/ohb_direction.csv
```

Report back:

- exact command used
- Python/PyTorch/CUDA versions
- tables from `summary.md`
- whether any expectation failed
