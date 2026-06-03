# QuantVLA Phase 2 Toy Results

Date: 2026-06-04

Command:

```bash
python toy_quantvla/run_toy_experiments.py
```

Environment:

| python | torch | cuda | cuda_available | device |
|---|---|---|---|---|
| 3.12.3 | 2.8.0+cu128 | 12.8 | True | NVIDIA GeForce RTX 5090 |

Required outputs were regenerated under `toy_quantvla/results/`:

- `summary.json`
- `summary.md`
- `atm_direction.csv`
- `ohb_direction.csv`
- `distribution_sensitivity.csv`
- `weight_stats_schema.json`

## Conclusions

| check | result |
|---|---|
| ATM multiply direction wins | True |
| OHB multiply direction wins | True |
| VLA-like attention quantization is more fragile than MLP-only quantization | True |
| Linear smoothing gain is larger under VLA-like distributions | True |
| Ready to plan Phase 3 | True |

No expected Phase 2 check failed.

## Key Evidence

ATM uses `alpha = std_teacher / std_student`. Under the VLA-like distribution, multiplying student logits by `alpha` reduced logits standard-deviation MAE from `14.7218` to `2.81685`; dividing worsened it to `30.7068`.

| distribution | direction | alpha_mean | logits_std_mae | attention_js | entropy_abs_error |
|---|---|---:|---:|---:|---:|
| standard_normal | none | 1 | 0.00419641 | 0.000337371 | 0.00269341 |
| standard_normal | multiply | 1 | 0.00419641 | 0.000337371 | 0.00269341 |
| standard_normal | divide | 1 | 0.00419641 | 0.000337371 | 0.00269341 |
| vla_like | none | 0.746869 | 14.7218 | 0.176171 | 0.0554185 |
| vla_like | multiply | 0.746869 | 2.81685 | 0.171566 | 0.00357416 |
| vla_like | divide | 0.746869 | 30.7068 | 0.180301 | 0.0923407 |

OHB uses `beta = rms_teacher / rms_student`. Under the VLA-like distribution, multiplying post-output tensors by `beta` reduced RMS absolute error from `2.42263` to `0`; dividing worsened it to `4.7059`.

| distribution | direction | beta | teacher_rms | student_rms | rms_abs_error |
|---|---|---:|---:|---:|---:|
| standard_normal | none | 1 | 0.311809 | 0.309279 | 0.00253016 |
| standard_normal | multiply | 1 | 0.311809 | 0.309279 | 0.00253016 |
| standard_normal | divide | 1 | 0.311809 | 0.309279 | 0.00253016 |
| vla_like | none | 1.06104 | 42.1143 | 39.6916 | 2.42263 |
| vla_like | multiply | 1.06104 | 42.1143 | 42.1143 | 0 |
| vla_like | divide | 1.06104 | 42.1143 | 37.4084 | 4.7059 |

Selective quantization behaved differently under standard-normal and VLA-like distributions. In the VLA-like setting, attention-only quantization produced about `2.466x` the final-output MSE of MLP-only quantization.

| distribution | variant | final_output_mse | logits_std_abs_error | attention_js | post_o_rms_abs_error |
|---|---|---:|---:|---:|---:|
| standard_normal | mlp_only | 0.014132 | 0 | 0 | 0 |
| standard_normal | attention_only | 0.00711276 | 0.0142579 | 0.00310641 | 0.00516179 |
| standard_normal | attention_and_mlp | 0.0211799 | 0.0142579 | 0.00310641 | 0.00516179 |
| vla_like | mlp_only | 43900.2 | 0 | 0 | 0 |
| vla_like | attention_only | 108263 | 117.548 | 0.384149 | 20.0766 |
| vla_like | attention_and_mlp | 111861 | 117.548 | 0.384149 | 20.0766 |

Linear smoothing mattered much more in the VLA-like case:

| distribution | naive_output_mse | smoothed_output_mse | improvement |
|---|---:|---:|---:|
| standard_normal | 0.0135451 | 0.0132462 | 0.000298981 |
| vla_like_dit_mlp | 33.8456 | 26.1289 | 7.7168 |

## Interpretation

Phase 2 supports the implementation choices from Phase 1:

- Apply ATM by multiplying quantized/student attention logits, or equivalently multiplying `Q`, by `alpha`.
- Apply OHB by multiplying the student post-output projection by `beta`.
- Keep DiT attention projections floating point in the main QuantVLA layout.
- Quantize LLM linears and DiT MLP linears first.
- Use VLA-like heavy-tailed and outlier-channel tests when debugging, because standard-normal tests hide the failure mode.

