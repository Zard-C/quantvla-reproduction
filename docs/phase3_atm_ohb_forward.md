# QuantVLA Phase 3 ATM/OHB Forward

Boundary: synthetic-input W4A8 fake quantization through the official GR00T policy path. ATM/OHB scales are calibrated from teacher/student attention statistics on the same synthetic calibration set.

## Run

- Model: `/root/autodl-tmp/models/gr00t-n1.5-libero-long-posttrain`
- Configs: `llm_only, dit_mlp_only, llm_dit_mlp`
- Synthetic variants: `zero, midgray, noise`
- Denoising steps: `1`
- Activation scale: `absmax`
- Smoothing alpha: `none`
- ATM/OHB log clamp: `0.3`

## Output Drift

| config | mode | modules | NMSE mean | NMSE max | rel RMSE mean | cosine mean | max abs diff |
|---|---|---:|---:|---:|---:|---:|---:|
| llm_only | none | 84 | 0.252094 | 0.567512 | 0.454875 | 0.865444 | 0.919922 |
| llm_only | atm | 84 | 0.289215 | 0.650607 | 0.482944 | 0.839354 | 0.945312 |
| llm_only | ohb | 84 | 0.247045 | 0.544986 | 0.448418 | 0.870767 | 0.923828 |
| llm_only | atm_ohb | 84 | 0.286515 | 0.656873 | 0.4722 | 0.83638 | 0.957031 |
| dit_mlp_only | none | 32 | 0.0157571 | 0.0261625 | 0.120712 | 0.99262 | 0.167969 |
| dit_mlp_only | atm | 32 | 0.0168741 | 0.0304722 | 0.122648 | 0.991998 | 0.179688 |
| dit_mlp_only | ohb | 32 | 0.0143528 | 0.0270767 | 0.112458 | 0.993165 | 0.179688 |
| dit_mlp_only | atm_ohb | 32 | 0.0140135 | 0.0243814 | 0.111504 | 0.993239 | 0.167969 |
| llm_dit_mlp | none | 116 | 0.248289 | 0.494445 | 0.46892 | 0.883665 | 0.873047 |
| llm_dit_mlp | atm | 116 | 0.272288 | 0.567637 | 0.486454 | 0.86415 | 0.898438 |
| llm_dit_mlp | ohb | 116 | 0.248055 | 0.473841 | 0.470595 | 0.882903 | 0.873047 |
| llm_dit_mlp | atm_ohb | 116 | 0.260355 | 0.573079 | 0.4665 | 0.864419 | 0.917969 |

## Scale Summary

| config | alpha mean | alpha min | alpha max | beta mean | beta min | beta max |
|---|---:|---:|---:|---:|---:|---:|
| llm_only | 1.1456 | 0.960239 | 1.26207 | 0.954158 | 0.806373 | 1.07383 |
| dit_mlp_only | 1.00129 | 0.973926 | 1.01873 | 1.01009 | 0.986875 | 1.03862 |
| llm_dit_mlp | 1.15007 | 0.97379 | 1.28196 | 0.963786 | 0.807695 | 1.10156 |

## Interpretation

- ATM is applied by multiplying the attention query, which scales attention logits by `alpha = std_teacher / std_student`.
- OHB is applied by multiplying the DiT attention output before residual addition, with `beta = rms_teacher / rms_student`.
- Scales are log-clamped by default to avoid a calibration artifact dominating the synthetic probe.
- This is still not a LIBERO benchmark; it tests whether ATM/OHB reduce teacher/student action drift under controlled synthetic inputs.
