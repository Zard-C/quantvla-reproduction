# QuantVLA Phase 3 Fake-Quant Forward

Boundary: this is W4A8 fake quantization on synthetic inputs through the official GR00T policy path. It is not packed-kernel inference and not a LIBERO benchmark.

## Run

- Model: `/root/autodl-tmp/models/gr00t-n1.5-libero-long-posttrain`
- Observations: `3`
- Synthetic variants: `zero, midgray, noise`
- Denoising steps: `1`
- Weight bits: `4`
- Activation bits: `8`
- Seeds: `260203 + observation_index`

## Output Drift

| config | act scale | quantized modules | NMSE mean | NMSE max | rel RMSE mean | cosine mean | max abs diff |
|---|---|---:|---:|---:|---:|---:|---:|
| llm_only | absmax | 84 | 0.252094 | 0.567512 | 0.454875 | 0.865444 | 0.919922 |
| dit_mlp_only | absmax | 32 | 0.0157571 | 0.0261625 | 0.120712 | 0.99262 | 0.167969 |
| llm_dit_mlp | absmax | 116 | 0.248289 | 0.494445 | 0.46892 | 0.883665 | 0.873047 |
| llm_only | p999 | 84 | 0.538064 | 0.710425 | 0.729038 | 0.767835 | 0.720703 |
| dit_mlp_only | p999 | 32 | 0.0283928 | 0.0479901 | 0.159474 | 0.99056 | 0.503906 |
| llm_dit_mlp | p999 | 116 | 0.497372 | 0.625962 | 0.700587 | 0.764734 | 0.925781 |

## Interpretation

- The comparison uses matched RNG seeds because GR00T action denoising starts from random Gaussian actions.
- `llm_only` quantizes all selected LLM attention and MLP linears; `dit_mlp_only` quantizes only DiT feed-forward linears; `llm_dit_mlp` is the intended QuantVLA selected set.
- `absmax` activation scale is conservative dynamic A8. `p999` clips roughly 0.1% of activation values per module call and is included to expose outlier sensitivity.
- These are fake-quant output-drift probes. They do not yet include SmoothQuant-style scale migration, ATM, or OHB.
