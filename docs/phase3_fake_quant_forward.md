# QuantVLA Phase 3 Fake-Quant Forward

Boundary: this is W4A8 fake quantization on synthetic inputs through the official GR00T policy path. It is not packed-kernel inference and not a LIBERO benchmark.

## Run

- Model: `/root/autodl-tmp/models/gr00t-n1.5-libero-long-posttrain`
- Observations: `3`
- Synthetic variants: `zero, midgray, noise`
- Denoising steps: `1`
- Weight bits: `4`
- Activation bits: `8`
- Smoothing alphas: `none, sq_alpha_0.03, sq_alpha_0.05, sq_alpha_0.1, sq_alpha_0.15, sq_alpha_0.3, sq_alpha_0.5`
- Seeds: `260203 + observation_index`

## Output Drift

| config | smoothing | act scale | quantized modules | NMSE mean | NMSE max | rel RMSE mean | cosine mean | max abs diff |
|---|---|---|---:|---:|---:|---:|---:|---:|
| llm_only | none | absmax | 84 | 0.252094 | 0.567512 | 0.454875 | 0.865444 | 0.919922 |
| dit_mlp_only | none | absmax | 32 | 0.0157571 | 0.0261625 | 0.120712 | 0.99262 | 0.167969 |
| llm_dit_mlp | none | absmax | 116 | 0.248289 | 0.494445 | 0.46892 | 0.883665 | 0.873047 |
| llm_only | none | p999 | 84 | 0.538064 | 0.710425 | 0.729038 | 0.767835 | 0.720703 |
| dit_mlp_only | none | p999 | 32 | 0.0283928 | 0.0479901 | 0.159474 | 0.99056 | 0.503906 |
| llm_dit_mlp | none | p999 | 116 | 0.497372 | 0.625962 | 0.700587 | 0.764734 | 0.925781 |
| llm_only | sq_alpha_0.03 | absmax | 84 | 0.324057 | 0.73354 | 0.515694 | 0.821703 | 0.984375 |
| dit_mlp_only | sq_alpha_0.03 | absmax | 32 | 0.0225279 | 0.035192 | 0.14667 | 0.992756 | 0.1875 |
| llm_dit_mlp | sq_alpha_0.03 | absmax | 116 | 0.341948 | 0.688845 | 0.547865 | 0.86418 | 0.925781 |
| llm_only | sq_alpha_0.03 | p999 | 84 | 0.555828 | 0.816647 | 0.734891 | 0.73358 | 0.927734 |
| dit_mlp_only | sq_alpha_0.03 | p999 | 32 | 0.0949315 | 0.164849 | 0.293618 | 0.96031 | 0.484375 |
| llm_dit_mlp | sq_alpha_0.03 | p999 | 116 | 0.64725 | 0.743461 | 0.80143 | 0.71214 | 0.849609 |
| llm_only | sq_alpha_0.05 | absmax | 84 | 0.34813 | 0.741812 | 0.546213 | 0.843025 | 0.925781 |
| dit_mlp_only | sq_alpha_0.05 | absmax | 32 | 0.0180923 | 0.0300292 | 0.130071 | 0.993739 | 0.131836 |
| llm_dit_mlp | sq_alpha_0.05 | absmax | 116 | 0.388673 | 0.662007 | 0.604426 | 0.879331 | 0.841797 |
| llm_only | sq_alpha_0.05 | p999 | 84 | 0.584947 | 0.725395 | 0.762167 | 0.78519 | 0.857422 |
| dit_mlp_only | sq_alpha_0.05 | p999 | 32 | 0.0858359 | 0.14978 | 0.280712 | 0.964046 | 0.457367 |
| llm_dit_mlp | sq_alpha_0.05 | p999 | 116 | 0.559307 | 0.590287 | 0.747295 | 0.802351 | 0.832031 |
| llm_only | sq_alpha_0.1 | absmax | 84 | 0.377869 | 0.743259 | 0.570247 | 0.769956 | 0.992188 |
| dit_mlp_only | sq_alpha_0.1 | absmax | 32 | 0.025578 | 0.0505545 | 0.149475 | 0.987675 | 0.140625 |
| llm_dit_mlp | sq_alpha_0.1 | absmax | 116 | 0.334924 | 0.712087 | 0.529114 | 0.810685 | 0.976562 |
| llm_only | sq_alpha_0.1 | p999 | 84 | 0.626552 | 0.832705 | 0.785667 | 0.704894 | 0.943359 |
| dit_mlp_only | sq_alpha_0.1 | p999 | 32 | 0.0317798 | 0.0470774 | 0.175884 | 0.987468 | 0.34375 |
| llm_dit_mlp | sq_alpha_0.1 | p999 | 116 | 0.62554 | 0.788367 | 0.783839 | 0.70129 | 0.96875 |
| llm_only | sq_alpha_0.15 | absmax | 84 | 0.508003 | 0.579073 | 0.71169 | 0.746221 | 0.949219 |
| dit_mlp_only | sq_alpha_0.15 | absmax | 32 | 0.0173543 | 0.0264677 | 0.129632 | 0.994229 | 0.183594 |
| llm_dit_mlp | sq_alpha_0.15 | absmax | 116 | 0.425923 | 0.536649 | 0.649937 | 0.814033 | 0.925781 |
| llm_only | sq_alpha_0.15 | p999 | 84 | 0.685945 | 0.756975 | 0.827383 | 0.625612 | 0.96875 |
| dit_mlp_only | sq_alpha_0.15 | p999 | 32 | 0.0314614 | 0.0414814 | 0.176259 | 0.987406 | 0.347656 |
| llm_dit_mlp | sq_alpha_0.15 | p999 | 116 | 0.623823 | 0.730891 | 0.788299 | 0.733732 | 0.96875 |
| llm_only | sq_alpha_0.3 | absmax | 84 | 0.303335 | 0.375725 | 0.546998 | 0.865412 | 0.849609 |
| dit_mlp_only | sq_alpha_0.3 | absmax | 32 | 0.0099748 | 0.0169677 | 0.0970072 | 0.997897 | 0.164062 |
| llm_dit_mlp | sq_alpha_0.3 | absmax | 116 | 0.281802 | 0.372525 | 0.527613 | 0.884681 | 0.800781 |
| llm_only | sq_alpha_0.3 | p999 | 84 | 0.554802 | 0.640671 | 0.741197 | 0.772955 | 0.941406 |
| dit_mlp_only | sq_alpha_0.3 | p999 | 32 | 0.0373143 | 0.0616107 | 0.188376 | 0.984549 | 0.319336 |
| llm_dit_mlp | sq_alpha_0.3 | p999 | 116 | 0.588668 | 0.711305 | 0.76324 | 0.754392 | 0.972656 |
| llm_only | sq_alpha_0.5 | absmax | 84 | 0.367736 | 0.60032 | 0.568173 | 0.806815 | 0.9375 |
| dit_mlp_only | sq_alpha_0.5 | absmax | 32 | 0.0168351 | 0.0250417 | 0.116709 | 0.992165 | 0.164062 |
| llm_dit_mlp | sq_alpha_0.5 | absmax | 116 | 0.299758 | 0.518403 | 0.519465 | 0.882257 | 0.880859 |
| llm_only | sq_alpha_0.5 | p999 | 84 | 0.613692 | 0.820643 | 0.775804 | 0.651997 | 0.984375 |
| dit_mlp_only | sq_alpha_0.5 | p999 | 32 | 0.00389831 | 0.00550521 | 0.06079 | 0.998132 | 0.0761719 |
| llm_dit_mlp | sq_alpha_0.5 | p999 | 116 | 0.625889 | 0.804132 | 0.785072 | 0.657227 | 0.958496 |

## Interpretation

- The comparison uses matched RNG seeds because GR00T action denoising starts from random Gaussian actions.
- `llm_only` quantizes all selected LLM attention and MLP linears; `dit_mlp_only` quantizes only DiT feed-forward linears; `llm_dit_mlp` is the intended QuantVLA selected set.
- `absmax` activation scale is conservative dynamic A8. `p999` clips roughly 0.1% of activation values per module call and is included to expose outlier sensitivity.
- `sq_alpha_*` applies calibration-based SmoothQuant-style scale migration using input-channel activation maxima from the same synthetic calibration set.
- These are fake-quant output-drift probes. They do not yet include ATM or OHB.
