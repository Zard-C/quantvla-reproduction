# QuantVLA Phase 3 Summary

## Status

Phase 3 is complete as an offline GR00T method probe. We can load the downloaded GR00T N1.5 LIBERO post-train checkpoint, run the official `Gr00tPolicy.get_action` path on RTX 5090, capture real activations, patch selected linears with W4A8 fake quantization, and compare teacher/student action outputs under matched RNG seeds.

This phase is not a LIBERO benchmark and does not claim task success rate.

## Environment

- GR00T source: `/root/autodl-tmp/Isaac-GR00T-n1.5`
- GR00T commit: `4af2b62` from `n1.5-release`
- Checkpoint: `/root/autodl-tmp/models/gr00t-n1.5-libero-long-posttrain`
- Python venv: `/root/autodl-tmp/envs/gr00t-py312-cu128`
- Torch: `2.8.0+cu128`
- CUDA reported by torch: `12.8`
- GPU capability: `[12, 0]`
- flash-attn: `2.8.3`

Python 3.12 required a temporary fail-fast `pytorch3d.transforms` stub because `pipablepytorch3d==0.7.6` does not install on Python 3.12. This is acceptable for the current LIBERO synthetic smoke tests because checkpoint metadata has `rotation_type: null`; a full dataset/eval environment should use Python 3.10 with real `pytorch3d`.

## Artifacts

- `docs/phase3_offline_weight_quant.md`
- `docs/phase3_gr00t_smoke.md`
- `docs/phase3_activation_capture.md`
- `docs/phase3_fake_quant_forward.md`
- `toy_quantvla/phase3_weight_quant_analysis.py`
- `toy_quantvla/phase3_gr00t_smoke.py`
- `toy_quantvla/phase3_activation_capture.py`
- `toy_quantvla/phase3_fake_quant_forward.py`

## Checkpoint And Selector

The Phase 1 selector matches the downloaded checkpoint naming:

| group | modules |
|---|---:|
| `llm_selected` | 84 |
| `dit_mlp_selected` | 32 |
| `dit_attention_excluded` | 64 |

Static W4 weight-only fake quantization shows low weight NMSE overall:

| group | params | W4 NMSE mean | W4 NMSE max | cosine mean |
|---|---:|---:|---:|---:|
| `llm_selected` | 603,979,776 | 0.02987 | 0.04428 | 0.98552 |
| `dit_mlp_selected` | 301,989,888 | 0.03727 | 0.18975 | 0.98229 |
| `dit_attention_excluded` | 163,577,856 | 0.02353 | 0.02763 | 0.98849 |

This supports moving beyond static weights: the weight tensors are not enough to decide forward behavior.

## Loading And Forward Smoke

The local LIBERO post-train checkpoint loads through the official GR00T policy path:

| metric | value |
|---|---:|
| load time | 9.67 s |
| action horizon | 16 |
| action dim | 32 |
| CUDA allocated after load | 5214.94 MiB |
| synthetic forward time, denoising steps 1 | 0.43 s |

The synthetic forward produces all seven expected action components: `x`, `y`, `z`, `roll`, `pitch`, `yaw`, and `gripper`, each with shape `[16, 1]`.

## Activation Capture

Using three synthetic variants (`zero`, `midgray`, `noise`) and `denoising_steps=1`, all target modules were exercised:

| group | called | input RMS mean | input absmax mean | input p99.9 mean | input tail mean | input W8 NMSE absmax | input W8 NMSE p99.9 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `llm_selected` | 84 | 0.53285 | 274.23 | 3.993 | 320.96 | 0.02885 | 0.05241 |
| `dit_mlp_selected` | 32 | 0.64895 | 18.32 | 7.860 | 2.670 | 0.00338 | 0.03834 |
| `dit_attention_excluded` | 64 | 0.78078 | 35.47 | 5.275 | 4.097 | 0.00810 | 0.04441 |

The largest activation tail is concentrated in LLM MLP down-projection inputs. The most extreme observed case is:

- `backbone.eagle_model.language_model.model.layers.2.mlp.down_proj`
- input tail ratio around `2.64e4`
- p99.9-scale W8 input NMSE around `1.0`

This explains why percentile clipping is risky for the LLM path: a tiny number of values can carry a very large fraction of activation energy.

## Naive W4A8 Forward Drift

The fake-quant wrapper uses W4 per-output-channel weight fake quantization and dynamic A8 activation fake quantization. Matched seeds are required because GR00T action denoising starts from random Gaussian actions.

Naive W4A8 results:

| config | act scale | modules | action NMSE mean | relative RMSE mean | cosine mean |
|---|---|---:|---:|---:|---:|
| `llm_only` | absmax | 84 | 0.25209 | 0.45488 | 0.86544 |
| `dit_mlp_only` | absmax | 32 | 0.01576 | 0.12071 | 0.99262 |
| `llm_dit_mlp` | absmax | 116 | 0.24829 | 0.46892 | 0.88366 |
| `llm_only` | p99.9 | 84 | 0.53806 | 0.72904 | 0.76783 |
| `dit_mlp_only` | p99.9 | 32 | 0.02839 | 0.15947 | 0.99056 |
| `llm_dit_mlp` | p99.9 | 116 | 0.49737 | 0.70059 | 0.76473 |

Naive dynamic absmax A8 is much safer than p99.9 clipping for the LLM path. DiT MLP-only quantization is comparatively stable.

## Smoothing Sweep

We added calibration-based SmoothQuant-style migration:

```text
scale = act_max^alpha / weight_max^(1 - alpha)
x_smooth = x / scale
W_smooth = W * scale
```

The calibration set is the same three synthetic variants. This is a controlled method probe, not a benchmark.

Best observed results by group:

| config | act scale | best smoothing | action NMSE mean | relative RMSE mean | cosine mean |
|---|---|---|---:|---:|---:|
| `llm_only` | absmax | none | 0.25209 | 0.45488 | 0.86544 |
| `llm_only` | p99.9 | none | 0.53806 | 0.72904 | 0.76783 |
| `dit_mlp_only` | absmax | `alpha=0.3` | 0.00997 | 0.09701 | 0.99790 |
| `dit_mlp_only` | p99.9 | `alpha=0.5` | 0.00390 | 0.06079 | 0.99813 |
| `llm_dit_mlp` | absmax | none | 0.24829 | 0.46892 | 0.88366 |
| `llm_dit_mlp` | p99.9 | none | 0.49737 | 0.70059 | 0.76473 |

Interpretation:

- SmoothQuant-style migration helps DiT MLP substantially.
- The same smoothing does not fix LLM drift in this synthetic setup; it often worsens it.
- For the combined selected set, LLM drift dominates, so the best tested combined setting is still naive dynamic absmax A8.
- This does not disprove QuantVLA. It tells us that the real bottleneck in this GR00T synthetic probe is the LLM activation outlier path, and that a faithful reproduction needs stronger LLM calibration, real calibration observations, or the paper's exact scale policies before benchmark evaluation.

## Phase 3 Conclusion

Phase 3 supports several method-level claims:

1. The selective layout is implementable on GR00T N1.5 LIBERO and matches checkpoint module names.
2. DiT attention Q/K/V/O modules are easy to identify and remain excluded in the main QuantVLA layout.
3. Real forward activations, not just weights, are the main risk.
4. DiT MLP W4A8 is relatively stable and benefits from smoothing.
5. LLM W4A8 is the dominant synthetic-output drift source; naive W4A8 is not sufficient.

Phase 3 does not establish benchmark-level performance. The next stage should move from synthetic probes to real calibration/evaluation observations, and then test ATM/OHB or equivalent output-balancing mechanisms on real student/teacher rollouts.
