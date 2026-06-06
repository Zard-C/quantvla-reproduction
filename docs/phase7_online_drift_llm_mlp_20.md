# Phase 7 Online Drift Smoke

Boundary: this is an online teacher-controlled drift diagnostic. The FP16 teacher steps the LIBERO environment; the FP4-like student is evaluated on the same live observations under matched denoising seeds. This is not a quantized closed-loop success-rate benchmark.

## Run

- Scope: `llm_mlp_only` (LLM MLP Linear modules only)
- Quant format: `fp4_e2m1_blockscaled_fake`
- FP4 block size: `32`
- Cases: `8:7,8:9`
- Episodes: `2`
- Teacher successes: `0/2`

## Episodes

| task | init | success | steps | raw rel RMSE mean | raw cosine mean | LIBERO action RMSE mean | max action diff | exception |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 8 | 7 | False | 20 | 0.0124133 | 0.99991 | 0.00525533 | 0.0238037 |  |
| 8 | 9 | False | 20 | 0.0114017 | 0.999899 | 0.00397119 | 0.0260925 |  |

## Interpretation

- Same-observation drift is measured before the environment step, so the teacher and student see exactly the same simulator state.
- The environment executes only the teacher action. This isolates numerical action drift from closed-loop trajectory divergence.
- A follow-up paired closed-loop run is only meaningful after this same-observation drift is acceptably small.
