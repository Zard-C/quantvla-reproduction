# Phase 7 Online Drift Smoke

Boundary: this is an online teacher-controlled drift diagnostic. The FP16 teacher steps the LIBERO environment; the FP4-like student is evaluated on the same live observations under matched denoising seeds. This is not a quantized closed-loop success-rate benchmark.

## Run

- Scope: `dit_mlp_only` (DiT feed-forward Linear modules only)
- Quant format: `fp4_e2m1_blockscaled_fake`
- FP4 block size: `32`
- Cases: `8:7,8:9,4:10,0:3`
- Episodes: `4`
- Teacher successes: `0/4`

## Episodes

| task | init | success | steps | raw rel RMSE mean | raw cosine mean | LIBERO action RMSE mean | max action diff | exception |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 0 | 3 | False | 40 | 0.0076393 | 0.999973 | 0.00223345 | 0.0128174 |  |
| 4 | 10 | False | 40 | 0.00748314 | 0.999973 | 0.00243436 | 0.0128174 |  |
| 8 | 7 | False | 40 | 0.00621284 | 0.999981 | 0.00190505 | 0.00846863 |  |
| 8 | 9 | False | 40 | 0.00553952 | 0.999985 | 0.00181031 | 0.00915527 |  |

## Interpretation

- Same-observation drift is measured before the environment step, so the teacher and student see exactly the same simulator state.
- The environment executes only the teacher action. This isolates numerical action drift from closed-loop trajectory divergence.
- A follow-up paired closed-loop run is only meaningful after this same-observation drift is acceptably small.
