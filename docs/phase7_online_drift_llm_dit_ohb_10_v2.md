# Phase 7 Online Drift Smoke

Boundary: this is an online teacher-controlled drift diagnostic. The FP16 teacher steps the LIBERO environment; the FP4-like student is evaluated on the same live observations under matched denoising seeds. This is not a quantized closed-loop success-rate benchmark.

## Run

- Scope: `llm_dit_mlp` (Phase 5-compatible full selected scope: LLM attention, LLM MLP, and DiT MLP)
- Mode: `ohb`
- Quant format: `fp4_e2m1_blockscaled_fake`
- FP4 block size: `32`
- Cases: `8:7,8:9`
- Episodes: `2`
- Teacher successes: `0/2`

## Episodes

| task | init | success | steps | raw rel RMSE mean | raw cosine mean | LIBERO action RMSE mean | max action diff | student/teacher time | exception |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 8 | 7 | False | 10 | 0.0254397 | 0.999665 | 0.00665288 | 0.0311279 | 5.565 |  |
| 8 | 9 | False | 10 | 0.0292438 | 0.999507 | 0.00837833 | 0.0265504 | 4.992 |  |

## ATM/OHB Scale Summary

- Attention processors: `16`
- Calibration observations: `3`
- Alpha mean/min/max: `1.02752` / `0.992142` / `1.06015`
- Beta mean/min/max: `1.00639` / `0.968971` / `1.03562`

## Interpretation

- Same-observation drift is measured before the environment step, so the teacher and student see exactly the same simulator state.
- The environment executes only the teacher action. This isolates numerical action drift from closed-loop trajectory divergence.
- A follow-up paired closed-loop run is only meaningful after this same-observation drift is acceptably small.
