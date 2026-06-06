# Phase 7 Online Drift Smoke

Boundary: this is an online teacher-controlled drift diagnostic. The FP16 teacher steps the LIBERO environment; the FP4-like student is evaluated on the same live observations under matched denoising seeds. This is not a quantized closed-loop success-rate benchmark.

## Run

- Scope: `llm_dit_mlp` (Phase 5-compatible full selected scope: LLM attention, LLM MLP, and DiT MLP)
- Quant format: `fp4_e2m1_blockscaled_fake`
- FP4 block size: `32`
- Cases: `8:7,8:9`
- Episodes: `2`
- Teacher successes: `0/2`

## Episodes

| task | init | success | steps | raw rel RMSE mean | raw cosine mean | LIBERO action RMSE mean | max action diff | exception |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 8 | 7 | False | 10 | 0.0245541 | 0.999688 | 0.00586444 | 0.0283813 |  |
| 8 | 9 | False | 10 | 0.0274691 | 0.999562 | 0.00855036 | 0.0297547 |  |

## Interpretation

- Same-observation drift is measured before the environment step, so the teacher and student see exactly the same simulator state.
- The environment executes only the teacher action. This isolates numerical action drift from closed-loop trajectory divergence.
- A follow-up paired closed-loop run is only meaningful after this same-observation drift is acceptably small.
