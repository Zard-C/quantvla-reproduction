# Phase 13 torch.compile Online Drift

Boundary: teacher-controlled online replay. The eager policy steps LIBERO; the compiled action-head path is evaluated on the same live observations under the same denoising seeds.

## Run

- Cases: `4:6,6:0`
- Episodes: `2`
- Eager successes: `2/2`
- Compile target: `action_head_model_ff_8_15_eager`
- Compile mode: `reduce-overhead`

## Episodes

| task | init | eager success | steps | raw rel RMSE mean | raw max diff | LIBERO RMSE mean | compiled/eager time |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | 6 | True | 245 | 0.00235262 | 0.129303 | 0.000454196 | 0.8489 |
| 6 | 0 | True | 210 | 0.00248151 | 0.00549316 | 0.000450796 | 1.332 |

## Interpretation

- Same-observation drift is measured before applying the eager action, so simulator feedback is held on the eager trajectory.
- This answers whether compile drift grows around contact observations even before the compiled policy controls the environment.
