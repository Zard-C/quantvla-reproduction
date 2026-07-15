# Phase 13 torch.compile Online Drift

Boundary: teacher-controlled online replay. The eager policy steps LIBERO; the compiled action-head path is evaluated on the same live observations under the same denoising seeds.

## Run

- Cases: `4:9,6:8`
- Episodes: `2`
- Eager successes: `2/2`
- Compile target: `action_head_model_blocks_0_3_eager`
- Compile mode: `reduce-overhead`

## Episodes

| task | init | eager success | steps | raw rel RMSE mean | raw max diff | LIBERO RMSE mean | compiled/eager time |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | 9 | True | 224 | 0.00129408 | 0.00732422 | 0.000321635 | 1.256 |
| 6 | 8 | True | 649 | 0.00337935 | 0.283691 | 0.000452108 | 0.8624 |

## Interpretation

- Same-observation drift is measured before applying the eager action, so simulator feedback is held on the eager trajectory.
- This answers whether compile drift grows around contact observations even before the compiled policy controls the environment.
