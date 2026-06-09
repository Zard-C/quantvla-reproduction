# Phase 13 torch.compile Online Drift

Boundary: teacher-controlled online replay. The eager policy steps LIBERO; the compiled action-head path is evaluated on the same live observations under the same denoising seeds.

## Run

- Cases: `4:6`
- Episodes: `1`
- Eager successes: `1/1`
- Compile target: `action_head_dit_ff_8_15`
- Compile mode: `reduce-overhead`

## Episodes

| task | init | eager success | steps | raw rel RMSE mean | raw max diff | LIBERO RMSE mean | compiled/eager time |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | 6 | True | 245 | 0.0019865 | 0.115875 | 0.000344949 | 1.419 |

## Interpretation

- Same-observation drift is measured before applying the eager action, so simulator feedback is held on the eager trajectory.
- This answers whether compile drift grows around contact observations even before the compiled policy controls the environment.
