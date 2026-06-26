# Phase 13 torch.compile Online Drift

Boundary: teacher-controlled online replay. The eager policy steps LIBERO; the compiled action-head path is evaluated on the same live observations under the same denoising seeds.

## Run

- Cases: `4:6`
- Episodes: `1`
- Eager successes: `0/1`
- Compile target: `action_head_dit_ff_all`
- Compile mode: `reduce-overhead`

## Episodes

| task | init | eager success | steps | raw rel RMSE mean | raw max diff | LIBERO RMSE mean | compiled/eager time |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | 6 | False | 5 | 0.00110667 | 0.00335693 | 0.000115894 | 2.139 |

## Interpretation

- Same-observation drift is measured before applying the eager action, so simulator feedback is held on the eager trajectory.
- This answers whether compile drift grows around contact observations even before the compiled policy controls the environment.
