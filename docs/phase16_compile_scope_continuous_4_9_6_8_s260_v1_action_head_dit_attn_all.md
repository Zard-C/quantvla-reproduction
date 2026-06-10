# Phase 13 torch.compile Online Drift

Boundary: teacher-controlled online replay. The eager policy steps LIBERO; the compiled action-head path is evaluated on the same live observations under the same denoising seeds.

## Run

- Cases: `4:9,6:8`
- Episodes: `2`
- Eager successes: `1/2`
- Compile target: `action_head_dit_attn_all`
- Compile mode: `reduce-overhead`

## Episodes

| task | init | eager success | steps | raw rel RMSE mean | raw max diff | LIBERO RMSE mean | compiled/eager time |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | 9 | True | 224 | 0.00168792 | 0.00640869 | 0.000373529 | 1.215 |
| 6 | 8 | False | 260 | 0.00231886 | 0.0230103 | 0.000322999 | 1.201 |

## Interpretation

- Same-observation drift is measured before applying the eager action, so simulator feedback is held on the eager trajectory.
- This answers whether compile drift grows around contact observations even before the compiled policy controls the environment.
