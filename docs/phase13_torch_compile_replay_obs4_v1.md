# Phase 13 torch.compile Fixed-Observation Replay

Boundary: offline replay on fixed real observations. This isolates RNG and numerical drift from closed-loop simulator trajectory changes.

## Run

- Dataset: `/root/autodl-tmp/datasets/libero_10_subset`
- Dataset indices: `115, 462, 632, 1063`
- Base seed: `20260613`
- Denoising steps: `8`
- Compile target: `action_head_model`
- Compile backend: `inductor`
- Compile mode: `reduce-overhead`

## Determinism Check

| comparison | rel RMSE mean | rel RMSE max | cosine mean | max abs diff |
|---|---:|---:|---:|---:|
| eager vs eager, same seed | 0 | 0 | 0.999999999 | 0 |

## Compile Drift

| comparison | rel RMSE mean | rel RMSE max | cosine mean | max abs diff |
|---|---:|---:|---:|---:|
| eager vs compiled, same observation and seed | 0.0026489 | 0.00396635 | 0.999996402 | 0.00549316 |

## Latency

| path | count | mean | p50 | p90 | p99 |
|---|---:|---:|---:|---:|---:|
| eager warm | 4 | 0.0890004s | 0.0877083s | 0.0935466s | 0.0957601s |
| compiled warm | 4 | 0.176802s | 0.146919s | 0.308016s | 0.350041s |

## Notes

- If eager/eager is near zero, matched request seeds are working.
- Any eager/compiled drift is a compile backend numerical effect, not simulator feedback.
- Rollout success-rate comparisons should be interpreted only after this fixed-observation gate.
