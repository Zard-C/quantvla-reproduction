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
| eager vs compiled, same observation and seed | 0.00269397 | 0.00466628 | 0.999995924 | 0.00457764 |

## Latency

| path | count | mean | p50 | p90 | p99 |
|---|---:|---:|---:|---:|---:|
| eager warm | 4 | 0.09512s | 0.0953891s | 0.103794s | 0.104105s |
| compiled warm | 4 | 0.0802958s | 0.0837351s | 0.100759s | 0.102443s |

## Notes

- If eager/eager is near zero, matched request seeds are working.
- Any eager/compiled drift is a compile backend numerical effect, not simulator feedback.
- Rollout success-rate comparisons should be interpreted only after this fixed-observation gate.
