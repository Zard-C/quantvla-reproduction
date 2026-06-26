# CUDA Graph DiT Probe

Boundary: fixed-observation offline replay. This does not run the LIBERO simulator.

## Setup

- Dataset indices: `115, 462, 632, 1063`
- Cases: `40` from `4` observations x `10` repeats
- Denoising steps: `8`
- Capture target: `policy.model.action_head.model.forward`
- Prepare-input pruning: `True`
- Action-head static cache: `True`
- CUDA graph failed: `False`
- CUDA graph error: `None`
- CUDA graph cache size: `3` / `8`

## Latency

| path | count | mean | p50 | p90 | p99 |
|---|---:|---:|---:|---:|---:|
| eager | 40 | 0.124393s | 0.103712s | 0.178494s | 0.423167s |
| cuda graph DiT | 40 | 0.0996902s | 0.101599s | 0.113588s | 0.173546s |
| p50 speedup |  |  | `1.021x` |  |  |

## Drift

| comparison | rel RMSE mean | rel RMSE max | cosine mean | max abs diff |
|---|---:|---:|---:|---:|
| eager A/A, same seed | 0 | 0 | 0.999999999 | 0 |
| eager vs cuda graph, same seed | 0.00663218 | 0.0205126 | 0.999974536 | 0.0328446 |

## Capture Counters

- Capture count: `3`
- Capture seconds: `0.510867`
- Graph calls: `352`
- Eager fallback calls: `0`
- Shape mismatch calls: `2`

## Interpretation

- This probes whether CUDA graph replay can reduce launch/Python overhead while preserving the eager DiT kernels.
- If drift is near the eager A/A row and p50 improves, the next step is an opt-in inference-service switch plus a small matched rollout.
- If graph capture fails or falls back, we should keep torch.compile/block-island as the engineering route and avoid rollout tests for this path.
