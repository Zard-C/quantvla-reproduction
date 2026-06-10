# Phase 16.5 Step-focused Replay

Boundary: eager-controlled same-observation replay. The simulator stays on the eager FP16 trajectory; focused observations are evaluated repeatedly with eager and compiled paths under matched denoising seeds.

## Run

- Cases: `6:8`
- Focus policy steps: `[248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258]`
- Seed offsets: `[-2, -1, 0, 1, 2]`
- Repeats per seed: `3`
- Warmup repeats per focused observation: `1`
- Compile target: `action_head_model`
- Compile mode: `reduce-overhead`

## Focus Steps

| case | policy step | samples | cont max max | cont max p99 | repeat span max | top dim | top seed offset | compiled p50 | eager p50 |
|---|---:|---:|---:|---:|---:|---|---:|---:|---:|
| task6:init8 | 248 | 15 | 0.00274658 | 0.00274658 | 0 | `z` | -2 | 62.4 ms | 142.2 ms |
| task6:init8 | 249 | 15 | 0.00208855 | 0.00208855 | 0 | `z` | 2 | 63.7 ms | 144.9 ms |
| task6:init8 | 250 | 15 | 0.00137329 | 0.00137329 | 0 | `y` | -1 | 62.1 ms | 141.6 ms |
| task6:init8 | 251 | 15 | 0.00137329 | 0.00137329 | 0 | `x` | 2 | 63.1 ms | 142.6 ms |
| task6:init8 | 252 | 15 | 0.00160217 | 0.00160217 | 0 | `z` | 1 | 63.4 ms | 142.7 ms |
| task6:init8 | 253 | 15 | 0.0549316 | 0.0549316 | 0 | `x` | 0 | 61.3 ms | 150.8 ms |
| task6:init8 | 254 | 15 | 0.00137329 | 0.00137329 | 0 | `y` | -2 | 61.5 ms | 141.3 ms |
| task6:init8 | 255 | 15 | 0.000915527 | 0.000915527 | 0 | `x` | -2 | 61.6 ms | 141.0 ms |
| task6:init8 | 256 | 15 | 0.00137329 | 0.00137329 | 0 | `x` | 1 | 61.6 ms | 140.8 ms |
| task6:init8 | 257 | 15 | 0.00274658 | 0.00274658 | 0 | `z` | 2 | 62.0 ms | 140.7 ms |
| task6:init8 | 258 | 15 | 0.00424862 | 0.00424862 | 0 | `x` | 1 | 63.5 ms | 143.9 ms |

## Reading This

- If the same step repeatedly owns the max drift, the spike is observation-local rather than random noise.
- If `same_seed_repeat_span` is near zero, repeated calls are deterministic under fixed seed; seed changes then explain sensitivity.
- This probe is not a closed-loop success benchmark; it is a numerical-behavior diagnostic before rollout expansion.
