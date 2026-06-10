# Phase 16.5 Step-focused Replay

Boundary: eager-controlled same-observation replay. The simulator stays on the eager FP16 trajectory; focused observations are evaluated repeatedly with eager and compiled paths under matched denoising seeds.

## Run

- Cases: `6:8`
- Focus policy steps: `[248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258]`
- Seed offsets: `[-2, -1, 0, 1, 2]`
- Repeats per seed: `3`
- Warmup repeats per focused observation: `1`
- Compile target: `action_head_model_blocks_8_15_eager`
- Compile mode: `reduce-overhead`

## Focus Steps

| case | policy step | samples | cont max max | cont max p99 | repeat span max | top dim | top seed offset | compiled p50 | eager p50 |
|---|---:|---:|---:|---:|---:|---|---:|---:|---:|
| task6:init8 | 248 | 15 | 0.00457764 | 0.00457764 | 0 | `z` | -2 | 84.9 ms | 122.3 ms |
| task6:init8 | 249 | 15 | 0.00185966 | 0.00185966 | 0 | `z` | 2 | 84.7 ms | 126.2 ms |
| task6:init8 | 250 | 15 | 0.00183105 | 0.00183105 | 0 | `z` | -2 | 90.6 ms | 120.2 ms |
| task6:init8 | 251 | 15 | 0.00137329 | 0.00137329 | 0 | `x` | 2 | 84.7 ms | 121.1 ms |
| task6:init8 | 252 | 15 | 0.00205994 | 0.00205994 | 0 | `z` | 1 | 90.1 ms | 120.7 ms |
| task6:init8 | 253 | 15 | 0.0549316 | 0.0549316 | 0 | `x` | 0 | 78.1 ms | 110.3 ms |
| task6:init8 | 254 | 15 | 0.00146806 | 0.00146806 | 0 | `x` | 0 | 76.1 ms | 101.0 ms |
| task6:init8 | 255 | 15 | 0.000957608 | 0.000957608 | 0 | `x` | -1 | 76.2 ms | 96.5 ms |
| task6:init8 | 256 | 15 | 0.00170094 | 0.00170094 | 0 | `z` | -1 | 84.4 ms | 117.9 ms |
| task6:init8 | 257 | 15 | 0.00183105 | 0.00183105 | 0 | `z` | 2 | 83.6 ms | 125.2 ms |
| task6:init8 | 258 | 15 | 0.0039053 | 0.0039053 | 0 | `x` | 1 | 85.0 ms | 125.9 ms |

## Reading This

- If the same step repeatedly owns the max drift, the spike is observation-local rather than random noise.
- If `same_seed_repeat_span` is near zero, repeated calls are deterministic under fixed seed; seed changes then explain sensitivity.
- This probe is not a closed-loop success benchmark; it is a numerical-behavior diagnostic before rollout expansion.
