# Phase 16.5 Step-focused Replay

Boundary: eager-controlled same-observation replay. The simulator stays on the eager FP16 trajectory; focused observations are evaluated repeatedly with eager and compiled paths under matched denoising seeds.

## Run

- Cases: `6:8`
- Focus policy steps: `[248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258]`
- Seed offsets: `[-2, -1, 0, 1, 2]`
- Repeats per seed: `3`
- Warmup repeats per focused observation: `1`
- Compile target: `action_head_dit_attn_all`
- Compile mode: `reduce-overhead`

## Focus Steps

| case | policy step | samples | cont max max | cont max p99 | repeat span max | top dim | top seed offset | compiled p50 | eager p50 |
|---|---:|---:|---:|---:|---:|---|---:|---:|---:|
| task6:init8 | 248 | 15 | 0.000915527 | 0.000915527 | 0 | `z` | -2 | 101.8 ms | 111.8 ms |
| task6:init8 | 249 | 15 | 0.00100136 | 0.00100136 | 0 | `z` | 2 | 102.9 ms | 106.3 ms |
| task6:init8 | 250 | 15 | 0.000915527 | 0.000915527 | 0 | `y` | 2 | 97.1 ms | 107.8 ms |
| task6:init8 | 251 | 15 | 0.000915527 | 0.000915527 | 0 | `z` | -2 | 104.9 ms | 103.8 ms |
| task6:init8 | 252 | 15 | 0.000915527 | 0.000915527 | 0 | `y` | -2 | 106.5 ms | 107.4 ms |
| task6:init8 | 253 | 15 | 0.00183105 | 0.00183105 | 0 | `x` | 0 | 104.7 ms | 107.8 ms |
| task6:init8 | 254 | 15 | 0.00137329 | 0.00137329 | 0 | `y` | -2 | 103.8 ms | 106.4 ms |
| task6:init8 | 255 | 15 | 0.00125885 | 0.00125885 | 0 | `z` | -1 | 103.8 ms | 106.5 ms |
| task6:init8 | 256 | 15 | 0.000915527 | 0.000915527 | 0 | `y` | 1 | 99.5 ms | 111.6 ms |
| task6:init8 | 257 | 15 | 0.000877321 | 0.000877321 | 0 | `pitch` | 2 | 103.7 ms | 106.7 ms |
| task6:init8 | 258 | 15 | 0.0019598 | 0.0019598 | 0 | `x` | 1 | 104.0 ms | 106.8 ms |

## Reading This

- If the same step repeatedly owns the max drift, the spike is observation-local rather than random noise.
- If `same_seed_repeat_span` is near zero, repeated calls are deterministic under fixed seed; seed changes then explain sensitivity.
- This probe is not a closed-loop success benchmark; it is a numerical-behavior diagnostic before rollout expansion.
