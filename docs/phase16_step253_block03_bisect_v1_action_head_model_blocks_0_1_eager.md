# Phase 16.5 Step-focused Replay

Boundary: eager-controlled same-observation replay. The simulator stays on the eager FP16 trajectory; focused observations are evaluated repeatedly with eager and compiled paths under matched denoising seeds.

## Run

- Cases: `6:8`
- Focus policy steps: `[253]`
- Seed offsets: `[-2, 0, 2]`
- Repeats per seed: `2`
- Warmup repeats per focused observation: `1`
- Compile target: `action_head_model_blocks_0_1_eager`
- Compile mode: `reduce-overhead`

## Focus Steps

| case | policy step | samples | cont max max | cont max p99 | repeat span max | top dim | top seed offset | compiled p50 | eager p50 |
|---|---:|---:|---:|---:|---:|---|---:|---:|---:|
| task6:init8 | 253 | 6 | 0.00157356 | 0.00157356 | 0 | `x` | -2 | 75.7 ms | 130.4 ms |

## Reading This

- If the same step repeatedly owns the max drift, the spike is observation-local rather than random noise.
- If `same_seed_repeat_span` is near zero, repeated calls are deterministic under fixed seed; seed changes then explain sensitivity.
- This probe is not a closed-loop success benchmark; it is a numerical-behavior diagnostic before rollout expansion.
