# Phase 16.5 Step-focused Replay Summary

Boundary: eager-controlled same-observation replay around the `task6:init8` action spike region.

## Target Ranking

| target | top case | top step | top dim | cont max | p99 at step | repeat span max | compiled p50 mean | eager p50 mean |
|---|---|---:|---|---:|---:|---:|---:|---:|
| action_head_model | `task6:init8` | 253 | `x` | 0.0549316 | 0.0549316 | 0 | 54.2 ms | 130.4 ms |
| action_head_model_blocks_4_7_eager | `task6:init8` | 253 | `x` | 0.0549316 | 0.0549316 | 0 | 73.3 ms | 123.1 ms |
| action_head_model_blocks_8_15_eager | `task6:init8` | 253 | `x` | 0.0549316 | 0.0549316 | 0 | 82.0 ms | 114.8 ms |
| action_head_model_attn_all_eager | `task6:init8` | 253 | `x` | 0.0544739 | 0.0544739 | 0 | 96.1 ms | 112.4 ms |
| action_head_model_ff_all_eager | `task6:init8` | 253 | `x` | 0.0544739 | 0.0544739 | 0 | 97.9 ms | 122.3 ms |
| action_head_model_blocks_0_3_eager | `task6:init8` | 253 | `x` | 0.00183105 | 0.00183105 | 0 | 73.9 ms | 130.4 ms |
| action_head_dit_attn_all | `task6:init8` | 253 | `x` | 0.00183105 | 0.00183105 | 0 | 102.7 ms | 103.4 ms |
| action_head_model_blocks_0_7_eager | `task6:init8` | 253 | `z` | 0.0013876 | 0.0013876 | 0 | 85.2 ms | 117.6 ms |
| action_head_model_blocks_0_15_eager | `task6:init8` | 253 | `z` | 0.000679493 | 0.000679493 | 0 | 101.1 ms | 108.2 ms |

## Per-Step Rows

| target | case | step | samples | cont max | cont p99 | L2 p99 | repeat span max | top dim | seed offset |
|---|---|---:|---:|---:|---:|---:|---:|---|---:|
| action_head_dit_attn_all | `task6:init8` | 253 | 6 | 0.00183105 | 0.00183105 | 0.00254211 | 0 | `x` | 0 |
| action_head_model | `task6:init8` | 253 | 6 | 0.0549316 | 0.0549316 | 0.0773858 | 0 | `x` | 0 |
| action_head_model_attn_all_eager | `task6:init8` | 253 | 6 | 0.0544739 | 0.0544739 | 0.0767405 | 0 | `x` | 0 |
| action_head_model_blocks_0_15_eager | `task6:init8` | 253 | 6 | 0.000679493 | 0.000679493 | 0.000706742 | 0 | `z` | 2 |
| action_head_model_blocks_0_3_eager | `task6:init8` | 253 | 6 | 0.00183105 | 0.00183105 | 0.00262615 | 0 | `x` | 0 |
| action_head_model_blocks_0_7_eager | `task6:init8` | 253 | 6 | 0.0013876 | 0.0013876 | 0.00182605 | 0 | `z` | -2 |
| action_head_model_blocks_4_7_eager | `task6:init8` | 253 | 6 | 0.0549316 | 0.0549316 | 0.0769532 | 0 | `x` | 0 |
| action_head_model_blocks_8_15_eager | `task6:init8` | 253 | 6 | 0.0549316 | 0.0549316 | 0.0757393 | 0 | `x` | 0 |
| action_head_model_ff_all_eager | `task6:init8` | 253 | 6 | 0.0544739 | 0.0544739 | 0.0754138 | 0 | `x` | 0 |

## Reading This

- `cont max` is the largest absolute drift among x/y/z/roll/pitch/yaw for that focused observation.
- `repeat span max` checks fixed-seed determinism across repeated calls; values near zero mean the spike is repeatable once seed and observation are fixed.
- A spike that survives across big-graph targets but disappears in small scopes points to graph-level numerical path rather than gripper thresholding.
