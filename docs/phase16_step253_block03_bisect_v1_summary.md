# Phase 16.5 Step-focused Replay Summary

Boundary: eager-controlled same-observation replay around the `task6:init8` action spike region.

## Target Ranking

| target | top case | top step | top dim | cont max | p99 at step | repeat span max | compiled p50 mean | eager p50 mean |
|---|---|---:|---|---:|---:|---:|---:|---:|
| action_head_model_blocks_3_3_eager | `task6:init8` | 253 | `x` | 0.0553894 | 0.0553894 | 0 | 64.7 ms | 137.8 ms |
| action_head_model | `task6:init8` | 253 | `x` | 0.0549316 | 0.0549316 | 0 | 62.9 ms | 139.5 ms |
| action_head_model_blocks_2_3_eager | `task6:init8` | 253 | `x` | 0.0549316 | 0.0549316 | 0 | 74.0 ms | 136.4 ms |
| action_head_model_blocks_2_2_eager | `task6:init8` | 253 | `x` | 0.0549316 | 0.0549316 | 0 | 66.9 ms | 131.9 ms |
| action_head_model_blocks_1_1_eager | `task6:init8` | 253 | `x` | 0.0544739 | 0.0544739 | 0 | 67.8 ms | 128.7 ms |
| action_head_model_blocks_0_3_eager | `task6:init8` | 253 | `x` | 0.00183105 | 0.00183105 | 0 | 80.3 ms | 121.5 ms |
| action_head_model_blocks_0_0_eager | `task6:init8` | 253 | `z` | 0.00183105 | 0.00183105 | 0 | 64.8 ms | 137.0 ms |
| action_head_model_blocks_0_1_eager | `task6:init8` | 253 | `x` | 0.00157356 | 0.00157356 | 0 | 75.7 ms | 130.4 ms |

## Per-Step Rows

| target | case | step | samples | cont max | cont p99 | L2 p99 | repeat span max | top dim | seed offset |
|---|---|---:|---:|---:|---:|---:|---:|---|---:|
| action_head_model | `task6:init8` | 253 | 6 | 0.0549316 | 0.0549316 | 0.0773858 | 0 | `x` | 0 |
| action_head_model_blocks_0_0_eager | `task6:init8` | 253 | 6 | 0.00183105 | 0.00183105 | 0.00253329 | 0 | `z` | 0 |
| action_head_model_blocks_0_1_eager | `task6:init8` | 253 | 6 | 0.00157356 | 0.00157356 | 0.00206819 | 0 | `x` | -2 |
| action_head_model_blocks_0_3_eager | `task6:init8` | 253 | 6 | 0.00183105 | 0.00183105 | 0.00262615 | 0 | `x` | 0 |
| action_head_model_blocks_1_1_eager | `task6:init8` | 253 | 6 | 0.0544739 | 0.0544739 | 0.0761407 | 0 | `x` | 0 |
| action_head_model_blocks_2_2_eager | `task6:init8` | 253 | 6 | 0.0549316 | 0.0549316 | 0.0764477 | 0 | `x` | 0 |
| action_head_model_blocks_2_3_eager | `task6:init8` | 253 | 6 | 0.0549316 | 0.0549316 | 0.0765795 | 0 | `x` | 0 |
| action_head_model_blocks_3_3_eager | `task6:init8` | 253 | 6 | 0.0553894 | 0.0553894 | 0.0771349 | 0 | `x` | 0 |

## Reading This

- `cont max` is the largest absolute drift among x/y/z/roll/pitch/yaw for that focused observation.
- `repeat span max` checks fixed-seed determinism across repeated calls; values near zero mean the spike is repeatable once seed and observation are fixed.
- A spike that survives across big-graph targets but disappears in small scopes points to graph-level numerical path rather than gripper thresholding.
