# Phase 16.5 Step-focused Replay Summary

Boundary: eager-controlled same-observation replay around the `task6:init8` action spike region.

## Target Ranking

| target | top case | top step | top dim | cont max | p99 at step | repeat span max | compiled p50 mean | eager p50 mean |
|---|---|---:|---|---:|---:|---:|---:|---:|
| action_head_model | `task6:init8` | 253 | `x` | 0.0549316 | 0.0549316 | 0 | 62.4 ms | 142.9 ms |
| action_head_model_blocks_8_15_eager | `task6:init8` | 253 | `x` | 0.0549316 | 0.0549316 | 0 | 83.5 ms | 117.0 ms |
| action_head_dit_attn_all | `task6:init8` | 258 | `x` | 0.0019598 | 0.0019598 | 0 | 103.0 ms | 107.5 ms |

## Per-Step Rows

| target | case | step | samples | cont max | cont p99 | L2 p99 | repeat span max | top dim | seed offset |
|---|---|---:|---:|---:|---:|---:|---:|---|---:|
| action_head_dit_attn_all | `task6:init8` | 248 | 15 | 0.000915527 | 0.000915527 | 0.00104494 | 0 | `z` | -2 |
| action_head_dit_attn_all | `task6:init8` | 249 | 15 | 0.00100136 | 0.00100136 | 0.00113632 | 0 | `z` | 2 |
| action_head_dit_attn_all | `task6:init8` | 250 | 15 | 0.000915527 | 0.000915527 | 0.00103114 | 0 | `y` | 2 |
| action_head_dit_attn_all | `task6:init8` | 251 | 15 | 0.000915527 | 0.000915527 | 0.000972018 | 0 | `z` | -2 |
| action_head_dit_attn_all | `task6:init8` | 252 | 15 | 0.000915527 | 0.000915527 | 0.000944681 | 0 | `y` | -2 |
| action_head_dit_attn_all | `task6:init8` | 253 | 15 | 0.00183105 | 0.00183105 | 0.00254211 | 0 | `x` | 0 |
| action_head_dit_attn_all | `task6:init8` | 254 | 15 | 0.00137329 | 0.00137329 | 0.00145349 | 0 | `y` | -2 |
| action_head_dit_attn_all | `task6:init8` | 255 | 15 | 0.00125885 | 0.00125885 | 0.00168635 | 0 | `z` | -1 |
| action_head_dit_attn_all | `task6:init8` | 256 | 15 | 0.000915527 | 0.000915527 | 0.000982101 | 0 | `y` | 1 |
| action_head_dit_attn_all | `task6:init8` | 257 | 15 | 0.000877321 | 0.000877321 | 0.000893518 | 0 | `pitch` | 2 |
| action_head_dit_attn_all | `task6:init8` | 258 | 15 | 0.0019598 | 0.0019598 | 0.00297917 | 0 | `x` | 1 |
| action_head_model | `task6:init8` | 248 | 15 | 0.00274658 | 0.00274658 | 0.00357605 | 0 | `z` | -2 |
| action_head_model | `task6:init8` | 249 | 15 | 0.00208855 | 0.00208855 | 0.00278252 | 0 | `z` | 2 |
| action_head_model | `task6:init8` | 250 | 15 | 0.00137329 | 0.00137329 | 0.0013852 | 0 | `y` | -1 |
| action_head_model | `task6:init8` | 251 | 15 | 0.00137329 | 0.00137329 | 0.00211291 | 0 | `x` | 2 |
| action_head_model | `task6:init8` | 252 | 15 | 0.00160217 | 0.00160217 | 0.00172564 | 0 | `z` | 1 |
| action_head_model | `task6:init8` | 253 | 15 | 0.0549316 | 0.0549316 | 0.0773858 | 0 | `x` | 0 |
| action_head_model | `task6:init8` | 254 | 15 | 0.00137329 | 0.00137329 | 0.0016899 | 0 | `y` | -2 |
| action_head_model | `task6:init8` | 255 | 15 | 0.000915527 | 0.000915527 | 0.0012005 | 0 | `x` | -2 |
| action_head_model | `task6:init8` | 256 | 15 | 0.00137329 | 0.00137329 | 0.00164918 | 0 | `x` | 1 |
| action_head_model | `task6:init8` | 257 | 15 | 0.00274658 | 0.00274658 | 0.00288756 | 0 | `z` | 2 |
| action_head_model | `task6:init8` | 258 | 15 | 0.00424862 | 0.00424862 | 0.00624536 | 0 | `x` | 1 |
| action_head_model_blocks_8_15_eager | `task6:init8` | 248 | 15 | 0.00457764 | 0.00457764 | 0.00495258 | 0 | `z` | -2 |
| action_head_model_blocks_8_15_eager | `task6:init8` | 249 | 15 | 0.00185966 | 0.00185966 | 0.00198003 | 0 | `z` | 2 |
| action_head_model_blocks_8_15_eager | `task6:init8` | 250 | 15 | 0.00183105 | 0.00183105 | 0.00208654 | 0 | `z` | -2 |
| action_head_model_blocks_8_15_eager | `task6:init8` | 251 | 15 | 0.00137329 | 0.00137329 | 0.00211234 | 0 | `x` | 2 |
| action_head_model_blocks_8_15_eager | `task6:init8` | 252 | 15 | 0.00205994 | 0.00205994 | 0.00216594 | 0 | `z` | 1 |
| action_head_model_blocks_8_15_eager | `task6:init8` | 253 | 15 | 0.0549316 | 0.0549316 | 0.0757393 | 0 | `x` | 0 |
| action_head_model_blocks_8_15_eager | `task6:init8` | 254 | 15 | 0.00146806 | 0.00146806 | 0.00178191 | 0 | `x` | 0 |
| action_head_model_blocks_8_15_eager | `task6:init8` | 255 | 15 | 0.000957608 | 0.000957608 | 0.00142451 | 0 | `x` | -1 |
| action_head_model_blocks_8_15_eager | `task6:init8` | 256 | 15 | 0.00170094 | 0.00170094 | 0.00194143 | 0 | `z` | -1 |
| action_head_model_blocks_8_15_eager | `task6:init8` | 257 | 15 | 0.00183105 | 0.00183105 | 0.00210814 | 0 | `z` | 2 |
| action_head_model_blocks_8_15_eager | `task6:init8` | 258 | 15 | 0.0039053 | 0.0039053 | 0.00523859 | 0 | `x` | 1 |

## Reading This

- `cont max` is the largest absolute drift among x/y/z/roll/pitch/yaw for that focused observation.
- `repeat span max` checks fixed-seed determinism across repeated calls; values near zero mean the spike is repeatable once seed and observation are fixed.
- A spike that survives across big-graph targets but disappears in small scopes points to graph-level numerical path rather than gripper thresholding.
