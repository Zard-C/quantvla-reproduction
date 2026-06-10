# Phase 16 Compile Scope Continuous Drift Sweep

Boundary: eager-controlled same-observation replay. The simulator is stepped by eager FP16 actions; compiled actions are measured on the same observations with the same denoising seeds.

## Ranking

| target | cases | valid steps | window done | cont max p99 | cont max max | cont L2 p99 | compiled p50 | eager p50 | time ratio p50 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| action_head_dit_ff_all | `4:9,6:8` | 484 | 1/2 | 0.00297546 | 0.00640869 | 0.00321614 | 123.5 ms | 90.5 ms | 1.28 |
| action_head_dit_attn_all | `4:9,6:8` | 484 | 1/2 | 0.0030899 | 0.00640869 | 0.00322269 | 102.9 ms | 92.3 ms | 1.11 |
| action_head_model_blocks_6_15_eager | `4:9,6:8` | 484 | 1/2 | 0.00411987 | 0.0549316 | 0.00422291 | 93.9 ms | 102.3 ms | 0.91 |
| action_head_model | `4:9,6:8` | 484 | 1/2 | 0.00434875 | 0.0549316 | 0.00499544 | 63.3 ms | 130.6 ms | 0.497 |
| action_head_model_blocks_8_15_eager | `4:9,6:8` | 484 | 1/2 | 0.0044632 | 0.0549316 | 0.00465726 | 87.9 ms | 105.7 ms | 0.838 |
| action_head_dit_blocks_0_7 | `4:9,6:8` | 0 | 0/2 | 0 | 0 | 0 | 0.0 ms | 0.0 ms | 0 |

## Per-Target Episodes

### `action_head_dit_ff_all`

| case | window done | steps | exception | cont max p99 | cont max max | translation L2 p99 | rotation L2 p99 |
|---|---:|---:|---|---:|---:|---:|---:|
| task4:init9 | True | 224 |  | 0.00340462 | 0.00640869 | 0.00343525 | 0.000670738 |
| task6:init8 | False | 260 |  | 0.00232458 | 0.00274658 | 0.00232758 | 0.000682534 |

### `action_head_dit_attn_all`

| case | window done | steps | exception | cont max p99 | cont max max | translation L2 p99 | rotation L2 p99 |
|---|---:|---:|---|---:|---:|---:|---:|
| task4:init9 | True | 224 |  | 0.00320435 | 0.00640869 | 0.00365898 | 0.000612314 |
| task6:init8 | False | 260 |  | 0.00274658 | 0.00331879 | 0.00290418 | 0.000599354 |

### `action_head_model_blocks_6_15_eager`

| case | window done | steps | exception | cont max p99 | cont max max | translation L2 p99 | rotation L2 p99 |
|---|---:|---:|---|---:|---:|---:|---:|
| task4:init9 | True | 224 |  | 0.00411987 | 0.00437737 | 0.00422037 | 0.000895296 |
| task6:init8 | False | 260 |  | 0.00366211 | 0.0549316 | 0.00413821 | 0.000940751 |

### `action_head_model`

| case | window done | steps | exception | cont max p99 | cont max max | translation L2 p99 | rotation L2 p99 |
|---|---:|---:|---|---:|---:|---:|---:|
| task4:init9 | True | 224 |  | 0.00457764 | 0.0050354 | 0.00494254 | 0.00106983 |
| task6:init8 | False | 260 |  | 0.00389099 | 0.0549316 | 0.00476272 | 0.00101357 |

### `action_head_model_blocks_8_15_eager`

| case | window done | steps | exception | cont max p99 | cont max max | translation L2 p99 | rotation L2 p99 |
|---|---:|---:|---|---:|---:|---:|---:|
| task4:init9 | True | 224 |  | 0.00366211 | 0.00509262 | 0.00419547 | 0.000809733 |
| task6:init8 | False | 260 |  | 0.00457764 | 0.0549316 | 0.00460047 | 0.00092009 |

### `action_head_dit_blocks_0_7`

| case | window done | steps | exception | cont max p99 | cont max max | translation L2 p99 | rotation L2 p99 |
|---|---:|---:|---|---:|---:|---:|---:|
| task4:init9 | False | 0 | `Error: accessing tensor output of CUDAGraphs that has been overwritten by a subsequent run. Stack trace: File "/root/autodl-tmp/Isaac-GR00T-n1.5/gr00t/model/action_head/cross_at...` | 0 | 0 | 0 | 0 |
| task6:init8 | False | 0 |  | 0 | 0 | 0 | 0 |

## Reading This

- Lower continuous p99/max is better for behavior transparency.
- Lower compiled p50 is better for speed, but small scopes can lose the large-graph benefit.
- Candidates should pass this same-observation screen before any closed-loop rollout expansion.
