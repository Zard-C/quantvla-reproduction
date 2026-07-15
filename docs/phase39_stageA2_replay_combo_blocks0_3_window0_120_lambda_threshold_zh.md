# Phase 39: 闭环扰动阈值 pilot

本阶段把后端实现误差转写成显式 action perturbation，并估计不同 case、方向和 rollout 阶段上的临界扰动区间。

## 配置

- Tag: `phase39_stageA2_replay_combo_blocks0_3_window0_120_lambda_seed20260613_v1_threshold`
- Cases: `4:9, 6:8`
- Directions: `real_combo_blocks0_3_window0_120_sameobs_sequence`
- Windows: `full, early`
- Epsilons: `0.25, 0.5, 1.0`
- Deterministic policy seed base: `20260613`
- Server p50: `154` ms
- Manifest: `toy_quantvla/results/phase39_stageA2_replay_combo_blocks0_3_window0_120_lambda_seed20260613_v1_threshold_manifest.jsonl`

## Baseline

| case | outcome | steps | client p50 ms | trace |
| --- | --- | --- | --- | --- |
| 4:9 | success | 224 | 159 | toy_quantvla/results/phase39_stageA2_replay_combo_blocks0_3_window0_120_lambda_seed20260613_v1_threshold_traces/phase39_stageA2_replay_combo_blocks0_3_window0_120_lambda_seed20260613_v1_threshold_baseline/task=04--init=09--episode=001--success=True--task=put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_righ.json |
| 6:8 | success | 649 | 160.1 | toy_quantvla/results/phase39_stageA2_replay_combo_blocks0_3_window0_120_lambda_seed20260613_v1_threshold_traces/phase39_stageA2_replay_combo_blocks0_3_window0_120_lambda_seed20260613_v1_threshold_baseline/task=06--init=08--episode=002--success=True--task=put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the.json |

## 临界扰动区间

记号：`0.006:S/224` 表示 epsilon=0.006 时 success，episode 有 224 个 policy steps。临界区间是 first flip 的区间，不假设结果严格单调；若 `non-monotonic=yes`，说明更大 epsilon 后结果又翻回来了，需要单独看 trace。

| case | direction | window | steps | baseline | epsilon crit | non-monotonic | outcomes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 4:9 | real_combo_blocks0_3_window0_120_sameobs_sequence | early | [0, 75) | success | > 1.0 | no | 0.25:S/224, 0.5:S/224, 1:S/224 |
| 4:9 | real_combo_blocks0_3_window0_120_sameobs_sequence | full | full | success | > 1.0 | no | 0.25:S/222, 0.5:S/223, 1:S/224 |
| 6:8 | real_combo_blocks0_3_window0_120_sameobs_sequence | early | [0, 200) | success | > 1.0 | no | 0.25:S/229, 0.5:S/840, 1:S/627 |
| 6:8 | real_combo_blocks0_3_window0_120_sameobs_sequence | full | full | success | > 1.0 | no | 0.25:S/229, 0.5:S/632, 1:S/464 |

## 已观察到的 rollout flip

_No flips observed yet._

## 解读口径

- 如果同一个 case 上 `y/yaw` 的 first-flip epsilon 明显小于 `z`，支持 not all dims are equal。
- 如果 `early` 的 first-flip epsilon 小于 `late`，支持 not all durations are equal。
- 如果某个方向全成功，报告为 `> max epsilon`，表示当前网格还没有碰到闭环失败边界。
- 如果出现非单调，不能把阈值当成确定常数；它更像是这个闭环系统在该扰动族上的局部 basin 边界采样。

- JSON artifact: `toy_quantvla/results/phase39_threshold_pilot_summary.json`
