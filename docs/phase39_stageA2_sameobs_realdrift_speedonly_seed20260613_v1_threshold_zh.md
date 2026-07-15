# Phase 39: 闭环扰动阈值 pilot

本阶段把后端实现误差转写成显式 action perturbation，并估计不同 case、方向和 rollout 阶段上的临界扰动区间。

## 配置

- Tag: `phase39_stageA2_sameobs_realdrift_speedonly_seed20260613_v1_threshold`
- Cases: `4:9, 6:8`
- Directions: `real_speed_only_sameobs_mean`
- Windows: `full, early`
- Epsilons: `1e-05, 2e-05, 3e-05`
- Deterministic policy seed base: `20260613`
- Server p50: `156.4` ms
- Manifest: `toy_quantvla/results/phase39_stageA2_sameobs_realdrift_speedonly_seed20260613_v1_threshold_manifest.jsonl`

## Baseline

| case | outcome | steps | client p50 ms | trace |
| --- | --- | --- | --- | --- |
| 4:9 | success | 224 | 160.5 | toy_quantvla/results/phase39_stageA2_sameobs_realdrift_speedonly_seed20260613_v1_threshold_traces/phase39_stageA2_sameobs_realdrift_speedonly_seed20260613_v1_threshold_baseline/task=04--init=09--episode=001--success=True--task=put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_righ.json |
| 6:8 | success | 649 | 162.2 | toy_quantvla/results/phase39_stageA2_sameobs_realdrift_speedonly_seed20260613_v1_threshold_traces/phase39_stageA2_sameobs_realdrift_speedonly_seed20260613_v1_threshold_baseline/task=06--init=08--episode=002--success=True--task=put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the.json |

## 临界扰动区间

记号：`0.006:S/224` 表示 epsilon=0.006 时 success，episode 有 224 个 policy steps。临界区间是 first flip 的区间，不假设结果严格单调；若 `non-monotonic=yes`，说明更大 epsilon 后结果又翻回来了，需要单独看 trace。

| case | direction | window | steps | baseline | epsilon crit | non-monotonic | outcomes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 4:9 | real_speed_only_sameobs_mean | early | [0, 75) | success | (1e-05, 2e-05] | yes | 1e-05:S/223, 2e-05:F/991, 3e-05:S/223 |
| 4:9 | real_speed_only_sameobs_mean | full | full | success | (1e-05, 2e-05] | no | 1e-05:S/223, 2e-05:F/991, 3e-05:F/991 |
| 6:8 | real_speed_only_sameobs_mean | early | [0, 200) | success | (1e-05, 2e-05] | no | 1e-05:S/744, 2e-05:F/991, 3e-05:F/991 |
| 6:8 | real_speed_only_sameobs_mean | full | full | success | <= 1e-05 | no | 1e-05:F/991, 2e-05:F/991, 3e-05:F/991 |

## 已观察到的 rollout flip

| case | direction | window | epsilon | flip | steps | trace |
| --- | --- | --- | --- | --- | --- | --- |
| 4:9 | real_speed_only_sameobs_mean | early | 2e-05 | S->F | 991 | toy_quantvla/results/phase39_stageA2_sameobs_realdrift_speedonly_seed20260613_v1_threshold_traces/phase39_stageA2_sameobs_realdrift_speedonly_seed20260613_v1_threshold_case_t4_i9_dir_real_speed_only_sameobs_mean_win_early_eps_0p00002/task=04--init=09--episode=001--success=False--task=put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_righ.json |
| 4:9 | real_speed_only_sameobs_mean | full | 2e-05 | S->F | 991 | toy_quantvla/results/phase39_stageA2_sameobs_realdrift_speedonly_seed20260613_v1_threshold_traces/phase39_stageA2_sameobs_realdrift_speedonly_seed20260613_v1_threshold_case_t4_i9_dir_real_speed_only_sameobs_mean_win_full_eps_0p00002/task=04--init=09--episode=001--success=False--task=put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_righ.json |
| 4:9 | real_speed_only_sameobs_mean | full | 3e-05 | S->F | 991 | toy_quantvla/results/phase39_stageA2_sameobs_realdrift_speedonly_seed20260613_v1_threshold_traces/phase39_stageA2_sameobs_realdrift_speedonly_seed20260613_v1_threshold_case_t4_i9_dir_real_speed_only_sameobs_mean_win_full_eps_0p00003/task=04--init=09--episode=001--success=False--task=put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_righ.json |
| 6:8 | real_speed_only_sameobs_mean | early | 2e-05 | S->F | 991 | toy_quantvla/results/phase39_stageA2_sameobs_realdrift_speedonly_seed20260613_v1_threshold_traces/phase39_stageA2_sameobs_realdrift_speedonly_seed20260613_v1_threshold_case_t6_i8_dir_real_speed_only_sameobs_mean_win_early_eps_0p00002/task=06--init=08--episode=001--success=False--task=put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the.json |
| 6:8 | real_speed_only_sameobs_mean | early | 3e-05 | S->F | 991 | toy_quantvla/results/phase39_stageA2_sameobs_realdrift_speedonly_seed20260613_v1_threshold_traces/phase39_stageA2_sameobs_realdrift_speedonly_seed20260613_v1_threshold_case_t6_i8_dir_real_speed_only_sameobs_mean_win_early_eps_0p00003/task=06--init=08--episode=001--success=False--task=put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the.json |
| 6:8 | real_speed_only_sameobs_mean | full | 1e-05 | S->F | 991 | toy_quantvla/results/phase39_stageA2_sameobs_realdrift_speedonly_seed20260613_v1_threshold_traces/phase39_stageA2_sameobs_realdrift_speedonly_seed20260613_v1_threshold_case_t6_i8_dir_real_speed_only_sameobs_mean_win_full_eps_0p00001/task=06--init=08--episode=001--success=False--task=put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the.json |
| 6:8 | real_speed_only_sameobs_mean | full | 2e-05 | S->F | 991 | toy_quantvla/results/phase39_stageA2_sameobs_realdrift_speedonly_seed20260613_v1_threshold_traces/phase39_stageA2_sameobs_realdrift_speedonly_seed20260613_v1_threshold_case_t6_i8_dir_real_speed_only_sameobs_mean_win_full_eps_0p00002/task=06--init=08--episode=001--success=False--task=put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the.json |
| 6:8 | real_speed_only_sameobs_mean | full | 3e-05 | S->F | 991 | toy_quantvla/results/phase39_stageA2_sameobs_realdrift_speedonly_seed20260613_v1_threshold_traces/phase39_stageA2_sameobs_realdrift_speedonly_seed20260613_v1_threshold_case_t6_i8_dir_real_speed_only_sameobs_mean_win_full_eps_0p00003/task=06--init=08--episode=001--success=False--task=put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the.json |

## 解读口径

- 如果同一个 case 上 `y/yaw` 的 first-flip epsilon 明显小于 `z`，支持 not all dims are equal。
- 如果 `early` 的 first-flip epsilon 小于 `late`，支持 not all durations are equal。
- 如果某个方向全成功，报告为 `> max epsilon`，表示当前网格还没有碰到闭环失败边界。
- 如果出现非单调，不能把阈值当成确定常数；它更像是这个闭环系统在该扰动族上的局部 basin 边界采样。

- JSON artifact: `toy_quantvla/results/phase39_threshold_pilot_summary.json`
