# Phase 39: 闭环扰动阈值 pilot

本阶段把后端实现误差转写成显式 action perturbation，并估计不同 case、方向和 rollout 阶段上的临界扰动区间。

## 本轮结论

这轮 ultra-low sweep 已完整完成，`complete_rows=15`，`missing=0`。

核心结果是：在 `task4:init9` 这个 FP16 成功但闭环脆弱的样本上，`y` 方向扰动无论作用于 `full` horizon 还是只作用于 `early` window `[0,75)`，first flip 都落在同一个区间：

```text
(1e-5, 3e-5]
```

也就是说，`1e-5` 仍成功，而 `3e-5` 开始稳定翻成 horizon failure。这个量级远小于很多工程局部数值 tolerance 的直觉阈值，因此它不能被解释成普通 open-loop numerical drift，而应该被解释成闭环 basin boundary 上的局部敏感性。

本阶段给论文提供的关键论据是：

```text
局部数值误差很小，不代表闭环安全；
扰动是否危险取决于方向、阶段、环境动力学、policy feedback 和任务 margin。
```

下一阶段 Stage A2：`real-backend-drift-aligned pilot` 已完成，综合报告见 [`docs/phase39_stageA2_real_backend_replay_summary_zh.md`](phase39_stageA2_real_backend_replay_summary_zh.md)。Stage A2 使用 same-observation diagnostic 在 FP16 nominal observation 上同时查询 reference 和 tactic，记录真实后端 drift sequence，再沿这个方向做闭环 replay。结果显示 `speed_only` 的真实 drift sequence 在多个 case/window 上导致 S->F，而 `blocks0-3 + window0-120` protected tactic 在 `lambda <= 1.0` 的 replay 中全部保持成功。

## 配置

- Tag: `phase39_ultralow_y_task4_seed20260613_v1`
- Cases: `4:9`
- Directions: `y`
- Windows: `full, early`
- Epsilons: `1e-06, 3e-06, 1e-05, 3e-05, 0.0001, 0.0003, 0.0005`
- Deterministic policy seed base: `20260613`
- Server p50: `156.2` ms
- Manifest: `toy_quantvla/results/phase39_ultralow_y_task4_seed20260613_v1_manifest.jsonl`

## Baseline

| case | outcome | steps | client p50 ms | trace |
| --- | --- | --- | --- | --- |
| 4:9 | success | 224 | 160.2 | toy_quantvla/results/phase39_ultralow_y_task4_seed20260613_v1_traces/phase39_ultralow_y_task4_seed20260613_v1_baseline/task=04--init=09--episode=001--success=True--task=put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_righ.json |

## 临界扰动区间

记号：`0.006:S/224` 表示 epsilon=0.006 时 success，episode 有 224 个 policy steps。临界区间是 first flip 的区间，不假设结果严格单调；若 `non-monotonic=yes`，说明更大 epsilon 后结果又翻回来了，需要单独看 trace。

| case | direction | window | steps | baseline | epsilon crit | non-monotonic | outcomes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 4:9 | y | early | [0, 75) | success | (1e-05, 3e-05] | no | 1e-06:S/224, 3e-06:S/222, 1e-05:S/221, 3e-05:F/991, 0.0001:F/991, 0.0003:F/991, 0.0005:F/991 |
| 4:9 | y | full | full | success | (1e-05, 3e-05] | no | 1e-06:S/224, 3e-06:S/219, 1e-05:S/218, 3e-05:F/991, 0.0001:F/991, 0.0003:F/991, 0.0005:F/991 |

## 已观察到的 rollout flip

| case | direction | window | epsilon | flip | steps | trace |
| --- | --- | --- | --- | --- | --- | --- |
| 4:9 | y | early | 3e-05 | S->F | 991 | toy_quantvla/results/phase39_ultralow_y_task4_seed20260613_v1_traces/phase39_ultralow_y_task4_seed20260613_v1_case_t4_i9_dir_y_win_early_eps_0p00003/task=04--init=09--episode=001--success=False--task=put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_righ.json |
| 4:9 | y | early | 0.0001 | S->F | 991 | toy_quantvla/results/phase39_ultralow_y_task4_seed20260613_v1_traces/phase39_ultralow_y_task4_seed20260613_v1_case_t4_i9_dir_y_win_early_eps_0p0001/task=04--init=09--episode=001--success=False--task=put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_righ.json |
| 4:9 | y | early | 0.0003 | S->F | 991 | toy_quantvla/results/phase39_ultralow_y_task4_seed20260613_v1_traces/phase39_ultralow_y_task4_seed20260613_v1_case_t4_i9_dir_y_win_early_eps_0p0003/task=04--init=09--episode=001--success=False--task=put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_righ.json |
| 4:9 | y | early | 0.0005 | S->F | 991 | toy_quantvla/results/phase39_ultralow_y_task4_seed20260613_v1_traces/phase39_ultralow_y_task4_seed20260613_v1_case_t4_i9_dir_y_win_early_eps_0p0005/task=04--init=09--episode=001--success=False--task=put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_righ.json |
| 4:9 | y | full | 3e-05 | S->F | 991 | toy_quantvla/results/phase39_ultralow_y_task4_seed20260613_v1_traces/phase39_ultralow_y_task4_seed20260613_v1_case_t4_i9_dir_y_win_full_eps_0p00003/task=04--init=09--episode=001--success=False--task=put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_righ.json |
| 4:9 | y | full | 0.0001 | S->F | 991 | toy_quantvla/results/phase39_ultralow_y_task4_seed20260613_v1_traces/phase39_ultralow_y_task4_seed20260613_v1_case_t4_i9_dir_y_win_full_eps_0p0001/task=04--init=09--episode=001--success=False--task=put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_righ.json |
| 4:9 | y | full | 0.0003 | S->F | 991 | toy_quantvla/results/phase39_ultralow_y_task4_seed20260613_v1_traces/phase39_ultralow_y_task4_seed20260613_v1_case_t4_i9_dir_y_win_full_eps_0p0003/task=04--init=09--episode=001--success=False--task=put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_righ.json |
| 4:9 | y | full | 0.0005 | S->F | 991 | toy_quantvla/results/phase39_ultralow_y_task4_seed20260613_v1_traces/phase39_ultralow_y_task4_seed20260613_v1_case_t4_i9_dir_y_win_full_eps_0p0005/task=04--init=09--episode=001--success=False--task=put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_righ.json |

## 解读口径

- 如果同一个 case 上 `y/yaw` 的 first-flip epsilon 明显小于 `z`，支持 not all dims are equal。
- 如果 `early` 的 first-flip epsilon 小于 `late`，支持 not all durations are equal。
- 如果某个方向全成功，报告为 `> max epsilon`，表示当前网格还没有碰到闭环失败边界。
- 如果出现非单调，不能把阈值当成确定常数；它更像是这个闭环系统在该扰动族上的局部 basin 边界采样。

- JSON artifact: `toy_quantvla/results/phase39_threshold_pilot_summary.json`
