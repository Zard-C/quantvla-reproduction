# Phase 39 A2: same-observation real backend drift 方向估计

这个文件从 eager-controlled same-observation drift trace 中估计真实 tactic 的动作差方向。

区别于 paired closed-loop trace：这里环境始终由 FP16/eager reference action 推进，tactic 只在同一个 observation 和同一个 denoising seed 上旁路查询。因此该方向更接近真实后端的函数误差，而不是 first-divergence 之后混入状态分布偏移的轨迹差异。

- Variant: `speed_only`
- Online drift JSON: `toy_quantvla/results/phase39_stageA2_sameobs_realdrift_combo_blocks0_3_window0_120_seed20260613_v1_sameobs_online_drift.json`
- JSON artifact: `toy_quantvla/results/phase39_stageA2_sameobs_realdrift_combo_blocks0_3_window0_120_seed20260613_v1_sameobs_real_drift_directions.json`

| case | direction | window | status | steps | mean L2 | p50 L2 | p95 L2 | mean vector6 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 4:9 | real_speed_only_sameobs_mean | full | complete | 224 | 0.0001813 | 0 | 0.002847 | -4.542e-05,0.0001733,-6.322e-06,-1.286e-05,-5.368e-06,-2.382e-05 |
| 4:9 | real_speed_only_sameobs_mean | early | zero_vector | 75 | 0 | 0 | 0 | 0,0,0,0,0,0 |
| 6:8 | real_speed_only_sameobs_mean | full | complete | 649 | 0.0002145 | 0.001167 | 0.002568 | -5.983e-05,-0.00013,-0.0001583,-1.214e-05,-1.722e-05,-4.286e-06 |
| 6:8 | real_speed_only_sameobs_mean | early | complete | 200 | 0.0003583 | 0 | 0.002153 | -0.0001886,-0.0001748,-0.0002473,-4.573e-06,-6.843e-06,-3.223e-05 |
