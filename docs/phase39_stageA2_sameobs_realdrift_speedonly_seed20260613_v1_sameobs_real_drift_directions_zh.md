# Phase 39 A2: same-observation real backend drift 方向估计

这个文件从 eager-controlled same-observation drift trace 中估计真实 tactic 的动作差方向。

区别于 paired closed-loop trace：这里环境始终由 FP16/eager reference action 推进，tactic 只在同一个 observation 和同一个 denoising seed 上旁路查询。因此该方向更接近真实后端的函数误差，而不是 first-divergence 之后混入状态分布偏移的轨迹差异。

- Variant: `speed_only`
- Online drift JSON: `toy_quantvla/results/phase39_stageA2_sameobs_realdrift_speedonly_seed20260613_v1_sameobs_online_drift.json`
- JSON artifact: `toy_quantvla/results/phase39_stageA2_sameobs_realdrift_speedonly_seed20260613_v1_sameobs_real_drift_directions.json`

| case | direction | window | status | steps | mean L2 | p50 L2 | p95 L2 | mean vector6 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 4:9 | real_speed_only_sameobs_mean | full | complete | 224 | 0.0001557 | 0.001735 | 0.003651 | -1.89e-05,-5.677e-05,0.0001351,-1.415e-05,3.761e-05,2.813e-05 |
| 4:9 | real_speed_only_sameobs_mean | early | complete | 75 | 0.0002734 | 0.001623 | 0.003343 | 9.508e-05,-0.0002338,8.869e-05,9.68e-06,5.051e-05,2.296e-05 |
| 6:8 | real_speed_only_sameobs_mean | full | complete | 649 | 0.0001585 | 0.001409 | 0.003068 | 0.0001054,-9.892e-05,8.619e-06,-3.158e-06,6.417e-05,5.397e-06 |
| 6:8 | real_speed_only_sameobs_mean | early | complete | 200 | 0.0001552 | 0.001458 | 0.003123 | -0.0001331,-6.073e-05,-2.248e-05,-9.499e-06,3.842e-05,-2.497e-05 |
