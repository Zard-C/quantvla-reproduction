# Phase 39: real backend drift 方向估计

这个文件从已有 paired closed-loop traces 中估计真实 tactic 的动作差方向，用于后续 `real_speed_only_mean` 方向的 threshold sweep。

注意：这些方向来自闭环轨迹对齐，不是严格 same-observation replay。first divergence 之后的差异会混入状态分布偏移，因此它适合作为 real tactic footprint，而不是纯函数误差的无偏估计。

- Variant: `speed_only`
- Baseline trace dir: `toy_quantvla/results/phase28A_proxy_guided_15case_v1_selected_trace/baseline`
- Variant trace dir: `toy_quantvla/results/phase28A_proxy_guided_15case_v1_selected_trace/speed_only`
- JSON artifact: `toy_quantvla/results/phase39_real_drift_directions.json`

| case | direction | window | status | steps | mean L2 | p50 L2 | p95 L2 | mean vector6 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 4:9 | real_speed_only_mean | full | complete | 222 | 0.006027 | 0.09203 | 0.329 | -0.002769,-5.535e-05,-0.004977,0.0009452,0.0008362,-0.001514 |
| 4:9 | real_speed_only_mean | early | complete | 75 | 0.02759 | 0.005926 | 0.3071 | 0.02247,-0.01399,-0.006017,0.001595,-0.0006633,-0.004624 |
| 4:9 | real_speed_only_mean | late | complete | 72 | 0.03421 | 0.1348 | 0.3961 | -0.02353,-0.02371,-0.006559,0.002963,-0.001075,-0.001213 |
| 6:8 | real_speed_only_mean | full | missing_trace | - | - | - | - | - |
| 6:8 | real_speed_only_mean | early | missing_trace | - | - | - | - | - |
| 6:8 | real_speed_only_mean | late | missing_trace | - | - | - | - | - |
