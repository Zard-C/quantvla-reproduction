# Phase 28A Selected Trace Divergence

这份报告使用从 5090 `/tmp` 拉回的 selected rollout traces，分析 `baseline`、`speed_only`、`proxy_blocks8_15` 在关键病例上的 first-divergence。阈值是经验诊断阈值：`libero_action` L2 大于 `0.05` 记为动作开始分叉，EEF 位置差大于 `1cm` / `5cm` 记为状态分叉。

注意：这里的 `S/F + steps` 来自 rollout trace 里的 env step 数；失败样本在 client latency summary 里可能显示为 `F991`，因为官方 eval loop 在 terminated episode 后还会多发一次 action request。

## speed-only vs proxy_blocks8_15

| case | speed | proxy | first action >0.05 | first EEF >1cm | first EEF >5cm | max EEF m | action p50 | action p95 | note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 4:6 | F990 | S241 | 35 | 83 | 192 | 0.17997 | 0.07309 | 2.30151 | speed-only regression; proxy_blocks8_15 restores a short successful branch close to baseline. |
| 6:0 | F990 | S206 | 62 | 120 | 143 | 0.13179 | 0.06457 | 2.25756 | speed-only regression; proxy_blocks8_15 restores the baseline branch with small EEF deviation. |
| 8:10 | F990 | S604 | 87 | 141 | 467 | 0.48578 | 0.07955 | 0.97416 | speed-only regression; proxy_blocks8_15 repairs outcome but follows a longer branch than baseline. |
| 8:9 | S476 | S424 | 55 | 157 | 166 | 0.38552 | 0.28122 | 2.18071 | speed-only repair; proxy_blocks8_15 preserves the beneficial success branch. |
| 4:9 | S222 | F990 | 57 | 104 | 125 | 0.18282 | 0.06748 | 2.22853 | proxy_blocks8_15 regression; protection changes a baseline/speed-only success into horizon failure. |

这张表说明两个事实。第一，动作差异通常早于可见状态分叉：例如 `4:6` 在 step `35` 已经出现 action L2 > 0.05，但 EEF 到 step `83` 才超过 1cm，到 step `192` 才超过 5cm。第二，成功/失败翻转不要求一开始就有巨大状态误差；小的动作分叉会先进入闭环，再经过几十到上百步放大成不同轨迹分支。

## baseline vs proxy_blocks8_15

| case | baseline | proxy | first EEF >1cm | first EEF >5cm | max EEF m | action p50 | action p95 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 4:6 | S245 | S241 | 74 | - | 0.01787 | 0.04817 | 0.21956 |
| 6:0 | S210 | S206 | 144 | - | 0.04941 | 0.02182 | 0.39075 |
| 8:10 | S385 | S604 | 121 | 132 | 0.47493 | 0.14755 | 0.86005 |
| 8:9 | F990 | S424 | 139 | 149 | 0.25572 | 0.4042 | 2.0917 |
| 4:9 | S224 | F990 | 59 | 115 | 0.17255 | 0.25241 | 2.2168 |

`4:6` 和 `6:0` 是最干净的 repair 证据：proxy_blocks8_15 与 baseline 都成功，而且最大 EEF 差分别只有约 `1.8cm` 和 `4.9cm`，没有超过 5cm 的大分叉。这说明保护 `blocks8-15` 不只是随机改变结果，而是在这些 case 上把 speed-only 拉回了接近 baseline 的成功 basin。

`8:10` 则更复杂：baseline 与 proxy 都成功，但 proxy 轨迹明显偏离 baseline，并且成功步数从 `385` 变成 `604`。这更像是 proxy 找到另一个可行 branch，而不是严格复刻 FP16。

`4:9` 是风险样本：baseline 和 speed-only 都成功，但 proxy_blocks8_15 失败，并且 EEF 在 step `115` 超过 5cm。这提醒我们，保护敏感层不是单调安全操作；它本身也会重分配闭环轨迹。

## 结论

Phase 28A 的 trace 证据支持一个更具体的工程判断：closed-loop proxy 不应该只输出一个“保护哪些层”的静态答案，还需要用 paired traces 检查它到底是在恢复 baseline basin、进入另一条可行 branch，还是制造新的失败 branch。

对当前配置来说，`proxy_blocks8_15` 是更好的 speed-quality 折中，但还不是最终方案。它修复了 speed-only 在 `4:6, 6:0, 8:10` 的退化，并保住 `8:9` 的 beneficial success；代价是 `4:9` 出现新 regression，且 server p50 加速从 `1.68x` 降到 `1.26x`。

## 产物

- Selected traces: `toy_quantvla/results/phase28A_proxy_guided_15case_v1_selected_trace`
- Divergence JSON: `toy_quantvla/results/phase28A_proxy_guided_15case_v1_selected_trace_divergence.json`
