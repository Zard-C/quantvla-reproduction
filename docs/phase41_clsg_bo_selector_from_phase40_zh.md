# CLSG-BO batch selector

## 输入

- `toy_quantvla/results/phase40_n17_active_tactic_search_batch1_15case_v1_summary.json`

## 搜索约束

- request-window search: `end <= 30`, grid step `2`, durations `5,8,10,12,15,18,20,25,30`
- 更长 horizon、layer islands、precision profiles 应作为下一阶段显式扩展。

## 已观测窗口

| tactic | success | speedup | repair | regress | behavior score |
| --- | ---: | ---: | ---: | ---: | ---: |
| `speed_only` | 12/15 | 1.39 | 1 | 2 | 0.806 |
| `window_0_20` | 14/15 | 1.13 | 1 | 0 | 0.977 |
| `window_5_15` | 13/15 | 1.40 | 1 | 1 | 0.897 |
| `window_8_18` | 13/15 | 1.32 | 1 | 1 | 0.894 |
| `window_10_20` | 12/15 | 1.28 | 0 | 1 | 0.819 |

## 推荐下一批

| slot | tactic | reason | predicted behavior | predicted speedup | acq |
| ---: | --- | --- | ---: | ---: | ---: |
| 1 | `window_0_18` | behavior_ucb | 0.965 +/- 0.019 | 1.20 +/- 0.03 | 1.027 |
| 2 | `window_0_25` | behavior_ucb | 0.954 +/- 0.037 | 1.11 +/- 0.06 | 1.026 |
| 3 | `window_0_10` | speed_constrained | 0.900 +/- 0.034 | 1.39 +/- 0.05 | 1.530 |
| 4 | `window_2_12` | speed_constrained | 0.907 +/- 0.026 | 1.39 +/- 0.04 | 1.521 |
| 5 | `window_24_29` | behavior_ucb | 0.835 +/- 0.065 | 1.28 +/- 0.10 | 0.935 |
| 6 | `window_18_30` | behavior_ucb | 0.821 +/- 0.058 | 1.25 +/- 0.09 | 0.915 |

## 可直接运行的 TACTICS

```text
window_0_18 window_0_25 window_0_10 window_2_12 window_24_29 window_18_30
```

## Top candidates

| tactic | reason | predicted behavior | predicted speedup | acq |
| --- | --- | ---: | ---: | ---: |
| `window_0_10` | speed_constrained | 0.900 +/- 0.034 | 1.39 +/- 0.05 | 1.530 |
| `window_2_12` | speed_constrained | 0.907 +/- 0.026 | 1.39 +/- 0.04 | 1.521 |
| `window_0_12` | speed_constrained | 0.922 +/- 0.032 | 1.36 +/- 0.05 | 1.493 |
| `window_2_14` | speed_constrained | 0.925 +/- 0.023 | 1.34 +/- 0.04 | 1.474 |
| `window_4_16` | speed_constrained | 0.916 +/- 0.016 | 1.32 +/- 0.03 | 1.447 |
| `window_2_15` | speed_constrained | 0.932 +/- 0.021 | 1.32 +/- 0.03 | 1.447 |
| `window_0_15` | speed_constrained | 0.949 +/- 0.026 | 1.28 +/- 0.04 | 1.415 |
| `window_2_17` | speed_constrained | 0.944 +/- 0.018 | 1.26 +/- 0.03 | 1.388 |
| `window_3_18` | speed_constrained | 0.937 +/- 0.017 | 1.25 +/- 0.03 | 1.375 |
| `window_4_19` | speed_constrained | 0.927 +/- 0.017 | 1.24 +/- 0.03 | 1.364 |
| `window_4_20` | speed_constrained | 0.929 +/- 0.019 | 1.21 +/- 0.03 | 1.339 |
| `window_0_18` | behavior_ucb | 0.965 +/- 0.019 | 1.20 +/- 0.03 | 1.027 |
| `window_0_25` | behavior_ucb | 0.954 +/- 0.037 | 1.11 +/- 0.06 | 1.026 |
| `window_2_22` | behavior_ucb | 0.952 +/- 0.023 | 1.14 +/- 0.04 | 1.015 |
| `window_0_30` | behavior_ucb | 0.925 +/- 0.058 | 1.16 +/- 0.09 | 1.014 |
| `window_2_27` | behavior_ucb | 0.936 +/- 0.045 | 1.12 +/- 0.07 | 1.014 |
| `window_2_20` | behavior_ucb | 0.952 +/- 0.018 | 1.18 +/- 0.03 | 1.013 |
| `window_4_29` | behavior_ucb | 0.915 +/- 0.052 | 1.14 +/- 0.08 | 1.000 |
| `window_4_24` | behavior_ucb | 0.929 +/- 0.032 | 1.14 +/- 0.05 | 0.998 |
| `window_4_22` | behavior_ucb | 0.930 +/- 0.024 | 1.17 +/- 0.04 | 0.996 |

## 解释

这是 batch BO/active search，而不是最终 deployment selector。它把 Phase40 的窗口结果当作 warm-start，用一个小 RBF-GP 估计 behavior score 和 speedup，并同时保留 behavior-first、speed-constrained 和 exploration 候选。
下一批跑完后，应把 Phase40 与 Phase41 的 summary 一起喂回本 selector，再决定是否进入 held-out validation。
