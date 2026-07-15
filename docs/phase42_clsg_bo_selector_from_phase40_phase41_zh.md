# CLSG-BO batch selector

## 输入

- `toy_quantvla/results/phase40_n17_active_tactic_search_batch1_15case_v1_summary.json`
- `toy_quantvla/results/phase41_clsg_bo_batch1_from_phase40_15case_v1_summary.json`

## 搜索约束

- request-window search: `end <= 30`, grid step `2`, durations `5,8,10,12,15,18,20,25,30`
- 更长 horizon、layer islands、precision profiles 应作为下一阶段显式扩展。

## 已观测窗口

| tactic | success | speedup | repair | regress | behavior score |
| --- | ---: | ---: | ---: | ---: | ---: |
| `speed_only` | 12/15 | 1.39 | 1 | 2 | 0.806 |
| `window_0_10` | 13/15 | 1.40 | 2 | 2 | 0.880 |
| `window_0_18` | 11/15 | 1.19 | 1 | 3 | 0.709 |
| `window_0_20` | 14/15 | 1.13 | 1 | 0 | 0.977 |
| `window_0_25` | 13/15 | 1.08 | 1 | 1 | 0.886 |
| `window_2_12` | 14/15 | 1.27 | 2 | 1 | 0.965 |
| `window_5_15` | 13/15 | 1.40 | 1 | 1 | 0.897 |
| `window_8_18` | 13/15 | 1.32 | 1 | 1 | 0.894 |
| `window_10_20` | 12/15 | 1.28 | 0 | 1 | 0.819 |
| `window_18_30` | 13/15 | 1.25 | 1 | 1 | 0.892 |
| `window_24_29` | 13/15 | 1.30 | 1 | 1 | 0.894 |

## 推荐下一批

| slot | tactic | reason | predicted behavior | predicted speedup | acq |
| ---: | --- | --- | ---: | ---: | ---: |
| 1 | `window_6_11` | speed_constrained | 0.936 +/- 0.027 | 1.40 +/- 0.04 | 1.529 |
| 2 | `window_4_9` | speed_constrained | 0.931 +/- 0.025 | 1.40 +/- 0.04 | 1.531 |
| 3 | `window_2_7` | speed_constrained | 0.911 +/- 0.022 | 1.40 +/- 0.03 | 1.525 |
| 4 | `window_8_13` | speed_constrained | 0.925 +/- 0.029 | 1.39 +/- 0.04 | 1.520 |
| 5 | `window_10_30` | behavior_ucb | 0.886 +/- 0.042 | 1.17 +/- 0.06 | 0.964 |
| 6 | `window_0_30` | behavior_ucb | 0.919 +/- 0.038 | 1.12 +/- 0.05 | 0.993 |

## 可直接运行的 TACTICS

```text
window_6_11 window_4_9 window_2_7 window_8_13 window_10_30 window_0_30
```

## Top candidates

| tactic | reason | predicted behavior | predicted speedup | acq |
| --- | --- | ---: | ---: | ---: |
| `window_4_9` | speed_constrained | 0.931 +/- 0.025 | 1.40 +/- 0.04 | 1.531 |
| `window_6_11` | speed_constrained | 0.936 +/- 0.027 | 1.40 +/- 0.04 | 1.529 |
| `window_2_7` | speed_constrained | 0.911 +/- 0.022 | 1.40 +/- 0.03 | 1.525 |
| `window_8_13` | speed_constrained | 0.925 +/- 0.029 | 1.39 +/- 0.04 | 1.520 |
| `window_2_10` | speed_constrained | 0.916 +/- 0.016 | 1.38 +/- 0.02 | 1.500 |
| `window_4_12` | speed_constrained | 0.922 +/- 0.016 | 1.37 +/- 0.02 | 1.497 |
| `window_6_14` | speed_constrained | 0.913 +/- 0.017 | 1.37 +/- 0.02 | 1.487 |
| `window_0_30` | behavior_ucb | 0.919 +/- 0.038 | 1.12 +/- 0.05 | 0.993 |
| `window_10_15` | behavior_ucb | 0.903 +/- 0.029 | 1.37 +/- 0.04 | 0.979 |
| `window_4_29` | behavior_ucb | 0.906 +/- 0.036 | 1.11 +/- 0.05 | 0.978 |
| `window_4_14` | behavior_ucb | 0.904 +/- 0.013 | 1.34 +/- 0.02 | 0.967 |
| `window_0_8` | behavior_ucb | 0.897 +/- 0.018 | 1.38 +/- 0.03 | 0.966 |
| `window_2_27` | behavior_ucb | 0.903 +/- 0.026 | 1.10 +/- 0.04 | 0.966 |
| `window_10_30` | behavior_ucb | 0.886 +/- 0.042 | 1.17 +/- 0.06 | 0.964 |
| `window_8_16` | behavior_ucb | 0.894 +/- 0.017 | 1.35 +/- 0.02 | 0.961 |
| `window_22_30` | behavior_ucb | 0.894 +/- 0.018 | 1.28 +/- 0.03 | 0.959 |
| `window_12_30` | behavior_ucb | 0.881 +/- 0.037 | 1.19 +/- 0.05 | 0.956 |
| `window_8_28` | behavior_ucb | 0.880 +/- 0.039 | 1.15 +/- 0.06 | 0.955 |
| `window_12_17` | behavior_ucb | 0.877 +/- 0.031 | 1.35 +/- 0.04 | 0.954 |
| `window_6_16` | behavior_ucb | 0.891 +/- 0.013 | 1.33 +/- 0.02 | 0.954 |

## 解释

这是 batch BO/active search，而不是最终 deployment selector。它把 Phase40 的窗口结果当作 warm-start，用一个小 RBF-GP 估计 behavior score 和 speedup，并同时保留 behavior-first、speed-constrained 和 exploration 候选。
下一批跑完后，应把 Phase40 与 Phase41 的 summary 一起喂回本 selector，再决定是否进入 held-out validation。
