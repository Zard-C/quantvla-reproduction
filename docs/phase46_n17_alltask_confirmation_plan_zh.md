# Phase46 计划：N1.7 all-task confirmation fold

## 目的

Phase45 all-task stress 里，`blocks0_3_window_2_12` 成为当前最值得继续确认的 hybrid candidate：

```text
18/20, 1.06x p50 speedup, 0 repair / 1 regression
```

Phase46 的目的不是继续扩大 BO 搜索空间，而是做 confirmation fold：

> 在新的 all-task init 上确认 `blocks0_3_window_2_12` 是否稳定保持比 `window_2_12` 更低 closed-loop risk，同时比 `window_0_20` 更有速度收益。

## 设置

```text
tasks 0-9 x init 40/41/42 = 30 cases
```

候选 tactic：

```text
fp16
speed_only
window_0_20
window_2_12
blocks0_3_window_2_12
```

总 rollout 数：

```text
5 tactics x 30 cases = 150 rollouts
```

默认运行：

```bash
bash toy_quantvla/run_phase46_n17_alltask_confirmation.sh
```

## 判定标准

核心比较：

| 比较 | 想回答的问题 |
| --- | --- |
| `blocks0_3_window_2_12` vs `window_2_12` | hybrid 是否降低 paired regression，且速度不更差太多 |
| `blocks0_3_window_2_12` vs `window_0_20` | hybrid 是否在相近 behavior 下更快 |
| `speed_only` vs hybrid | aggressive compile 的闭环代价是否仍明显更高 |

如果 `blocks0_3_window_2_12` 在 Phase46 中仍然满足：

```text
regressions <= window_2_12
speedup >= window_0_20
success >= window_2_12 or close
```

则可以把它写成当前 N1.7 hybrid search 的 confirmed Pareto candidate。

如果它退化，则论文应强调：

```text
hybrid CLSG-BO can generate useful candidates, but the final tactic remains slice-dependent and must be validated on held-out folds.
```

## 预期用法

Phase46 跑完后，建议停止扩展实验，把 Phase40-46 组织成一条完整实验链：

```text
duration-only BO
-> held-out validation
-> hybrid layer-duration BO
-> all-task stress
-> confirmation fold
```
