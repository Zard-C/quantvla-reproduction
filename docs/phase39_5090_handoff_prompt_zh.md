# Phase 39 5090 handoff prompt

目标：开展 closed-loop perturbation threshold / deployment error budget 实验。

请先阅读：

```text
docs/phase39_closed_loop_perturbation_budget_plan_zh.md
docs/phase25_anisotropic_sensitivity_report_zh.md
docs/phase29_finer_duration_proxy_report_zh.md
docs/phase38_experiment_freeze_summary_zh.md
```

## 当前任务

从 Stage A pilot 开始，不要直接跑大规模 sweep。

需要完成：

1. 检查 Phase 25 的 action perturbation 注入路径在哪里实现。
2. 把注入参数扩展成可配置：
   - case list
   - action direction
   - epsilon / amplitude
   - perturbation window
   - output tag
   - trace on/off
3. 写最小 pilot runner：

```text
toy_quantvla/run_phase39_perturb_threshold_pilot.sh
```

4. pilot 只跑：

```text
cases:
  4:9
  6:8

directions:
  y
  z
  yaw
  continuous_6d

windows:
  full
  early
  mid
  late

epsilons:
  0.003
  0.006
  0.01
  0.02
  0.03
  0.05
  0.08
```

如果 rollout 数太多，先跑 reduced pilot：

```text
directions:
  y
  z
  yaw

windows:
  full
  early
  late

epsilons:
  0.006
  0.01
  0.03
  0.05
```

## 输出要求

生成：

```text
toy_quantvla/results/phase39_threshold_pilot_summary.json
docs/phase39_closed_loop_perturbation_budget_pilot_zh.md
```

summary 至少包含：

```text
case
direction
window
epsilon
success
steps
flip_type relative to FP16 baseline
server p50 if available
trace path if generated
```

并给出临界区间：

```text
epsilon_crit_lower
epsilon_crit_upper
```

例如：

```text
task6:init8, y, early: success at 0.006, fail at 0.01
=> epsilon_crit in (0.006, 0.01]
```

## 判断标准

pilot 目标不是跑很多 case，而是验证是否能看到：

```text
y/yaw threshold < z threshold
early threshold < late threshold
```

如果这个成立，再进入 Stage B discovery。

## 注意事项

- 所有 rollout 必须 paired 到同一 task/init/seed 的 FP16 baseline。
- 不要只报告 aggregate success，要报告 S->F regression 和 F->S repair。
- trace 优先保存失败边界附近的 epsilon，例如刚从 success 变 fail 的点。
- 如果某个方向全成功或全失败，不要硬插值；报告 threshold 为 `> max_epsilon` 或 `<= min_epsilon`。
- 先保证 pilot 可解释，再扩大。

