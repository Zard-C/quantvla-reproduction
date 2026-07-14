# Phase 39: Closed-loop perturbation threshold / error budget 实验方案

## 目标

本阶段把已有的 closed-loop sensitivity 现象升级成可量化的 deployment error budget。

核心问题：

```text
一个后端实现引入的 action perturbation 多大时，会改变闭环 rollout 结果？
```

最终希望得到：

```text
epsilon_star(direction, phase, task_slice)
```

也就是不同 action 方向、rollout 阶段、任务切片上的临界扰动阈值。这个阈值图用于指导后端部署，而不是只解释某个 PyTorch tactic。

## 论文 claim

本阶段服务于一个更强的主张：

```text
Backend optimization should satisfy closed-loop perturbation budgets,
not only local numerical tolerances.
```

对应三个需要证明的链路：

1. 临界扰动阈值存在，而且在 action dimension、duration、task slice 上明显各向异性。
2. 真实 backend drift 是否危险，不取决于平均 action drift，而取决于它是否落在低阈值区域。
3. 用 threshold map 选择 tactic，可以比 speed-only、norm-only、random protection 获得更好的 speed-risk trade-off。

## 现有基础

已有 Phase 25 证明了固定幅度 `0.03` 下的各向异性：

- `z` 方向在两个重点 case 上仍成功。
- `y/roll/pitch/yaw` 在同样 L2 幅度下更容易失败。
- 同样 `y` 扰动打在 early/mid/late 阶段，闭环结果不同。

已有 Phase 29/32/33/37 证明了真实 tactic 的 repair/regression：

- `speed_only`
- `window_0_120`
- `combo_blocks0_3_window_0_120`
- N1.7 held-out windows / routed tactic

Phase 39 不重复这些结论，而是估计“多大扰动会出事”。

## 实验设计总览

### Stage A: 2-case pilot，验证 threshold sweep

目的：先验证脚本、指标和阈值曲线，不追求统计结论。

case：

```text
task4:init9
task6:init8
```

这两个 case 已在 Phase 25 中反复使用，baseline 成功，且对不同扰动方向有明显差异。

扰动方向：

```text
y
z
yaw
continuous_6d
```

窗口：

```text
full
early
mid
late
```

建议窗口：

```text
task4: early=0:75, mid=75:150, late=150:225
task6: early=0:200, mid=200:450, late=450:700
```

epsilon grid：

```text
0.003, 0.006, 0.01, 0.02, 0.03, 0.05, 0.08
```

输出：

```text
success/failure
steps
first_action_divergence_step
first_eef_divergence_step
terminal EEF/object/gripper proxy
```

pilot 成功条件：

```text
至少看到 y/yaw 的 flip threshold 低于 z。
至少看到 early threshold 低于 late。
```

### Stage B: 33-case discovery threshold map

目的：建立 discovery 阈值图，不直接用于最终 claim。

case list 沿用 Phase 29：

```text
tasks 4, 6, 8
init 0..10
共 33 cases
```

为了控制成本，不做完整笛卡尔积。建议先跑：

扰动方向：

```text
y
z
yaw
gripper
real_speed_only_drift_direction
```

窗口：

```text
full
0:120
120:250
250:end
```

epsilon grid：

```text
0.006, 0.01, 0.02, 0.03, 0.05
```

如果预算紧，先用 adaptive bisection：

1. 跑 `0.01` 和 `0.03`。
2. 如果两者 outcome 一样，跳过中间点。
3. 如果发生 flip，再补 `0.006/0.02/0.05`。

输出每个 `(case, direction, window)` 的：

```text
epsilon_crit_lower
epsilon_crit_upper
flip_type: S->F / F->S / none
```

其中 `epsilon_crit_lower/upper` 表示临界阈值所在区间，而不是假装精确到一个点。

### Stage C: held-out prediction

目的：证明 threshold map 不只是解释 discovery，而能预测真实 tactic 风险。

held-out 数据：

```text
N1.5 held-out folds:
- phase32 30case
- phase33 backtest 30case

可选：
- N1.7 phase37b 15case
```

对真实 backend tactic 计算 open-loop 或 request-trace drift：

```text
eta_t = action_tactic(s_t) - action_fp16(s_t)
```

用 discovery threshold map 构造风险分数：

```text
risk(case, tactic)
  = max over t, channel of |eta_t[channel]| / epsilon_star(channel, phase(t), task_slice)
```

也可以做更稳的聚合：

```text
risk_p95 = p95_t normalized_drift_t
risk_mass = sum_t max(0, normalized_drift_t - 1)
```

预测任务：

```text
label = whether tactic regresses FP16 success on the same case
```

报告：

```text
AUROC
precision@top_k
top-risk regression rate
correlation between risk score and first divergence step
```

成功条件：

```text
threshold risk score 比 raw action L2 / mean drift 更能预测 regression。
```

### Stage D: deployment selection 对照

目的：证明 threshold map 可以指导部署选择。

比较 selector：

```text
speed-only selector
mean-drift selector
random protection selector
threshold-guided selector
behavior-first oracle upper bound
```

输出：

```text
success
paired repairs
paired regressions
server p50 latency
speedup
```

目标表格：

| selector | success | regressions | repairs | speedup |
| --- | ---: | ---: | ---: | ---: |
| speed-only | high speed / high risk | | | |
| norm-only | medium | | | |
| random protection | unstable | | | |
| threshold-guided | lower regressions at useful speed | | | |

这张表是论文最关键的 deployment 证据。

## 最小执行顺序

建议 5090 开机后按下面顺序跑：

1. **Stage A pilot**
   - 只跑 2 cases。
   - 确认 perturbation sweep、trace、summary 都能出结果。
   - 预计几十个 rollout。

2. **Stage A summary**
   - 画 `epsilon vs success/failure`。
   - 输出 `epsilon_crit` 表。

3. **Stage B discovery**
   - 先跑 task4/6/8 的 33-case，小 grid 或 adaptive grid。

4. **Stage C prediction**
   - 不急着重跑 held-out tactic。
   - 先用已有 request_trace / action drift 计算 risk score。
   - 如果缺 trace，再补最少量 held-out trace。

5. **Stage D selector**
   - 只有当 Stage C 显示 threshold risk score 有预测力时再做。

## 建议新增脚本

建议新阶段脚本命名：

```text
toy_quantvla/run_phase39_perturb_threshold_pilot.sh
toy_quantvla/run_phase39_perturb_threshold_discovery.sh
toy_quantvla/phase39_perturb_threshold_summary.py
toy_quantvla/phase39_backend_drift_risk_score.py
toy_quantvla/phase39_selector_comparison.py
```

如果想少改代码，可以复用 Phase 25 的 action perturbation 注入路径，只把 `amplitude`、`dimension`、`window`、`case list` 参数化。

## 结果文件建议

```text
toy_quantvla/results/phase39_threshold_pilot_summary.json
toy_quantvla/results/phase39_threshold_discovery_summary.json
toy_quantvla/results/phase39_backend_risk_prediction.json
toy_quantvla/results/phase39_selector_comparison.json

docs/phase39_closed_loop_perturbation_budget_pilot_zh.md
docs/phase39_closed_loop_perturbation_budget_prediction_zh.md
```

## 论文改写方向

如果 Stage C/D 成功，论文主线建议从：

```text
Closed-Loop Sensitivity-Guided Tactic Search
```

升级为：

```text
Closed-Loop Perturbation Budgets for VLA Inference Acceleration
```

tactic search 降级为应用案例：

```text
Given a perturbation budget map, choose a backend tactic that stays under the budget while maximizing speed.
```

这样论文贡献更不依赖 PyTorch 后端，也更接近部署问题本身。

## 风险与止损

如果 Stage A 不能得到清晰阈值：

```text
说明阈值受 case margin 影响太强，需要先按 easy/fragile/failure 分层。
```

如果 Stage B 有阈值，但 Stage C 不能预测 backend regression：

```text
说明人工 action perturbation 方向和真实 backend drift 不对齐。
需要加入 real_speed_only_drift_direction，而不是只测坐标轴方向。
```

如果 Stage C 能预测，但 Stage D selector 没提升：

```text
说明现有 tactic 候选空间太窄。
需要新增更多 protection/window/backend 候选，而不是否定 threshold map。
```

