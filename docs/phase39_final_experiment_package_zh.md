# Phase39 final experiment package: closed-loop perturbation budget

Phase39 是论文里连接理论和 tactic search 的关键阶段。它回答的问题是：

```text
多小的 action-level implementation perturbation 会改变闭环 rollout outcome？
真实 backend drift 是否也会落在这些敏感方向上？
protected tactic 是否真的降低真实 drift 的闭环风险？
```

## 证据链

### A1: ultra-low controlled perturbation

报告：

- [`phase39_closed_loop_perturbation_budget_pilot_zh.md`](phase39_closed_loop_perturbation_budget_pilot_zh.md)

核心设置：

```text
case: task4:init9
direction: y action channel
windows: full, early [0,75)
epsilon: 1e-6 ... 5e-4
```

核心结果：

| case | direction | window | first flip interval | interpretation |
| --- | --- | --- | --- | --- |
| `task4:init9` | `y` | full | `(1e-5, 3e-5]` | ultra-low perturbation can cross a local basin boundary |
| `task4:init9` | `y` | early `[0,75)` | `(1e-5, 3e-5]` | early contact/approach region is equally sensitive |

结论：

```text
local numerical tolerance != closed-loop safety budget
```

即使 `3e-5` 量级的局部 action perturbation，只要方向和阶段落在低 margin / 高 sensitivity 区域，也能把 FP16 success 变成 horizon failure。

### A2.1: same-observation real backend drift

报告：

- [`phase39_stageA2_sameobs_realdrift_speedonly_seed20260613_v1_sameobs_online_drift.md`](phase39_stageA2_sameobs_realdrift_speedonly_seed20260613_v1_sameobs_online_drift.md)
- [`phase39_stageA2_sameobs_realdrift_combo_blocks0_3_window0_120_seed20260613_v1_sameobs_online_drift.md`](phase39_stageA2_sameobs_realdrift_combo_blocks0_3_window0_120_seed20260613_v1_sameobs_online_drift.md)

方法：

沿 FP16 nominal trajectory 同时查询 reference 与 tactic：

$$
\eta_t = \pi_{\mathrm{tactic}}(s_t) - \pi_{\mathrm{fp16}}(s_t)
$$

这样得到真实 backend drift 的方向、时间结构和 per-step sequence。

### A2.2: mean drift direction threshold

报告：

- [`phase39_stageA2_sameobs_realdrift_speedonly_seed20260613_v1_threshold_zh.md`](phase39_stageA2_sameobs_realdrift_speedonly_seed20260613_v1_threshold_zh.md)
- [`phase39_stageA2_sign_neg_speedonly_mean_threshold_zh.md`](phase39_stageA2_sign_neg_speedonly_mean_threshold_zh.md)

结论：

真实 `speed_only` mean drift direction 在 `1e-5` 到 `3e-5` 附近也能触发闭环 flip。正负方向表现不对称，并且存在非单调 outcome，说明局部边界不是简单的线性半空间。

### A2.3: real backend drift sequence replay

综合报告：

- [`phase39_stageA2_real_backend_replay_summary_zh.md`](phase39_stageA2_real_backend_replay_summary_zh.md)

方法：

在闭环中 replay 真实 drift sequence：

$$
a_t = a_t^{\mathrm{fp16}} + \lambda \eta_t
$$

其中 `eta_t` 来自 same-observation diagnostic。

核心结果：

| residual source | case/window | lambda outcomes | conclusion |
| --- | --- | --- | --- |
| `speed_only` | `task4` full/early | `0.25:F, 0.5:S, 1.0:F` | true backend drift can trigger non-monotonic basin flips |
| `speed_only` | `task6` full | `0.25:S, 0.5:S, 1.0:F` | full-horizon true drift fails at 1.0x |
| `speed_only` | `task6` early | `0.25:F, 0.5:F, 1.0:F` | early true drift is highly sensitive |
| `blocks0-3 + window0-120` | both cases, full/early | all success through `1.0x` | protected tactic lowers outcome-level regression risk |

结论：

```text
真实 backend drift 方向本身就是闭环敏感对象；
sensitivity-guided protection 改变的不只是误差大小，而是误差的闭环风险结构。
```

## 关键 artifacts

推荐入库保留：

```text
docs/phase39_*.md
toy_quantvla/results/phase39_threshold_pilot_summary.json
toy_quantvla/results/phase39_stageA2_*_sameobs_online_drift.json
toy_quantvla/results/phase39_stageA2_*_sameobs_real_drift_directions.json
toy_quantvla/results/phase39_stageA2_*_sameobs_real_drift_sequences.json
```

不建议入库：

```text
*_client.log
*_server.log
*_eval.log
*_client_latency.json
*_server_latency.json
*_request_trace.jsonl
*_manifest.jsonl
```

这些 raw rollout artifacts 已在 `.gitignore` 中忽略。

## 论文使用口径

建议写成：

> Controlled perturbation sweeps show that a fragile FP16-success rollout can flip under an action perturbation between \(10^{-5}\) and \(3\times 10^{-5}\). To connect this with actual backend behavior, we record same-observation residual sequences between FP16 and accelerated tactics and replay scaled versions in closed loop. Speed-only residuals cause multiple S->F flips, whereas the sensitivity-guided protected tactic remains successful up to the unscaled residual sequence.

避免过度 claim：

- 不说我们找到了真实成功边界法向量。
- 不说 threshold 是全局 numerical tolerance。
- 不说 protected tactic 保持 FP16 trajectory 完全一致。
- 只说这是 directional vulnerability probe，说明真实后端残差也会被 closed-loop sensitivity 过滤。

## 和 CLSG-TS 的关系

Phase39 的定位不是一个最终 tactic，而是为 CLSG-TS 提供三个先验：

1. **方向先验**：并非所有 action/drift directions 同样危险。
2. **阶段先验**：早期窗口可能更靠近 basin selection / contact-critical boundary。
3. **backend 先验**：真实 tactic residual sequence 本身需要闭环验证。

因此 Phase39 后合理的下一步不是继续手工找单点，而是转入 active tactic search：

```text
probe tactics -> estimate paired risk -> select next tactic/window/routing candidate
```
