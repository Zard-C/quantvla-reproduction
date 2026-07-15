# Phase 39 Stage A2: real-backend drift replay 总结

本阶段的目标是避免只研究手工扰动方向。我们用 same-observation diagnostic 在 FP16 nominal observation 上同时查询 reference 与 tactic，得到真实后端诱导的 per-step action drift sequence：

$$
\eta_t = \pi_{\mathrm{tactic}}(s_t) - \pi_{\mathrm{fp16}}(s_t)
$$

然后在闭环 rollout 中显式 replay：

$$
a_t = a_t^{\mathrm{fp16}} + \lambda \eta_t
$$

这里 `lambda` 用来缩放真实 backend drift sequence。这个实验不等价于直接部署 tactic，而是把真实 tactic 的误差方向拿出来，单独测试这个方向是否闭环敏感。

## 实验设置

- Cases: `task4:init9`, `task6:init8`
- Baseline: FP16 两个 case 都成功
  - `task4:init9`: success, 224 policy requests
  - `task6:init8`: success, 649 policy requests
- Replay windows:
  - `full`: 整个 rollout 注入 drift sequence
  - `early`: 只在早期窗口注入
- Lambdas: `0.25, 0.5, 1.0`
- 对比 tactic:
  - `speed_only`: compile action head，不加保护
  - `combo_blocks0_3_window0_120`: 保护 blocks 0--3，并在 policy steps 0--120 使用 fallback

## 结果

### speed-only real drift sequence

| case | window | lambda=0.25 | lambda=0.5 | lambda=1.0 | 现象 |
| --- | --- | --- | --- | --- | --- |
| `task4:init9` | full | F/991 | S/221 | F/991 | 非单调，真实 drift 方向很敏感 |
| `task4:init9` | early | F/991 | S/224 | F/991 | 非单调，早期注入同样危险 |
| `task6:init8` | full | S/496 | S/833 | F/991 | 1.0x drift 触发失败 |
| `task6:init8` | early | F/991 | F/991 | F/991 | 早期真实 drift 在该 case 上极敏感 |

### protected combo real drift sequence

| case | window | lambda=0.25 | lambda=0.5 | lambda=1.0 | 现象 |
| --- | --- | --- | --- | --- | --- |
| `task4:init9` | full | S/222 | S/223 | S/224 | 到 1.0x 仍成功 |
| `task4:init9` | early | S/224 | S/224 | S/224 | 到 1.0x 仍成功 |
| `task6:init8` | full | S/229 | S/632 | S/464 | outcome 稳，但 steps 有明显重分配 |
| `task6:init8` | early | S/229 | S/840 | S/627 | outcome 稳，但 steps 有明显重分配 |

## 结论

1. **真实 backend drift 方向本身是闭环敏感的。**
   `speed_only` 的 replay 在多个 case/window 上出现 S->F，说明风险不是手工构造方向才有。

2. **同样是“真实 drift”，protected tactic 的 drift sequence 明显更安全。**
   `combo_blocks0_3_window0_120` 在所有 `lambda <= 1.0` 的 replay 中都保持 success，说明 sensitivity-guided protection 确实改变了 drift 的闭环风险。

3. **闭环边界不是简单的单调半空间。**
   `task4:init9` 的 speed-only replay 出现 `F -> S -> F`，说明扰动幅度增大不一定单调变差。这更像是在高维、非凸 trajectory basin boundary 上采样，而不是沿一个固定法向量做线性阈值测试。

4. **protected tactic 不是让轨迹完全不变，而是降低 outcome-level regression risk。**
   `task6:init8` 在 protected combo 下全部成功，但 steps 从 `229` 到 `840` 都出现过，说明轨迹仍然被重分配。我们应该说它降低了闭环失败风险，而不是说它恢复了 FP16 trajectory。

5. **这为 CLSG-TS 提供了直接证据。**
   后端 tactic 不能只按 latency 或 open-loop drift 选择；更合理的做法是估计真实 drift direction/sequence 的闭环风险，然后在 speed--risk frontier 上选择 tactic。

## 论文口径

建议写成：

> To connect controlled perturbations with actual backend residuals, we recorded same-observation drift sequences between FP16 and accelerated tactics on nominal FP16 rollouts, then replayed scaled versions of these residuals in closed loop. Speed-only residuals caused failures under several windows and scaling factors, whereas the sensitivity-guided protected tactic remained successful up to the unscaled residual sequence. This indicates that closed-loop risk depends not only on residual magnitude, but also on the direction, timing, and tactic-induced structure of backend errors.

需要避免过度表述：

- 不能说我们找到了真实成功边界法向量。
- 不能说 protected tactic 完全保持 FP16 trajectory。
- 不能说这个阈值是普适 numerical tolerance。
- 更准确的说法是：我们用真实 backend drift sequence 做了 directional vulnerability probe。
