# Phase38: 实验冻结与论文收敛总结

## 当前状态

Phase37B held-out routing validation 已完成。至此，N1.5/N1.7 两条线已经足够支撑论文主线：VLA/world-action model 的推理加速不是一个静态转换问题，而是一个受闭环敏感性约束的 tactic search 问题。

本阶段建议进入 experiment freeze：

- 不再扩展新模型和新仿真平台。
- 不再继续大规模扫新的 compile/window tactic。
- 不再把 FP4/CUDA kernel 作为主实验线。
- 后续只做论文收敛、图表整理、算法表述和必要的 sanity check。

## 最终实验信号

Phase37B 使用 N1.7 held-out cases：

```text
tasks: 0, 1, 4, 6, 8
init indices: 24, 25, 26
total cases: 15
```

核心结果如下：

| Deployment point | Success | Avg p50 ms | Speedup vs FP16 | Interpretation |
| --- | ---: | ---: | ---: | --- |
| FP16 baseline | 13/15 | 110.40 | 1.00x | reference behavior |
| speed_only | 11/15 | 74.79 | 1.48x | speed-first, higher regression risk |
| window_5_15 | 13/15 | 90.81 | 1.22x | moderate window, still one regression |
| window_0_20 | 14/15 | 100.98 | 1.09x | behavior-first, best held-out success |
| window_10_30 | 12/15 | 97.58 | 1.13x | later window misses early sensitivity |
| routed tactic | 13/15 | 79.07 | 1.40x | high-speed trade-off, not regression-free |

该 routed tactic 使用：

```text
task0, task8 -> window_5_15
task1, task4, task6 -> speed_only
```

该结果说明 routing 不是无条件更稳。它更像一个速度-稳健性 Pareto 点：相比 `speed_only` 保留了大部分速度，但仍会出现 held-out regression。

## 论文应收束成的主线

论文不应写成 QuantVLA 复现报告，也不应写成某个单点 tactic 的性能报告。更稳的主线是：

```text
inference acceleration perturbations
-> closed-loop sensitivity
-> non-uniform dimensions / durations / layers / tasks
-> sensitivity-guided tactic search
-> held-out validation
-> speed-robustness Pareto selection
```

对应的中心 claim：

> For VLA/world-action models, inference acceleration should be treated as closed-loop-constrained tactic search rather than static model conversion.

## 五个论点的收束

1. **Quantization/implementation error is filtered by closed-loop sensitivity.**  
   单步 action drift 不直接决定 rollout outcome。扰动要经过环境动力学、policy feedback 和 task margin 的共同过滤。

2. **Outcome flips depend on task margin, not only perturbation norm.**  
   相似幅度的扰动可能只沿成功边界切向移动，也可能穿过成功/失败边界。

3. **Open-loop drift is necessary but insufficient.**  
   固定数据分布上的 drift 不能覆盖量化或 compile tactic 自己诱导出来的 closed-loop state distribution。

4. **Not all dimensions, durations, layers, and tasks are equal.**  
   敏感性高度非均匀。早期接触、抓取、对齐阶段往往比后期搬运阶段更脆弱。

5. **Acceleration should be selected on a Pareto frontier.**  
   `speed_only`、window fallback 和 routed tactic 分别代表不同部署点。最快不一定稳，最稳不一定快。

## 方法表述

建议把方法命名为：

```text
Closed-Loop Sensitivity-Guided Tactic Search
```

输入：

- candidate tactics，例如 FP16/eager、speed-only compile、window fallback、task-conditioned routing；
- probe rollouts；
- latency and memory measurements；
- paired repair/regression statistics against FP16；
- optional sensitivity proxies from dimensions, layers, durations, or first-divergence analysis。

选择规则：

- behavior-first：最大化 held-out 或 fold-wise success，优先最小化 regressions；
- speed-constrained：在 regression budget 内最大化 speedup；
- Pareto mode：报告多个 speed-risk trade-off points，而不是只给一个 winner。

## Routing 的定位

Task-conditioned tactic routing 可以作为 discussion 亮点，但语气要克制：

```text
MoE routes tokens or inputs to model experts.
Here, the router selects inference tactics rather than model parameters.
```

这可以被描述为 closed-loop-aware inference scheduling。它的价值是把推理加速从单一静态配置，扩展成按任务或轨迹 regime 选择 tactic 的机制。

但 Phase37B 也提醒我们：

```text
probe-optimal routing may overfit to specific init states.
therefore, routing itself must be validated on held-out rollouts.
```

## 最终建议

论文下一步只做四件事：

1. 把 Phase37B 加入最终实验表和 Pareto 图。
2. 在方法部分正式写出 Closed-Loop Sensitivity-Guided Tactic Search 算法。
3. 在 discussion 中加入 tactic routing / MoE-like inference scheduling 的观点。
4. 在 limitations 中明确：当前 routing 是 task-level、offline-selected，未来可以扩展到 online sensitivity-triggered routing。

这阶段的实验可以正式收尾。
