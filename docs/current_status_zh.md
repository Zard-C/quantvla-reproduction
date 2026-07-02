# 当前进度与结论

这份说明是给 GitHub 网页阅读用的入口页，方便快速定位论文、最新实验和下一步计划。

## 论文

- PDF: [`paper/main.pdf`](../paper/main.pdf)
- LaTeX source: [`paper/main.tex`](../paper/main.tex)
- 当前标题: **Inference Acceleration as Closed-Loop Perturbation: Sensitivity-Guided Speedups for VLA Policies**

论文主线已经从单纯的量化复现，扩展为 VLA/world-action policy 的推理加速闭环分析。这里的推理加速包括 fake quantization、graph compile、eager island、mixed precision 和未来的 packed kernel / CUDA kernel 路线。

## 核心判断

1. 推理加速不只是系统优化问题。对闭环 VLA policy 来说，加速后端可能产生很小的 action-level perturbation，但这些扰动会被环境动力学、policy feedback 和任务成功边界共同过滤。
2. open-loop drift 小不等于 closed-loop rollout 稳。固定数据分布上的 action drift 只能说明局部误差，不能保证量化或 compile 后策略自己走出来的状态分布仍然安全。
3. aggregate success rate 不够，需要 paired repair/regression。相同 task/init/seed 下，候选后端可能同时修复一些 baseline 失败样本，又打坏一些 baseline 成功样本。
4. not all dims / durations / layers are equal。闭环敏感性在 action channel、rollout 阶段和模型层级上都是各向异性的。
5. coarse layer proxy 不够。`blocks8-15` 在 15-case 小样本上看起来有效，但扩大到 33-case 后没有泛化，说明 proxy-guided 策略需要 held-out matched rollouts 验证。
6. finer duration proxy 证明了 duration sensitivity，但不是最终固定策略。`0-120 eager + rest compiled` 在 Phase29 probe set 上达到 FP16 baseline 的 `19/33` 成功，同时 p50 latency `69.66 ms`，基本保留 speed-only compile 的 `70.20 ms` 速度；但 Phase30 held-out sanity 显示它没有泛化。
7. held-out 选择本身很关键。Phase30 held-out 会让 `speed-only compile` 看起来最优，但 Phase32 held-out 中 `speed-only` 从 FP16 的 `25/30` 掉到 `20/30`，出现 5 个 paired regression。单一 held-out slice 可能误选 tactic。
8. 当前最稳的候选不是最快的候选。`blocks0-3 + window 0-120` 在 Phase32 上逐 case 复现 FP16 outcome，`0` repair / `0` regression，同时 p50 仍有 `1.75x` 加速；它是目前最强的 behavior-preserving candidate，但还需要回测 Phase30 或第三组 held-out 才能写成最终推荐 tactic。
9. Phase34 multi-fold selection 把 Phase30 和 Phase32 作为两个 validation folds 后，结论变成一个 trade-off：行为优先选 `blocks0-3 + window 0-120`，但它的 worst-fold speedup 只有 `1.07x`；若要求 worst-fold speedup >= `1.5x`，当前更均衡的是 `window 0-120`。

## Phase28A: 15-case proxy-guided probe

报告: [`docs/phase28_proxy_guided_mixed_precision_report_zh.md`](phase28_proxy_guided_mixed_precision_report_zh.md)

| run | success | server p50 | 相对 FP16 |
| --- | ---: | ---: | ---: |
| FP16 baseline | 7/15 | 84.76 ms | 1.00x |
| speed-only compile | 5/15 | 50.35 ms | 1.68x |
| block0 eager | 6/15 | 50.96 ms | 1.66x |
| blocks8-15 eager | 7/15 | 67.36 ms | 1.26x |
| random block1 eager | 5/15 | 51.54 ms | 1.64x |

Phase28A 给了一个正向信号：`blocks8-15 eager` 相比 speed-only 修复 `4:6, 6:0, 8:10`，只新增 `4:9` 一个 regression。但这只是小样本 diagnostic，不能作为最终工程结论。

## Phase28B: 33-case held-out expansion

报告: [`docs/phase28b_proxy_guided_33case_report_zh.md`](phase28b_proxy_guided_33case_report_zh.md)

| run | success | server p50 | 相对 FP16 |
| --- | ---: | ---: | ---: |
| FP16 baseline | 19/33 | 156.22 ms | 1.00x |
| speed-only compile | 16/33 | 70.20 ms | 2.23x |
| blocks8-15 eager | 15/33 | 82.64 ms | 1.89x |
| random block1 eager | 16/33 | 68.37 ms | 2.28x |

Phase28B 的结论更重要：`blocks8-15 eager` 没有泛化。它相对 speed-only 修复 5 个 case，但新增 6 个 regression，net `-1`；相对 FP16 baseline 是 1 个 repair / 5 个 regression，net `-4`。

这不是否定 proxy-guided acceleration，而是说明静态 coarse layer mask 不够。更好的策略应当基于 layer × duration × action channel 的闭环敏感性，而不是只保护某一段层。

## Phase28C: 33-case candidate search

报告: [`docs/phase28c_proxy_guided_33case_report_zh.md`](phase28c_proxy_guided_33case_report_zh.md)

脚本:

- [`toy_quantvla/run_phase28c_proxy_guided_33case.sh`](../toy_quantvla/run_phase28c_proxy_guided_33case.sh)
- [`toy_quantvla/phase28c_proxy_guided_summary.py`](../toy_quantvla/phase28c_proxy_guided_summary.py)

Phase28C 固定 Phase28B 的 33-case matched set，继续搜索更好的 speed/behavior trade-off：

| candidate | idea |
| --- | --- |
| A: `block0 eager` | same-observation spike proxy，保留 `transformer_blocks.0` eager |
| B: `block0 + blocks8-15 eager` | 合并 spike proxy 和早期 repair/regression proxy |
| C: `blocks0-3 eager` | 更宽的 early-block protection |
| D: `step 120-320 eager window` | 第一版 duration-aware fallback，窗口内走 eager，其余走 compiled；早期因 eval step key 漏传，已经在 Phase28D 中重跑 |

当前最好工程 tradeoff 是 `blocks0-3 eager`：`18/33`，server p50 约 `67.6 ms`，接近 speed-only 的速度，同时比 speed-only 多 2 个成功 case。

## Phase28D: duration-window fallback

报告: [`docs/phase28d_duration_window_analysis_zh.md`](phase28d_duration_window_analysis_zh.md)

数据:

- [`toy_quantvla/results/phase28D_duration_window_33case_v1_summary.json`](../toy_quantvla/results/phase28D_duration_window_33case_v1_summary.json)
- [`toy_quantvla/results/phase28D_duration_window_33case_v1_analysis.json`](../toy_quantvla/results/phase28D_duration_window_33case_v1_analysis.json)

| run | success | server p50 | 相对 FP16 |
| --- | ---: | ---: | ---: |
| speed-only compile | 16/33 | 70.20 ms | 2.23x |
| window 80-220 | 14/33 | 75.11 ms | 2.08x |
| window 120-320 | 16/33 | 78.09 ms | 2.00x |
| window 180-420 | 16/33 | 78.82 ms | 1.98x |
| window 0-250 | 18/33 | 88.75 ms | 1.76x |

`window 0-250` 是 D 方案中最好的窗口：相对 speed-only 修复 `4:6, 6:0, 6:6, 8:7`，打坏 `6:7, 6:9`，净 `+2`。它很可能覆盖了接近、对齐、接触、夹爪闭合和初始搬起这些 grasp-critical steps；后续步骤更像规则搬运/放置，对误差没有同样敏感。

不过 `0-250` 的代价也明显：server eager fraction 约 `0.38`，p50 升到 `88.75 ms`。所以 D 的主要价值是证明 duration sensitivity 存在，而不是替代 Phase28C 的最佳工程方案。

## Phase29: finer duration proxy

报告: [`docs/phase29_finer_duration_proxy_report_zh.md`](phase29_finer_duration_proxy_report_zh.md)

数据:

- [`toy_quantvla/results/phase29_finer_duration_proxy_33case_v1_summary.json`](../toy_quantvla/results/phase29_finer_duration_proxy_33case_v1_summary.json)

脚本:

- [`toy_quantvla/run_phase29_finer_duration_proxy.sh`](../toy_quantvla/run_phase29_finer_duration_proxy.sh)
- [`toy_quantvla/phase29_finer_duration_proxy_summary.py`](../toy_quantvla/phase29_finer_duration_proxy_summary.py)

Phase29 把 `0-250` 拆成更细的 duration windows，结果表明“保护越多越安全”并不成立，窗口位置比窗口长度更关键。

| run | success | server p50 | eager frac | 相对 FP16 |
| --- | ---: | ---: | ---: | ---: |
| FP16 baseline | 19/33 | 156.22 ms | 1.000 | 1.00x |
| speed-only compile | 16/33 | 70.20 ms | 0.000 | 2.23x |
| window 0-120 | 19/33 | 69.66 ms | 0.190 | 2.24x |
| window 0-180 | 18/33 | 77.55 ms | 0.281 | 2.01x |
| window 0-220 | 16/33 | 90.49 ms | 0.328 | 1.73x |
| window 80-240 | 14/33 | 75.98 ms | 0.214 | 2.06x |
| window 120-280 | 16/33 | 73.48 ms | 0.216 | 2.13x |
| window 160-240 | 15/33 | 69.29 ms | 0.111 | 2.25x |
| window 240-320 | 17/33 | 66.71 ms | 0.084 | 2.34x |
| window 0-250 | 18/33 | 88.75 ms | 0.376 | 1.76x |

`0-120` 是当前最值得写进论文的 proxy-guided acceleration 结果：它相对 speed-only 修复 7 个 case，新增 4 个 regression，净 `+3`；相对 FP16 baseline 则是 2 个 repair / 2 个 regression，aggregate success 持平。它说明闭环敏感阶段更接近“早期 basin selection / approach / alignment”，而不是完整抓取和搬运全过程。

不过 `0-120` 不是“找到了唯一正确窗口”，而是一个更强的论据：VLA 加速的解空间不是单点最优，需要通过 matched rollouts、repair/regression 和 trace divergence 去定位哪些阶段、维度、层边界值得保护。

## Phase30: held-out sanity

报告: [`docs/phase30_heldout_sanity_report_zh.md`](phase30_heldout_sanity_report_zh.md)

数据:

- [`toy_quantvla/results/phase30_heldout_sanity_30case_v1_summary.json`](../toy_quantvla/results/phase30_heldout_sanity_30case_v1_summary.json)

Phase30 使用新的 init set `15/16/17` 作为 held-out sanity set，不参与 Phase29 的 `0-120` 选择。这个 slice 上 `speed-only compile` 最强，但 Phase32 后我们不再把它称为稳定 incumbent，而是把它视为“某个 held-out slice 上的 winner”。

| run | success | server p50 | server p90 | 相对 FP16 |
| --- | ---: | ---: | ---: | ---: |
| FP16 baseline | 23/30 | 88.81 ms | 94.18 ms | 1.00x |
| speed-only compile | 25/30 | 50.72 ms | 61.07 ms | 1.75x |
| blocks0-3 eager | 21/30 | 61.96 ms | 66.39 ms | 1.43x |
| window 0-120 | 22/30 | 51.82 ms | 87.97 ms | 1.71x |

paired repair/regression 相对 FP16：

| run | repair | regress | net |
| --- | ---: | ---: | ---: |
| speed-only compile | 4 | 2 | +2 |
| blocks0-3 eager | 3 | 5 | -2 |
| window 0-120 | 1 | 2 | -1 |

这个结果不削弱论文主线，反而让主线更清晰：`0-120` 应该写成 probe diagnostic，而不是 universal tactic。真正的贡献是把 VLA 加速表述成 closed-loop tactic search。

## Phase31: retrospective tactic search

报告: [`docs/phase31_tactic_search_retrospective_zh.md`](phase31_tactic_search_retrospective_zh.md)

数据:

- [`toy_quantvla/results/phase31_tactic_search_retrospective.json`](../toy_quantvla/results/phase31_tactic_search_retrospective.json)

Phase31 把 Phase28/29/30 的结果统一成一个启发式 scorer：

```text
score = 100 * success_rate
      + 4 * speedup
      + repair_vs_baseline
      - 2.5 * regress_vs_baseline
      - 10 * baseline_success_gap
      - 4 * worst_task_gap
```

在 Phase29 probe set 上，scorer 选择 `window_0_120`；Phase30 held-out set 会选择 `speed_only`；Phase32 held-out set 则选择 `combo_blocks0_3_window_0_120`。这给论文一个更稳的结论：固定 duration window 或固定 speed-only 都不是答案，held-out closed-loop validation 的划分本身会影响 tactic ranking。

## Phase32: held-out tactic validation v2

计划: [`docs/phase32_tactic_validation_plan_zh.md`](phase32_tactic_validation_plan_zh.md)
报告: [`docs/phase32_tactic_validation_report_zh.md`](phase32_tactic_validation_report_zh.md)

Phase32 使用新的 held-out init set `18/19/20`，继续验证 Phase31 的结论。候选包括 FP16 baseline、`speed_only`、`window_0_120`，以及 `blocks0-3 + window_0_120` 组合策略。

核心问题：

结果：

| run | success | p50 | p90 | paired vs FP16 |
| --- | ---: | ---: | ---: | --- |
| FP16 baseline | 25/30 | 154.72 ms | 163.94 ms | - |
| speed-only compile | 20/30 | 68.01 ms | 144.72 ms | 0 repair / 5 regress |
| window 0-120 | 25/30 | 78.26 ms | 159.40 ms | 1 repair / 1 regress |
| blocks0-3 + window 0-120 | 25/30 | 88.47 ms | 159.19 ms | 0 repair / 0 regress |

Phase32 是目前最强的“held-out selection matters”证据：Phase30 让 `speed_only` 看起来是 winner，但 Phase32 暴露出它对另一组 init 的 regression；组合候选虽然慢一些，却最接近 behavior-preserving acceleration。

## Phase33: combo backtest on Phase30 slice

计划: [`docs/phase33_combo_backtest_phase30_plan_zh.md`](phase33_combo_backtest_phase30_plan_zh.md)
报告: [`docs/phase33_combo_backtest_phase30_report_zh.md`](phase33_combo_backtest_phase30_report_zh.md)

Phase33 将 Phase32 上最稳的 `blocks0-3 + window 0-120` 组合候选回测到 Phase30 的 init `15/16/17` slice。目标是检查 combo 是否跨 held-out slice 保持 FP16 behavior，而不是只在 Phase32 上有效。

结果显示 combo 没有跨 slice 稳住：

| run | success | p50 | paired vs FP16 |
| --- | ---: | ---: | --- |
| speed-only compile | 25/30 | 50.72 ms | 4 repair / 2 regress |
| window 0-120 | 22/30 | 51.82 ms | 1 repair / 2 regress |
| blocks0-3 + window 0-120 | 22/30 | 83.31 ms | 0 repair / 1 regress |

combo 的 regression 比 `speed_only` 少，但成功率和速度都不占优。这说明 combo 也是 slice-dependent candidate，不能作为固定最终 tactic。

## Phase34: multi-fold robust tactic selection

报告: [`docs/phase34_multifold_tactic_selection_zh.md`](phase34_multifold_tactic_selection_zh.md)

数据:

- [`toy_quantvla/results/phase34_multifold_tactic_selection.json`](../toy_quantvla/results/phase34_multifold_tactic_selection.json)

Phase34 把 Phase30 slice 和 Phase32 slice 作为两个 validation folds，统一比较 `speed_only`、`window_0_120`、`combo_blocks0_3_window_0_120`。

| tactic | pooled success | worst success | mean speedup | worst speedup | total regress |
| --- | ---: | ---: | ---: | ---: | ---: |
| speed_only | 45/60 | 0.667 | 2.01x | 1.75x | 7 |
| window_0_120 | 47/60 | 0.733 | 1.84x | 1.71x | 3 |
| combo_blocks0_3_window_0_120 | 47/60 | 0.733 | 1.41x | 1.07x | 1 |

判读：

- behavior-first 选择 combo，因为它 total regression 最少。
- speed-constrained 选择 `window_0_120`，因为它在两个 folds 都保持至少 `1.7x` 加速，同时 regression 明显少于 `speed_only`。
- `speed_only` 是最快但不稳，不能作为闭环稳健加速策略。

## Phase36-37B: GR00T N1.7 transfer and routed tactic validation

计划与报告：

- [`docs/phase36_n17_transfer_plan_zh.md`](phase36_n17_transfer_plan_zh.md)
- [`docs/phase36b_n17_tactic_probe_report_zh.md`](phase36b_n17_tactic_probe_report_zh.md)
- [`docs/phase37a_n17_finer_window_report_zh.md`](phase37a_n17_finer_window_report_zh.md)
- [`docs/phase37b_n17_heldout_routing_report_zh.md`](phase37b_n17_heldout_routing_report_zh.md)

N1.7 线的目的不是证明 `0-120` 可以直接迁移，而是验证同一套 Closed-Loop Sensitivity-Guided Tactic Search 流程是否仍然有价值。Phase37B 在 N1.7 checkpoint 上使用 15 个 held-out cases：

```text
tasks: 0, 1, 4, 6, 8
init indices: 24, 25, 26
```

核心结果：

| tactic | success | p50 | speedup | paired vs FP16 |
| --- | ---: | ---: | ---: | --- |
| FP16 baseline | 13/15 | 110.40 ms | 1.00x | - |
| speed_only | 11/15 | 74.79 ms | 1.48x | 1 repair / 3 regress |
| window_5_15 | 13/15 | 90.81 ms | 1.22x | 1 repair / 1 regress |
| window_0_20 | 14/15 | 100.98 ms | 1.09x | 2 repair / 1 regress |
| window_10_30 | 12/15 | 97.58 ms | 1.13x | 1 repair / 2 regress |
| routed tactic | 13/15 | 79.07 ms | 1.40x | 2 repair / 2 regress |

`routed tactic` 使用：

```text
task0, task8 -> window_5_15
task1, task4, task6 -> speed_only
```

判读：

- N1.7 仍然呈现 speed-risk Pareto：最快的 `speed_only` regression 最多，最稳的 `window_0_20` 速度收益最小。
- task-conditioned routing 有 MoE-like 的味道，但路由对象是 inference tactic，不是模型 expert。
- routing 是高速度折中点，不是无回退风险的 safe default；它本身也需要 held-out validation。

## Phase38: experiment freeze and paper convergence

总结：[`docs/phase38_experiment_freeze_summary_zh.md`](phase38_experiment_freeze_summary_zh.md)

Phase38 后实验主线冻结。当前论文已经从“QuantVLA 复现”收束为：

```text
inference acceleration perturbations
-> closed-loop sensitivity
-> non-uniform dimensions / durations / layers / tasks
-> sensitivity-guided tactic search
-> held-out validation
-> speed-robustness Pareto selection
```

最终中心 claim：

> For VLA/world-action models, inference acceleration should be treated as closed-loop-constrained tactic search rather than static model conversion.

当前 `paper/main.pdf` 已包含：

- 五个理论 claim；
- quantization redistribution 证据；
- action channel / duration / layer anisotropy；
- N1.5 multi-fold tactic validation；
- N1.7 held-out routing check；
- speed-risk Pareto 图；
- `Closed-Loop Sensitivity-Guided Tactic Search (CLSG-TS)` 算法盒子；
- limitations 中对 fake quant、prototype compile、small N1.7 check 和 offline task-level routing 的边界说明。

## 阅读顺序

建议按这个顺序看：

1. [`paper/main.pdf`](../paper/main.pdf)
2. [`docs/phase25_anisotropic_sensitivity_report_zh.md`](phase25_anisotropic_sensitivity_report_zh.md)
3. [`docs/phase28_proxy_guided_mixed_precision_report_zh.md`](phase28_proxy_guided_mixed_precision_report_zh.md)
4. [`docs/phase28b_proxy_guided_33case_report_zh.md`](phase28b_proxy_guided_33case_report_zh.md)
5. [`docs/phase28c_proxy_guided_33case_report_zh.md`](phase28c_proxy_guided_33case_report_zh.md)
6. [`docs/phase28d_duration_window_analysis_zh.md`](phase28d_duration_window_analysis_zh.md)
7. [`docs/phase29_finer_duration_proxy_report_zh.md`](phase29_finer_duration_proxy_report_zh.md)
8. [`docs/phase30_heldout_sanity_report_zh.md`](phase30_heldout_sanity_report_zh.md)
9. [`docs/phase31_tactic_search_retrospective_zh.md`](phase31_tactic_search_retrospective_zh.md)
10. [`docs/phase32_tactic_validation_plan_zh.md`](phase32_tactic_validation_plan_zh.md)
11. [`docs/phase32_tactic_validation_report_zh.md`](phase32_tactic_validation_report_zh.md)
12. [`docs/phase33_combo_backtest_phase30_plan_zh.md`](phase33_combo_backtest_phase30_plan_zh.md)
13. [`docs/phase33_combo_backtest_phase30_report_zh.md`](phase33_combo_backtest_phase30_report_zh.md)
14. [`docs/phase34_multifold_tactic_selection_zh.md`](phase34_multifold_tactic_selection_zh.md)
15. [`docs/phase36_n17_transfer_plan_zh.md`](phase36_n17_transfer_plan_zh.md)
16. [`docs/phase36b_n17_tactic_probe_report_zh.md`](phase36b_n17_tactic_probe_report_zh.md)
17. [`docs/phase37a_n17_finer_window_report_zh.md`](phase37a_n17_finer_window_report_zh.md)
18. [`docs/phase37b_n17_heldout_routing_report_zh.md`](phase37b_n17_heldout_routing_report_zh.md)
19. [`docs/phase38_experiment_freeze_summary_zh.md`](phase38_experiment_freeze_summary_zh.md)
