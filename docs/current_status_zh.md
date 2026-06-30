# 当前进度与结论

这份说明是给 GitHub 网页阅读用的入口页，方便快速定位论文、最新实验和下一步计划。

## 论文

- PDF: [`paper/main.pdf`](../paper/main.pdf)
- LaTeX source: [`paper/main.tex`](../paper/main.tex)
- 当前标题: **When Inference Acceleration Changes Behavior: Closed-Loop Analysis for VLA Policies**

论文主线已经从单纯的量化复现，扩展为 VLA/world-action policy 的推理加速闭环分析。这里的推理加速包括 fake quantization、graph compile、eager island、mixed precision 和未来的 packed kernel / CUDA kernel 路线。

## 核心判断

1. 推理加速不只是系统优化问题。对闭环 VLA policy 来说，加速后端可能产生很小的 action-level perturbation，但这些扰动会被环境动力学、policy feedback 和任务成功边界共同过滤。
2. open-loop drift 小不等于 closed-loop rollout 稳。固定数据分布上的 action drift 只能说明局部误差，不能保证量化或 compile 后策略自己走出来的状态分布仍然安全。
3. aggregate success rate 不够，需要 paired repair/regression。相同 task/init/seed 下，候选后端可能同时修复一些 baseline 失败样本，又打坏一些 baseline 成功样本。
4. not all dims / durations / layers are equal。闭环敏感性在 action channel、rollout 阶段和模型层级上都是各向异性的。
5. coarse layer proxy 不够。`blocks8-15` 在 15-case 小样本上看起来有效，但扩大到 33-case 后没有泛化，说明 proxy-guided 策略需要 held-out matched rollouts 验证。

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

计划: [`docs/phase29_finer_duration_proxy_plan_zh.md`](phase29_finer_duration_proxy_plan_zh.md)

脚本:

- [`toy_quantvla/run_phase29_finer_duration_proxy.sh`](../toy_quantvla/run_phase29_finer_duration_proxy.sh)
- [`toy_quantvla/phase29_finer_duration_proxy_summary.py`](../toy_quantvla/phase29_finer_duration_proxy_summary.py)

Phase29 会把 `0-250` 拆成更细的 grasp-centered windows，例如 `0:120,0:180,0:220,80:240,120:280,160:240,240:320`。目标是找到比 broad `0-250` 更便宜的 duration proxy：尽量保留 repair，同时减少 eager fallback 和 latency。

## 阅读顺序

建议按这个顺序看：

1. [`paper/main.pdf`](../paper/main.pdf)
2. [`docs/phase25_anisotropic_sensitivity_report_zh.md`](phase25_anisotropic_sensitivity_report_zh.md)
3. [`docs/phase28_proxy_guided_mixed_precision_report_zh.md`](phase28_proxy_guided_mixed_precision_report_zh.md)
4. [`docs/phase28b_proxy_guided_33case_report_zh.md`](phase28b_proxy_guided_33case_report_zh.md)
5. [`docs/phase28c_proxy_guided_33case_report_zh.md`](phase28c_proxy_guided_33case_report_zh.md)
6. [`docs/phase28d_duration_window_analysis_zh.md`](phase28d_duration_window_analysis_zh.md)
7. [`docs/phase29_finer_duration_proxy_plan_zh.md`](phase29_finer_duration_proxy_plan_zh.md)
