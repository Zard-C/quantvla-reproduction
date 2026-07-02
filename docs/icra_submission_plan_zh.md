# ICRA 投稿收敛计划

这份计划把当前技术报告收敛成 ICRA regular paper + video + arXiv technical report 的双轨路线。原则是：**ICRA 只投一个匿名主稿；arXiv 作为公开长版，不和其他会议双投。**

## 目标版本

- `paper/`: arXiv 技术报告长版，保留完整推导、长表、附录和 artifact 说明。
- `paper_icra/`: ICRA 匿名短版，目标 8 页主文，突出机器人闭环行为、视频证据和设计 guide。
- `docs/`: 实验报告、阶段记录、视频脚本和 rebuttal 风险清单。

## ICRA 主线

论文不是“QuantVLA 复现”，而是：

> VLA inference acceleration is a closed-loop policy perturbation problem.

核心结构：

1. 推理加速会改变闭环 state distribution。
2. 误差影响由 dynamics、policy feedback、success margin 过滤。
3. open-loop drift 和 aggregate success 都不够，需要 paired repair/regression。
4. not all dims / durations / layers are equal。
5. sensitivity-guided protection 可以在速度和闭环行为之间找到更好的点。

可以把工程路径类比成 TensorRT tactic search，但闭环验证更贵：

```text
compiler tactic search:
candidate kernels -> microbenchmark latency -> local numerical check -> fastest valid tactic

closed-loop VLA implementation search:
candidate acceleration policies -> cheap sensitivity proxy -> small matched rollouts -> held-out validation
```

所以我们的 guide 不是暴力枚举所有 quantization / compile / fallback 候选，而是 **sensitivity-guided implementation search**：先用便宜的静态/open-loop proxy 剪枝，再把昂贵的 closed-loop rollout 留给少数高收益、低风险候选。

## 当前主结果

### N1.5 discovery probe

| mode | success | server p50 | speedup | 备注 |
| --- | ---: | ---: | ---: | --- |
| FP16 baseline | 19/33 | 156.22 ms | 1.00x | 33-case matched probe |
| speed-only compile | 16/33 | 70.20 ms | 2.23x | 快，但闭环 regress |
| blocks0-3 eager | 18/33 | 67.58 ms | 2.31x | layer proxy 有效但不满血 |
| duration 0-120 | 19/33 | 69.66 ms | 2.24x | probe 上最佳 speed/behavior trade-off |
| duration 0-250 | 18/33 | 88.75 ms | 1.76x | 保护太宽，损失速度 |

`0-120` 相对 speed-only 修复 7 个 case，regress 4 个 case，净 `+3`；相对 FP16 是 2 repair / 2 regress，aggregate 持平。这支持“加速是在附近解空间里探索，而不是单点模仿 FP16”。

### N1.5 multi-fold validation

Phase34 把两个 30-case held-out folds 合并比较：

| tactic | pooled success | worst success | mean speedup | worst speedup | total regress |
| --- | ---: | ---: | ---: | ---: | ---: |
| speed_only | 45/60 | 0.667 | 2.01x | 1.75x | 7 |
| window_0_120 | 47/60 | 0.733 | 1.84x | 1.71x | 3 |
| combo_blocks0_3_window_0_120 | 47/60 | 0.733 | 1.41x | 1.07x | 1 |

结论不是 `0-120` universal，而是：不同 objective 会选不同 Pareto 点。Behavior-first 偏向更少 regression 的 combo；speed-constrained 偏向 `window_0_120`。

### N1.7 held-out transfer check

Phase37B 在 N1.7 checkpoint 上跑了 15 个 held-out cases：

| tactic | success | p50 | speedup | paired vs FP16 |
| --- | ---: | ---: | ---: | --- |
| FP16 baseline | 13/15 | 110.40 ms | 1.00x | - |
| speed_only | 11/15 | 74.79 ms | 1.48x | 1 repair / 3 regress |
| window_0_20 | 14/15 | 100.98 ms | 1.09x | 2 repair / 1 regress |
| routed tactic | 13/15 | 79.07 ms | 1.40x | 2 repair / 2 regress |

这说明 tactic-search 流程可以迁移到新 checkpoint，但具体 tactic 仍然要重新估计。Task-conditioned routing 可以作为 high-speed Pareto 点，但不是 safe default。

## 视频附件脚本

ICRA 审稿人最需要看到：小扰动如何变成轨迹分叉。

建议 4 组 case，每组做三列对比：FP16 / speed-only compile / duration 0-120。

| case | 目的 | 期望展示 |
| --- | --- | --- |
| `4:6` | 0-120 repair, 回到 baseline-like basin | speed-only F990，0-120 S241，EEF divergence 后期才明显 |
| `6:0` | 0-120 repair, 快速恢复成功 | speed-only F990，0-120 S205 |
| `6:3` | 0-120 发现新成功分支 | FP16 和 speed-only 都不是该成功分支，0-120 S376 |
| `6:10` | 0-120 regression | speed-only S225，0-120 F990，说明保护也会破坏 beneficial branch |

每组视频建议 overlay：

- task/init id
- mode 名称
- success/fail 和 episode steps
- first action divergence step
- first EEF > 1 cm / > 5 cm step
- gripper mismatch step，如果有

## 投稿前最小工作包

优先级从高到低：

1. **ICRA short version**
   从 `paper/main.tex` 裁剪出匿名 8 页主文，保留理论框架、五个 claim、核心表格、Pareto 图和 Algorithm 1；长表、keyframes 和 artifact 细节放 supplement/arXiv。

2. **video/keyframe supplement**
   把代表性 repair/regression case 做成短视频或 contact sheet，突出 first divergence、EEF drift、gripper/contact 变化和最终 outcome flip。

3. **latency/profiling wording**
   主文只报告 warm server p50 speedup；冷启动 compile spike、max latency 和 prototype fallback 开销放到 limitations 或 supplement，避免被误读成最终部署吞吐。

4. **anonymous artifact plan**
   ICRA 主稿不要放 GitHub/arXiv 链接；supplement 可以匿名化脚本结构。arXiv 长版保留完整 artifact 链接。

## 主要风险和应对

| 风险 | 应对 |
| --- | --- |
| 单模型单仿真 | 明确本文贡献是 evaluation/design guide；GR00T/LIBERO 是 case study。 |
| fake quantization 不是真实 kernel | 主线扩展为 inference acceleration；量化是行为扰动证据，compile 是速度证据。 |
| 0-120 是人工 proxy | 承认它是 sensitivity-guided candidate，不是 universal tactic；用 multi-fold 和 N1.7 结果证明最终对象是 tactic search。 |
| aggregate gap 小 | 强调 paired repair/regression 和 trace divergence，而不是宣称 policy dominance。 |
| compile/eager fallback 不是真实部署 | 把它定位成 prototype acceleration boundary；后续 packed kernel 是工程扩展。 |
| routed tactic 像过拟合 | 明确 routing 是 high-speed Pareto point，本身需要 held-out validation，不作为 safe default。 |

## 下一步执行

1. 从 `paper/main.tex` 建立 `paper_icra/` 匿名短版骨架。
2. 选择 4-6 个最能说明 repair/regression、duration sensitivity 和 routing trade-off 的视频/关键帧 case。
3. 把 arXiv 长版继续作为完整技术报告维护，ICRA 主稿只保留最强证据链。
4. 准备 rebuttal 风险清单：单模型、fake quant、prototype compile、小规模 N1.7、task-level routing。
5. 最后做格式审查：表格不压 references、公式引用正常、图中文字对齐、匿名信息清理。

## 最后冲刺目标

最后一档冲刺不是继续手工找更多窗口，而是把经验 guide 固化成一个可复用算法：

> Closed-Loop Sensitivity-Guided Tactic Search 自动寻找并验证 acceleration tactic。

这个算法的目标不是证明存在固定 universal tactic，而是提出一个可迁移的 tactic-search procedure。输入新模型、新权重或新任务分布后，它用少量 cheap proxy 和少量 matched rollouts 自动选择候选 implementation tactic。

当前可交付形态已经具备：

1. Phase29 展示 probe discovery。
2. Phase34 展示 multi-fold objective 会改变 tactic ranking。
3. Phase37B 展示新 checkpoint 上同一流程可比较 fixed tactic 与 task-conditioned routing。
4. `paper/main.pdf` 已把它写成 Algorithm 1。

详细计划见 [`docs/phase31_auto_tactic_search_plan_zh.md`](phase31_auto_tactic_search_plan_zh.md)。
