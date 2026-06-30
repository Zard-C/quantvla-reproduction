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

| mode | success | server p50 | speedup | 备注 |
| --- | ---: | ---: | ---: | --- |
| FP16 baseline | 19/33 | 156.22 ms | 1.00x | 33-case matched set |
| speed-only compile | 16/33 | 70.20 ms | 2.23x | 快，但闭环 regress |
| blocks0-3 eager | 18/33 | 67.58 ms | 2.31x | layer proxy 有效但不满血 |
| duration 0-120 | 19/33 | 69.66 ms | 2.24x | 当前最佳 speed/behavior trade-off |
| duration 0-250 | 18/33 | 88.75 ms | 1.76x | 保护太宽，损失速度 |

`0-120` 相对 speed-only 修复 7 个 case，regress 4 个 case，净 `+3`；相对 FP16 是 2 repair / 2 regress，aggregate 持平。这正好支持“加速是在附近解空间里探索，而不是单点模仿 FP16”。

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

## 还值得补的最小实验

优先级从高到低：

1. **held-out sanity set**
   再选 20-30 个未用于 Phase29 选择的 task/init，只跑 FP16、speed-only、0-120、blocks0-3。目标不是刷大分，而是回答“0-120 是否只是 33-case 过拟合”。

2. **memory/profiling table**
   对 FP16、speed-only、0-120 统计 warm latency、prepare/compile cost、峰值显存、server request 数。ICRA 论文里要说清 cold compile 不是部署 steady-state。

3. **video/keyframe supplement**
   把上述 4 个 case 做成短视频或 contact sheet，补到 supplementary。

4. **anonymous artifact plan**
   ICRA 主稿不要放 GitHub/arXiv 链接；supplement 可以匿名化脚本结构。arXiv 长版保留完整 artifact 链接。

## 主要风险和应对

| 风险 | 应对 |
| --- | --- |
| 单模型单仿真 | 明确本文贡献是 evaluation/design guide；GR00T/LIBERO 是 case study。 |
| fake quantization 不是真实 kernel | 主线扩展为 inference acceleration；量化是行为扰动证据，compile 是速度证据。 |
| 0-120 是人工 proxy | 承认它是 sensitivity-guided proxy，不是自动最优；补 held-out sanity。 |
| aggregate gap 小 | 强调 paired repair/regression 和 trace divergence，而不是宣称 policy dominance。 |
| compile/eager fallback 不是真实部署 | 把它定位成 prototype acceleration boundary；后续 packed kernel 是工程扩展。 |

## 下一步执行

1. 等 Phase30 held-out sanity set 跑完，判断 `0-120` 是否仍然不低于 speed-only。
2. 生成 video case list 和素材路径。
3. 补 memory/profiling table。
4. 根据 Phase30 结果更新 `paper_icra/main.tex`。
5. arXiv 长版补 artifact/reproducibility 段落。

## 最后冲刺目标

如果 Phase30 验证通过，最后一档冲刺不是继续手工找更多窗口，而是把经验 guide 升级为一个小算法：

> Sensitivity-Guided Tactic Search 自动寻找类似 `0-120` 的 tactic。

这个算法的目标不是证明存在固定 universal tactic，而是提出一个可迁移的 tactic-search procedure。输入新模型、新权重或新任务分布后，它用少量 cheap proxy 和少量 matched rollouts 自动选择候选 implementation tactic。

最小可交付形态：

1. 在已有 Phase28/29 数据上做 retrospective search，展示算法能从候选集合中选出 `0-120` 或同级别候选。
2. 用 Phase30 held-out sanity set 检查该选择是否能保持 speed/behavior trade-off。
3. 在 ICRA 版中把它作为 algorithmic guide，而不是重写成完整系统论文。

详细计划见 [`docs/phase31_auto_tactic_search_plan_zh.md`](phase31_auto_tactic_search_plan_zh.md)。
