# ICRA Video / Keyframe Supplement Plan

这份文档用于收束 ICRA 投稿的 qualitative supplement。目标不是继续扩实验，而是把现有关键帧、trace divergence 和 paired rollout 结果组织成一条审稿人能快速看懂的证据链：

```text
small inference perturbations -> trajectory branching -> repair/regression -> tactic selection
```

当前本地仓库没有发现 `mp4/mov/webm/gif` 成片文件；已有材料主要是 contact sheets、关键帧图、trace JSON 和 divergence 报告。因此 supplement 分两档：

1. **最小可交付版**：直接使用现有 contact sheets / trace plots，做 PDF supplement 或主文 appendix 图。
2. **视频增强版**：如果后续能从 5090 找回原始 rollout videos，或重新渲染少量病例，再导出 4-6 组短视频。

## 选择原则

每个病例都要满足至少两个条件：

- 有 paired outcome flip，不只是 aggregate success 差异。
- 能对应论文里的一个 claim：量化闭环扰动、margin/basin、duration/layer sensitivity、tactic search trade-off。
- 已经有本地关键帧或 trace 证据，避免为 supplement 再开新一轮大实验。
- 同时包含 repair 和 regression，避免把故事讲成“扰动总是有益”或“保护总是安全”。

## 主推荐病例

| id | case | 对比模式 | outcome | 现有材料 | 主要信息 |
| --- | --- | --- | --- | --- | --- |
| Q1 | `task8:init7` | FP16 / W4A8 none / W4A8 ATM+OHB | FP16 F991; ATM+OHB F991; none S388 | [`analysis_keyframes/batch2/none_repair_task8_init7.jpg`](../analysis_keyframes/batch2/none_repair_task8_init7.jpg), [`paper/figures_keyframes/none_repair_task8_init7.jpg`](../paper/figures_keyframes/none_repair_task8_init7.jpg) | raw quantization 把一个 FP16 failure slice 推入更短的成功分支。 |
| Q2 | `task4:init10` | FP16 / W4A8 none / W4A8 ATM+OHB | FP16 F991; none F991; ATM+OHB S242 | [`analysis_keyframes/batch2/atmohb_repair_task4_init10.jpg`](../analysis_keyframes/batch2/atmohb_repair_task4_init10.jpg), [`paper/figures_keyframes/atmohb_repair_task4_init10.jpg`](../paper/figures_keyframes/atmohb_repair_task4_init10.jpg) | ATM+OHB 可以修复接触时序或物体交互顺序。 |
| Q3 | `task0:init3` | FP16 / W4A8 none / W4A8 ATM+OHB | FP16 S250; ATM+OHB S267; none F991 | [`analysis_keyframes/regressions/none_regress_task0_init3.jpg`](../analysis_keyframes/regressions/none_regress_task0_init3.jpg), [`paper/figures_keyframes/none_regress_task0_init3.jpg`](../paper/figures_keyframes/none_regress_task0_init3.jpg) | raw quantization 也会破坏原本快速成功的 FP16 轨迹。 |
| Q4 | `task8:init0` | FP16 / W4A8 none / W4A8 ATM+OHB | FP16 S657; none F991; ATM+OHB F991 | [`analysis_keyframes/regressions/both_quant_regress_task8_init0.jpg`](../analysis_keyframes/regressions/both_quant_regress_task8_init0.jpg), [`paper/figures_keyframes/both_quant_regress_task8_init0.jpg`](../paper/figures_keyframes/both_quant_regress_task8_init0.jpg) | 同一任务族既有 quant repair 又有 quant regression，说明敏感性是 init/basin dependent。 |
| C1 | `task4:init6` | FP16 baseline / speed-only compile / protected tactic | baseline S245; speed-only F990/F991; proxy_blocks8_15 S241; window_0_120 S241 | [`docs/phase13_visuals/phase13_reduce_task4_init6_alignment.png`](phase13_visuals/phase13_reduce_task4_init6_alignment.png), [`docs/phase13_visuals/phase13_reduce_task4_init6_compiled_tail.png`](phase13_visuals/phase13_reduce_task4_init6_compiled_tail.png), [`analysis_keyframes/task4_init6_mug_plate_contact.jpg`](../analysis_keyframes/task4_init6_mug_plate_contact.jpg) | speed-only compile 会退化；保护敏感部分可恢复接近 baseline 的成功 basin。 |
| C2 | `task6:init0` | FP16 baseline / speed-only compile / protected tactic | baseline S210; speed-only F990/F991; proxy_blocks8_15 S206; window_0_120 S205 | [`docs/phase13_visuals/phase13_reduce_task6_init0_alignment.png`](phase13_visuals/phase13_reduce_task6_init0_alignment.png), [`docs/phase13_visuals/phase13_reduce_task6_init0_compiled_tail.png`](phase13_visuals/phase13_reduce_task6_init0_compiled_tail.png) | action divergence 先出现，EEF divergence 后出现；小扰动经过闭环后变成 outcome flip。 |

其中 Q1-Q4 支撑“量化误差不是单纯 open-loop drift，而是改变闭环 trajectory basin membership”；C1-C2 支撑“compile/tactic 也是推理加速扰动，需要 closed-loop sensitivity-guided search”。

## 备选病例

| case | outcome / trace | 用途 |
| --- | --- | --- |
| `task6:init1` | FP16 S222; none S477; ATM+OHB F991; sheet: [`analysis_keyframes/regressions/atmohb_regress_task6_init1.jpg`](../analysis_keyframes/regressions/atmohb_regress_task6_init1.jpg) | 展示 ATM+OHB 不是单调安全，过度补偿或 timing shift 可能失败。 |
| `task8:init9` | speed-only S476; proxy_blocks8_15 S424; quant ATM+OHB repair sheet: [`analysis_keyframes/batch/atmohb_repair_task8_init9.jpg`](../analysis_keyframes/batch/atmohb_repair_task8_init9.jpg) | 展示 protected tactic 可以保留 beneficial speed-only branch。 |
| `task4:init9` | baseline S224; speed-only S222; proxy_blocks8_15 F990 | 展示 coarse layer proxy 会制造新 regression，说明需要 held-out validation。 |
| `task6:init10` | window_0_120 regression in Phase29 | 展示 duration protection 也可能破坏本来有益的分支。 |

备选病例适合放到 appendix 表格，不一定进入视频主线。视频主线控制在 4-6 组，否则审稿人很难看完。

## Trace 证据摘录

这些数值用于给视频或 keyframe 加 overlay。

| case | comparison | first action divergence | first EEF > 1 cm | first EEF > 5 cm | max EEF drift | 解释 |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| `4:6` | speed-only vs proxy_blocks8_15 | 35 | 83 | 192 | 0.17997 m | 动作分叉早于可见状态分叉；proxy 恢复短成功分支。 |
| `6:0` | speed-only vs proxy_blocks8_15 | 62 | 120 | 143 | 0.13179 m | speed-only 失败，proxy 恢复 baseline-like success。 |
| `8:10` | speed-only vs proxy_blocks8_15 | 87 | 141 | 467 | 0.48578 m | proxy 修复 outcome，但走更长的新 branch。 |
| `8:9` | speed-only vs proxy_blocks8_15 | 55 | 157 | 166 | 0.38552 m | proxy 保留 speed-only 的 beneficial branch。 |
| `4:9` | speed-only vs proxy_blocks8_15 | 57 | 104 | 125 | 0.18282 m | proxy regression，说明保护敏感层不是单调安全。 |

这些数据来自 [`docs/phase28_proxy_guided_trace_divergence_zh.md`](phase28_proxy_guided_trace_divergence_zh.md)。Phase29 的 finer duration report 进一步说明 `window_0_120` 在 probe set 上相对 speed-only 修复 7 个 case，同时新增 4 个 regression；它是 duration sensitivity 的证据，不是 universal tactic。

## 视频结构

建议总时长控制在 2-3 分钟。

### 开场

一页静态图或字幕：

```text
Inference acceleration changes closed-loop trajectory basins.
We diagnose this through paired rollouts, first-divergence traces, and tactic search.
```

### Block A: Quantization-Induced Trajectory Redistribution

病例：Q1-Q4。

每组用三列或三行同步展示：

```text
FP16 baseline
W4A8 llm_dit_mlp, none
W4A8 llm_dit_mlp, ATM+OHB
```

重点标注：

- success/failure 与 episode steps。
- 早期 branch selection，而不是只看最后一帧。
- repair 与 regression 各至少两例。

### Block B: Compile/Tactic-Induced Branching

病例：C1-C2。

每组展示：

```text
FP16 baseline
speed-only compile
protected tactic
```

`protected tactic` 可按材料选择：

- trace 图更完整时用 `proxy_blocks8_15`。
- 与最终论文 tactic search 更一致时用 `window_0_120` 或组合 tactic。

重点标注：

- first action divergence step。
- first EEF > 1 cm / 5 cm step。
- speed-only regression 如何被 protected tactic 修复。
- protected tactic 不是最终答案，只是 CLSG-TS 搜索空间里的一个候选。

### 结尾

放 Pareto 图和 Algorithm 1 的简化流程：

```text
generate candidate tactics -> probe paired rollouts -> measure speed and regressions -> select behavior-first or speed-constrained tactic
```

这能把 qualitative 病例和论文算法接起来，避免 supplement 只是“好看的失败视频”。

## Overlay 字段

每个视频片段建议统一使用以下字段：

| 字段 | 示例 |
| --- | --- |
| case id | `task4:init6` |
| mode | `FP16`, `speed-only`, `window_0_120`, `W4A8 none`, `W4A8 ATM+OHB` |
| outcome | `S241`, `F990`, `F991` |
| divergence | `first action > 0.05: step 35` |
| state drift | `EEF > 1 cm: step 83; EEF > 5 cm: step 192` |
| latency | `p50 69.66 ms, 2.24x` for compile/tactic cases only |

视频里不要出现本地路径、用户名、服务器地址、GitHub 用户名或 token。ICRA 匿名版只保留 case id 和方法名。

## 最小制作清单

| 任务 | 状态 | 备注 |
| --- | --- | --- |
| 确认可用 contact sheets | done | Q1-Q4 和 C1-C2 都有本地图片或 trace 图。 |
| 确认本地视频文件 | blocked | 当前仓库未发现 `mp4/mov/webm/gif`。 |
| 生成 PDF supplement 页面 | ready | 可直接引用本文件列出的图片。 |
| 找回或重渲染 raw videos | optional | 只需要 4-6 个病例，不需要重新跑大规模 eval。 |
| 添加统一 overlay | pending | 有视频后再做；关键帧版可用 caption 替代。 |
| 匿名化检查 | pending | 投稿前统一检查图片/视频是否暴露路径和账号信息。 |

## 和论文 claim 的对应关系

| 论文观点 | 对应病例 |
| --- | --- |
| quantization error is filtered by closed-loop sensitivity | Q1-Q4 |
| rollout flip depends on margin/basin crossing, not just perturbation norm | Q1, Q3, Q4 |
| open-loop drift is necessary but insufficient | C1, C2 trace divergence |
| not all durations/layers are equal | C1, C2, `task4:init9` backup |
| CLSG-TS is a reusable selection procedure | Block B + Pareto 图 + Algorithm 1 |

## 推荐下一步

1. ICRA 主稿保持 Figure 3 + Algorithm 1 + 核心表格，不再塞更多关键帧。
2. Supplement 先做关键帧 PDF 版，使用 Q1-Q4 + C1-C2。
3. 如果 5090 上能找回原始视频，只重导出这 6 组，不重跑完整 benchmark。
4. 若必须重新推理，优先重跑 C1/C2，因为它们最直接支持“推理加速不只是量化，也包括 compile/tactic”的新主线。
