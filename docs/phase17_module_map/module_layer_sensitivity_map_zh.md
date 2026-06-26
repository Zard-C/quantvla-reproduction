# Phase 17: 模块/层级敏感性地图

## 结论摘要

这张地图把目前的证据收敛成一个判断：闭环敏感性主要集中在 action head 的 DiT 路径，尤其是大图编译后的前段 block0 compiled path；LLM/DiT MLP 的低比特替换也会改变轨迹分布，但当前更像行为扰动源，不是已经可用的加速 backend。

最重要的更新是：早期 Phase 13 看到 blocks 8-15 eager island 能救回部分 case，曾让后半 blocks 看起来像主要风险区；但 Phase 16.6 的 step253 bisect 进一步把一个稳定 outlier 缩到 `block0 compiled path`。所以现在的地图不是“后半 blocks 单点敏感”，而是“action_head.model 大图路径存在多处敏感边界，block0 是已定位的确定性尖峰源”。

## Action Head 结构

- action head: `FlowmatchingActionHead`
- action head model: `DiT`
- DiT transformer blocks: `16`
- transformer block total parameters: `541,335,552`

| block | linear layers | leaf params |
| --- | --- | --- |
| 0 | 7 | 34,619,904 |
| 1 | 7 | 33,047,040 |
| 2 | 7 | 34,619,904 |
| 3 | 7 | 33,047,040 |
| 4 | 7 | 34,619,904 |
| 5 | 7 | 33,047,040 |
| 6 | 7 | 34,619,904 |
| 7 | 7 | 33,047,040 |
| 8 | 7 | 34,619,904 |
| 9 | 7 | 33,047,040 |
| 10 | 7 | 34,619,904 |
| 11 | 7 | 33,047,040 |
| 12 | 7 | 34,619,904 |
| 13 | 7 | 33,047,040 |
| 14 | 7 | 34,619,904 |
| 15 | 7 | 33,047,040 |

每个 transformer block 的参数规模基本一致，因此敏感性差异不能简单归因于某个 block 参数更多；更可能来自 block 所处阶段、residual/norm/attention/FFN 组合路径，以及编译边界造成的数值路径变化。

## Same-Observation Compile Scope 地图

| target | steps | cont max p99 | cont max max | cont L2 p99 | compiled p50 | speedup | sensitivity |
| --- | --- | --- | --- | --- | --- | --- | --- |
| action_head_dit_ff_all | 484 | 0.002975 | 0.006409 | 0.003216 | 123.5 ms | 0.73x | low-drift-slow |
| action_head_dit_attn_all | 484 | 0.00309 | 0.006409 | 0.003223 | 102.9 ms | 0.9x | low-drift-slow |
| action_head_model_blocks_6_15_eager | 484 | 0.00412 | 0.054932 | 0.004223 | 93.9 ms | 1.09x | high |
| action_head_model | 484 | 0.004349 | 0.054932 | 0.004995 | 63.3 ms | 2.06x | high |
| action_head_model_blocks_8_15_eager | 484 | 0.004463 | 0.054932 | 0.004657 | 87.9 ms | 1.2x | high |
| action_head_dit_blocks_0_7 | 0 | - | - | - | - | - | invalid |

读法：小 scope 的 `attn_all` / `ff_all` 漂移最低，但速度没有收益；大图 `action_head_model` 快，但存在 `0.054932` 连续动作尖峰；后半 eager island 没有消掉这个尖峰。

## Step253 层级 Bisect

| target | top step | dim | cont max | L2 p99 | spike removed | compiled p50 | speedup |
| --- | --- | --- | --- | --- | --- | --- | --- |
| action_head_model | 253 | x | 0.054932 | 0.077386 | False | 54.2 ms | 2.41x |
| action_head_model_blocks_0_0_eager | 253 | z | 0.001831 | 0.002533 | True | 64.8 ms | 2.11x |
| action_head_model_blocks_1_1_eager | 253 | x | 0.054474 | 0.076141 | False | 67.8 ms | 1.9x |
| action_head_model_blocks_2_2_eager | 253 | x | 0.054932 | 0.076448 | False | 66.9 ms | 1.97x |
| action_head_model_blocks_3_3_eager | 253 | x | 0.055389 | 0.077135 | False | 64.7 ms | 2.13x |
| action_head_model_blocks_4_7_eager | 253 | x | 0.054932 | 0.076953 | False | 73.3 ms | 1.68x |
| action_head_model_blocks_8_15_eager | 253 | x | 0.054932 | 0.075739 | False | 82 ms | 1.4x |
| action_head_dit_attn_all | 253 | x | 0.001831 | 0.002542 | True | 102.7 ms | 1.01x |

关键结论：`blocks_0_0_eager` 能消除 step253 尖峰；`blocks_1_1_eager`、`blocks_2_2_eager`、`blocks_3_3_eager`、`blocks_4_7_eager`、`blocks_8_15_eager` 都不能。当前最小已知责任范围是 `DiT block0 compiled path`。

## Block-Level Heatmap

| block/scope | evidence target | cont max | spike removed | speedup | note |
| --- | --- | --- | --- | --- | --- |
| block0 | action_head_model_blocks_0_0_eager | 0.001831 | True | 2.11x | 只保留 block0 eager 即可消除 step253 spike；当前最强层级定位。 |
| block1 | action_head_model_blocks_1_1_eager | 0.054474 | False | 1.9x | 单独 block1 eager 不能消除 step253 spike。 |
| block2 | action_head_model_blocks_2_2_eager | 0.054932 | False | 1.97x | 单独 block2 eager 不能消除 step253 spike。 |
| block3 | action_head_model_blocks_3_3_eager | 0.055389 | False | 2.13x | 单独 block3 eager 不能消除 step253 spike。 |
| blocks2-3 | action_head_model_blocks_2_3_eager | 0.054932 | False | 1.84x | block2-3 eager 不能消除 step253 spike；问题不在这个后续小段。 |
| blocks4-7 | action_head_model_blocks_4_7_eager | 0.054932 | False | 1.68x | block4-7 eager 不能消除 step253 spike。 |
| blocks8-15 | action_head_model_blocks_8_15_eager | 0.054932 | False | 1.4x | 历史上改善部分 15-case，但 30-case 净回归，且不消除 task6 step253 spike。 |

这张 heatmap 只对 step253 这个已定位 outlier 有效，不等于所有任务所有阶段的全局敏感性。它的价值在于给出了一个可以工程化验证的最小保护候选：`action_head_model_blocks_0_0_eager`。

## Matched Rollout 工程地图

| comparison | baseline | candidate | delta | baseline p50 | candidate p50 | p50 speedup |
| --- | --- | --- | --- | --- | --- | --- |
| blocks_8_15_eager_vs_fp16_30 | 16/30 | 13/30 | -3 | 161.1 ms | 81 ms | 1.99x |
| blocks_6_15_eager_vs_fp16_5_flip | 3/5 | 1/5 | -2 | 161.1 ms | 132.8 ms | 1.21x |
| fp4_llm_up_proj_vs_fp16_15 | 7/15 | 9/15 | 2 | 95.2 ms | 96.6 ms | 0.99x |
| fp4_dit_mlp_vs_fp16_15 | 7/15 | 7/15 | 0 | 95.2 ms | 147.7 ms | 0.64x |
| residual_cache_b12_15_vs_fp16_15 | 7/15 | 5/15 | -2 | 95.2 ms | 161.3 ms | 0.59x |

这里能看到几个工程事实：`blocks_8_15_eager` 有接近 2x p50，但 30-case 成功率从 16/30 到 13/30；`blocks_6_15_eager` 更保守但速度掉到约 1.2x，行为也没有改善；DiT MLP FP4 15-case 行为持平但更慢；residual cache 离线漂移低，但闭环 15-case 回归。

## Quant Scope Drift 补充

| scope | mode | modules | episodes | raw RMSE mean | raw RMSE max | LIBERO max diff | student/teacher time |
| --- | --- | --- | --- | --- | --- | --- | --- |
| llm_mlp_only | none | 36 | 2 | 0.011908 | 0.031984 | 0.026093 | - |
| dit_mlp_only | none | 32 | 4 | 0.006719 | 0.011616 | 0.012817 | - |
| llm_dit_mlp | none | 116 | 2 | 0.026216 | 0.051182 | 0.029755 | 8.64 |
| llm_dit_mlp | atm_ohb | 116 | 2 | 0.026071 | 0.053025 | 0.02655 | 5.809 |

这组是 teacher-controlled online drift，不是闭环成功率。它说明量化 scope 越大，动作漂移整体会变大；ATM/OHB 能缓解 full selected scope 的漂移，但不能把问题变成单调安全。

## 当前模块优先级

| module/scope | sensitivity | evidence | next action |
| --- | --- | --- | --- |
| DiT block0 compiled path | high | step253 spike can be removed by block0 eager only | 优先做 block0 eager closed-loop 或 block0 内部 hook。 |
| DiT blocks8-15 eager island | medium/high | keeps ~2x p50 but 30-case 13/30 vs FP16 16/30 | 可作为速度候选，但不是行为透明 backend。 |
| DiT attention/FF small scopes | low drift / low utility | continuous max <=0.0064 but compiled p50 not faster | 适合诊断，不适合部署加速。 |
| LLM up_proj FP4 | behavior-changing | 15-case seeded 9/15 vs FP16 7/15, but speed not improved | 可作为轨迹重分配研究对象，不是稳定加速结论。 |
| DiT MLP FP4 | medium | 15-case 7/15 vs FP16 7/15 but p50 slower | kernel/packing 成本没解决前不适合 rollout 扩大。 |
| DiT residual cache b12,b15 | offline-stable but closed-loop risky | offline drift low; 15-case 5/15 vs FP16 7/15 | 缓存类方法也要通过闭环门槛，不能只看 action drift。 |

## 对后续工程的含义

1. 如果继续 torch.compile 路线，下一步最值得验证的是 `action_head_model_blocks_0_0_eager` 的小规模 closed-loop，而不是继续盲目扩大后半 eager island。
2. 如果继续 FP4/CUTLASS 路线，不能只按模块数量扩大 scope；要先用 same-observation continuous drift 和 first-divergence 过滤。
3. LLM up_proj、DiT MLP、residual cache 都能改变轨迹分布，但目前都没有同时满足“速度收益 + 闭环透明”。
4. 真正有价值的敏感性指标应该绑定模块边界：`module scope -> continuous drift -> first divergence -> outcome flip`，而不是只看 module-level average MSE。

## 限制

这张地图汇总的是已有实验，不是新的 GPU benchmark。某些结论是 case-specific：例如 block0 定位来自 `task6:init8 step253`，后半 blocks 结论来自 Phase 13/14 的 matched subsets。它适合指导下一步实验优先级，不应被解读成所有 VLA 模型的普适层级定律。

## 产物

- JSON: `toy_quantvla/results/phase17_module_layer_sensitivity_map.json`
- Report: `docs/phase17_module_map/module_layer_sensitivity_map_zh.md`
