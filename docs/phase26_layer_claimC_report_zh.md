# Phase 26: Layer/Boundary Closed-Loop Intervention

本阶段补强 Phase25 的第三个判断：`not all layers are equal`。实验不是重新做量化，而是用 torch.compile 的 action-head compile boundary 作为可控 layer/boundary intervention：外层都围绕 `action_head.model`，只改变 eager island 的 layer 位置。

## 设置

- case list: `4:9,6:8`
- deterministic policy seed base: `20260613`
- model: GR00T N1.5 LIBERO long posttrain
- comparison: FP16 baseline vs full `action_head_model` compile vs `block0 eager` vs `block1 eager`

## 结果表

| run | compile target | eager island | success | per-case outcome / steps | p50 ms | speedup |
| --- | --- | --- | --- | --- | --- | --- |
| FP16 baseline | none | - | 2/2 | 4:9=success/224, 6:8=success/649 | 160.53 | 1.0x |
| compile action_head_model | action_head_model | - | 2/2 | 4:9=success/222, 6:8=success/916 | 72.8 | 2.21x |
| compile action_head_model + block0 eager | action_head_model_blocks_0_0_eager | 0 | 1/2 | 4:9=fail/991, 6:8=success/404 | 72.62 | 2.21x |
| compile action_head_model + block1 eager | action_head_model_blocks_1_1_eager | 1 | 0/2 | 4:9=fail/991, 6:8=fail/991 | 73.1 | 2.2x |

## 按 case 看

| case | outcomes |
| --- | --- |
| 4:9 | fp16_baseline=success/224; action_head_model=success/222; blocks_0_0_eager=fail/991; blocks_1_1_eager=fail/991 |
| 6:8 | fp16_baseline=success/649; action_head_model=success/916; blocks_0_0_eager=success/404; blocks_1_1_eager=fail/991 |

## 关键观察

1. `action_head_model` full compile 在这两个 case 上没有直接造成失败：`2/2` 成功，而且 p50 从 baseline 的 `160.53ms` 降到 `72.80ms`，约 `2.21x`。
2. 但 full compile 不是行为透明的：`6:8` 从 baseline 的 649 步变成 916 步。也就是说它虽然最终成功，但闭环轨迹已经明显重分配。
3. `block0 eager` 并没有在闭环上变得更安全：`4:9` 从成功变成 991 步失败；`6:8` 反而从 649 步变成 404 步成功。它是明显的 case-dependent intervention。
4. `block1 eager` 更差：两个 case 都失败，说明不是随便把某个 block 留在 eager 就能提高鲁棒性。
5. 三个 compiled variants 的 p50 都在 `72-73ms` 左右，但成功率分别是 `2/2`、`1/2`、`0/2`。因此工程上不能只看推理速度，也不能只看同观测 drift outlier 是否被消掉。

## 对 Claim C 的含义

这轮结果比简单的 `block0 是敏感层` 更强，也更谨慎：layer/boundary 的作用不是单调的。相同外层 compile 目标、相近 runtime p50，只改变 eager island 的 layer 位置，就能把闭环结果从 `2/2` 改成 `1/2` 或 `0/2`。

所以更准确的 claim 是：

```text
Not all layer/boundary perturbations are equal; their effect is filtered by task, phase, action direction, and closed-loop margin.
```

这也解释了为什么 Phase17 里 `block0 eager` 能消除 step253 same-observation spike，但 Phase26 中它仍然会让 `4:9` 闭环失败：same-observation spike removal 只是局部数值指标，不等价于 full rollout robustness。

## 下一步

1. 对这三个 variants 做 first-divergence 分析，尤其比较 `4:9` 中 full compile 成功而 block0/block1 eager 失败的最早分叉点。
2. 把 `action_head_model` full compile 纳入更大 case set，确认它是否只是这两个 case 上好运，还是可以作为速度候选。
3. 将 layer intervention 和 Phase25 的 action-channel/phase sensitivity map 关联，检查失败是否集中投影到 `y/rotation` 等敏感方向。

## 产物

- JSON: `toy_quantvla/results/phase26_layer_claimC_summary.json`
- Report: `docs/phase26_layer_claimC_report_zh.md`
