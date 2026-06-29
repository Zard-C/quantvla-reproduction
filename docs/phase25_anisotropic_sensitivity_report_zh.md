# Phase 25: 闭环敏感性的三重各向异性

本阶段验证三个判断：

- not all dims are equal
- not all durations / phases are equal
- not all layers are equal

实验使用 Phase24 的无损 CUDA Graph FP16 service 作为 baseline。这个 baseline 在 Phase24 已验证与 FP16 hygiene success map 一致，因此这里可以把 rollout 变化归因到显式注入的 action perturbation 或已有 layer/scope 变体，而不是推理加速路径本身。

## 1. Baseline

| case | outcome | steps |
| --- | --- | --- |
| 4:9 | success | 224 |
| 6:8 | success | 649 |

两个重点病例在无扰动 baseline 下都是成功：`4:9` 为 224 步，`6:8` 为 649 步。

## 2. Calibration: continuous 6D 全程扰动

设置：`keys=continuous`，`amplitude=0.03`，对 6 个连续 action 维度按 `sqrt(6)` 归一化，因此每步扰动 L2 约为 0.03。

| case | outcome | steps |
| --- | --- | --- |
| 4:9 | fail | 991 |
| 6:8 | fail | 991 |

这个扰动预算会把两个 baseline 成功病例都推成失败，说明幅度足以触发闭环分叉。后续单维实验使用同样 L2 预算，比较不同维度的相对敏感性。

## 3. Claim A: not all dims are equal

设置：固定两个 case、固定 deterministic policy seed、固定每步扰动 L2 为 0.03，只改变扰动注入的 action dimension。

| dimension | success | per-case outcome / steps |
| --- | --- | --- |
| x | 1/2 | 4:9=success/349, 6:8=fail/991 |
| y | 0/2 | 4:9=fail/991, 6:8=fail/991 |
| z | 2/2 | 4:9=success/239, 6:8=success/666 |
| roll | 0/2 | 4:9=fail/991, 6:8=fail/991 |
| pitch | 0/2 | 4:9=fail/991, 6:8=fail/991 |
| yaw | 0/2 | 4:9=fail/991, 6:8=fail/991 |

结论非常清楚：同样的 L2 扰动预算下，`z` 维度保持 `2/2` 成功，`x` 为 `1/2`，而 `y/roll/pitch/yaw` 全部 `0/2`。这说明闭环敏感性不是 action-space isotropic 的，不能只看整体 action MSE 或 L2 drift。

更有意思的是，`z` 并不是完全无影响：`6:8` 从 baseline 649 步变成 666 步；但它没有跨过失败边界。`y/roll/pitch/yaw` 则更容易把轨迹推过成功/失败 margin。

## 4. Claim B: not all durations / phases are equal

设置：固定扰动维度为 `y`，固定 amplitude 为 0.03，只改变注入时间窗口。窗口按 policy step 计数。

| case | phase | window | outcome | steps |
| --- | --- | --- | --- | --- |
| task4:init9 | early | 0:75 | fail | 991 |
| task4:init9 | mid | 75:150 | fail | 991 |
| task4:init9 | late | 150:225 | success | 219 |
| task6:init8 | early | 0:200 | fail | 991 |
| task6:init8 | mid | 200:450 | success | 697 |
| task6:init8 | late | 450:700 | success | 752 |

task4:init9 中，同样的 y 扰动打在 early 或 mid 会失败，但打在 late 仍成功。这说明任务后段对这个方向的扰动更可吸收，或者该扰动已经不能再把轨迹推过关键失败边界。

task6:init8 中，early 失败，但 mid 和 late 都成功，只是步数从 baseline 649 变为 697/752。这里同样说明闭环系统对误差的敏感性随阶段变化，不是简单按扰动累计时间决定。

## 5. Claim C: not all layers are equal

本轮没有重新跑 layer-level intervention；这里先引用 Phase17 的 module/layer sensitivity map 作为已有证据。它来自 compile scope、step253 focused replay、block-level bisect 和 matched rollout artifacts。

| scope | sensitivity | max drift | p50 speedup | reason |
| --- | --- | --- | --- | --- |
| action_head_model_blocks_6_15_eager | high | 0.054932 | 1.09 | contains deterministic continuous-action spike |
| action_head_model | high | 0.054932 | 2.06 | contains deterministic continuous-action spike |
| action_head_model_blocks_8_15_eager | high | 0.054932 | 1.2 | contains deterministic continuous-action spike |
| action_head_dit_ff_all | low-drift-slow | 0.006409 | 0.73 | low same-observation drift but no useful speedup |
| action_head_dit_attn_all | low-drift-slow | 0.006409 | 0.9 | low same-observation drift but no useful speedup |

| block/scope | risk | max abs | spike removed | p50 speedup |
| --- | --- | --- | --- | --- |
| block0 | low for step253 | 0.001831 | True | 2.11 |
| block1 | high for step253 | 0.054474 | False | 1.9 |
| block2 | high for step253 | 0.054932 | False | 1.97 |
| block3 | high for step253 | 0.055389 | False | 2.13 |
| blocks2-3 | high for step253 | 0.054932 | False | 1.84 |
| blocks4-7 | high for step253 | 0.054932 | False | 1.68 |
| blocks8-15 | high for step253 | 0.054932 | False | 1.4 |

Phase17 的关键发现是：`action_head_model` 和 `blocks_8_15_eager` 等 scope 包含 deterministic continuous-action spike，而 `ff_all` / `attn_all` 这类 scope 同观测 drift 很低但没有实际速度收益。更细的 step253 bisect 显示，保留 `block0` eager 就能消除该 spike，而单独保留 block1/2/3 或 blocks4-7 不能消除。这已经支持 layer/scope sensitivity 不均匀。

不过，Claim C 目前仍比 A/B 弱一些：A/B 是本阶段新跑的 controlled intervention；C 主要依赖已有 compile/layer artifacts。下一步如果要把 Claim C 写成强实验，需要重新做 matched-norm layer perturbation 或 layer-wise fake quant sweep，并用 Phase24 baseline service 加速闭环评估。

## 6. 统一解释

三个结果可以统一写成：

```text
Δh ≈ Σ_t c_t^T η_t
```

其中 `η_t` 的来源可以是 action-level 注入、量化误差，或者某层扰动经过 policy Jacobian 后映射到 action space 的结果。Phase25 的实验证明：

- `c_t` 在 action dimension 上不均匀：同样 L2 的 z/y/roll 扰动闭环结果不同。
- `c_t` 随 rollout phase 改变：同样 y 扰动在 early/mid/late 的结果不同。
- layer perturbation 通过 `J_{t,l}` 投影到 action-sensitive directions，因此不同 layer/scope 风险不同。

所以真正需要优化的不是全局 action MSE，而是 sensitivity-weighted closed-loop perturbation。

## 7. 当前结论

Phase25 已经给出足够强的初步证据来支撑：闭环量化风险在 action dimension、task phase、model layer 上都是各向异性的。

最强新证据：

- 同样 L2=0.03 的单维扰动，`z=2/2` 成功，`y/roll/pitch/yaw=0/2`。
- 同样 `y+0.03`，task4 early/mid 失败但 late 成功；task6 early 失败但 mid/late 成功。

这为后续 optimize guide 提供了方向：保护预算应优先分配给闭环敏感 action directions、敏感 task phases，以及能投影到这些方向的敏感 layer/scope。

## 8. 下一步

1. 对 action dim 做双符号和多幅度 sweep，例如 `±0.01/0.02/0.03`，区分方向性和维度性。
2. 对 phase windows 做更细分的 sliding-window sweep，估计每个 step 区间的 empirical sensitivity curve。
3. 做 layer-wise matched-norm perturbation 或 fake-quant sweep，把 Claim C 从已有证据升级成 controlled intervention。
4. 将 empirical sensitivity map 转成量化策略：敏感维度/阶段/层保留更高精度或使用闭环校正。
