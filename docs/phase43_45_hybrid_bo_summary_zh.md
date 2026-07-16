# Phase43-45 总结：hybrid CLSG-BO 的收益和边界

## 一句话结论

Phase43-45 把 BO 从单纯 duration-window search 扩展到：

```text
duration window x layer eager island
```

结果显示：layer 维度确实会影响闭环行为，但它没有给出一个稳定碾压 duration-only 的 tactic。最有价值的候选是：

```text
blocks0_3_window_2_12
```

它在 Phase45 all-task stress 上达到 `18/20`，相对 FP16 是 `0` repair / `1` regression，p50 speedup `1.06x`；相比 `window_2_12` 的 `17/20`、`0` repair / `2` regressions，它修复了一个额外失败。但它在 Phase44 上没有明显优于 `window_2_12`，而且速度收益不大。

所以当前判断是：

> Hybrid CLSG-BO 有价值，但不是免费午餐。早期 block island 可以改变 repair/regression profile；mid/late block island 明显更危险；最终仍必须靠 held-out validation 选 Pareto 点。

## Phase43 probe：10 cases

```text
tasks 0/1/4/6/8 x init 33/34
```

| tactic | success | p50 ms | speedup | paired vs FP16 |
| --- | ---: | ---: | ---: | --- |
| `fp16` | `9/10` | `87.32` | `-` | baseline |
| `speed_only` | `9/10` | `77.29` | `1.13x` | `0` repair / `0` regression |
| `window_0_20` | `8/10` | `85.77` | `1.02x` | `1` repair / `2` regressions |
| `window_2_12` | `9/10` | `83.85` | `1.04x` | `1` repair / `1` regression |
| `blocks0_3` | `9/10` | `68.97` | `1.27x` | `1` repair / `1` regression |
| `blocks0_3_window_0_20` | `9/10` | `90.35` | `0.97x` | `0` repair / `0` regression |
| `blocks0_3_window_2_12` | `9/10` | `80.41` | `1.09x` | `1` repair / `1` regression |
| `blocks8_15_window_2_12` | `7/10` | `82.88` | `1.05x` | `1` repair / `3` regressions |
| `blocks16_31_window_2_12` | `7/10` | `82.04` | `1.06x` | `1` repair / `3` regressions |

Phase43 的主要信息：

- `blocks0_3` 有速度潜力，`1.27x`，但不是零风险。
- `blocks0_3_window_0_20` 行为最保守，但几乎没有加速。
- `blocks8_15_window_2_12` 和 `blocks16_31_window_2_12` 明显更危险，说明 layer island 不是“随便保护一段都会好”。

## Phase44 held-out：15 cases

```text
tasks 0/1/4/6/8 x init 35/36/37
```

| tactic | success | p50 ms | speedup | paired vs FP16 |
| --- | ---: | ---: | ---: | --- |
| `fp16` | `11/15` | `88.78` | `-` | baseline |
| `speed_only` | `10/15` | `82.14` | `1.08x` | `2` repairs / `3` regressions |
| `window_0_20` | `9/15` | `88.36` | `1.00x` | `2` repairs / `4` regressions |
| `window_2_12` | `13/15` | `81.88` | `1.08x` | `3` repairs / `1` regression |
| `blocks0_3_window_0_20` | `9/15` | `85.13` | `1.04x` | `1` repair / `3` regressions |
| `blocks0_3` | `12/15` | `71.57` | `1.24x` | `3` repairs / `2` regressions |
| `blocks0_3_window_2_12` | `13/15` | `86.20` | `1.03x` | `3` repairs / `1` regression |

Phase44 的主要信息：

- `window_2_12` 和 `blocks0_3_window_2_12` 行为并列最好：都是 `13/15`、`3` repairs / `1` regression。
- 但 `window_2_12` 更快：`1.08x` vs `1.03x`。
- `blocks0_3` 速度最好，`1.24x`，但 regression 更多。

因此 Phase44 不支持“hybrid 明显优于 duration-only”，但支持“早期 block island 会改变 repair/regression profile”。

## Phase45 all-task stress：20 cases

```text
tasks 0-9 x init 38/39
```

| tactic | success | p50 ms | speedup | paired vs FP16 |
| --- | ---: | ---: | ---: | --- |
| `fp16` | `19/20` | `89.22` | `-` | baseline |
| `speed_only` | `17/20` | `82.42` | `1.08x` | `0` repair / `2` regressions |
| `window_0_20` | `18/20` | `90.57` | `0.99x` | `0` repair / `1` regression |
| `window_2_12` | `17/20` | `85.99` | `1.04x` | `0` repair / `2` regressions |
| `blocks0_3_window_2_12` | `18/20` | `84.48` | `1.06x` | `0` repair / `1` regression |
| `blocks0_3` | `16/20` | `72.19` | `1.24x` | `0` repair / `3` regressions |

Phase45 的主要信息：

- `blocks0_3_window_2_12` 是这一轮最好的 speed-risk point：比 `window_0_20` 更快，同时 regression 数相同；比 `window_2_12` 成功率更高、regression 更少。
- `blocks0_3` 仍然是速度点，但 closed-loop risk 太高。
- `speed_only` 在 all-task stress 上不稳，两个 regressions 都来自 task 8。

## 对论文的意义

这组三阶段结果非常适合支撑两个论点：

1. **Not all layers are equal.**  
   早期 block island、mid block island、late block island 的 repair/regression profile 不同。mid/late 组合在 Phase43 直接退化到 `7/10`，说明 layer 维度是闭环敏感的。

2. **CLSG-BO 是 candidate generator，不是 oracle.**  
   Phase43 probe 上看起来保守的 `blocks0_3_window_0_20` 到 Phase44 并没有保持优势；Phase44 看起来和 `window_2_12` 打平的 `blocks0_3_window_2_12` 到 Phase45 又表现更好。最终选择必须靠 held-out / stress validation。

## 当前可写进论文的结论

当前最稳妥的写法不是“hybrid tactic 最优”，而是：

> Extending CLSG-BO from duration-only windows to hybrid layer-duration tactics reveals additional structure in the speed-risk frontier. Early-block protection can change paired repair/regression outcomes, while mid/late block combinations can be harmful. However, the hybrid search does not eliminate closed-loop risk; it produces candidate Pareto points that still require held-out validation.

更具体地说：

- behavior-first：`window_0_20` 或 `blocks0_3_window_2_12`，取决于 validation slice。
- speed-constrained：`blocks0_3` 有速度但 regression 多，不适合作为默认安全 tactic。
- 当前最值得继续验证：`blocks0_3_window_2_12`，因为它在 all-task stress 上比 `window_2_12` 更稳。

## 下一步建议

不建议继续无限扩展 BO 搜索空间。下一步更有价值的是：

1. 把 Phase43-45 写入论文，作为 hybrid search 的边界验证。
2. 若还要跑实验，只跑一个小的 confirmation fold：

```text
fp16
window_0_20
window_2_12
blocks0_3_window_2_12
```

在新的 all-task init 上验证 `blocks0_3_window_2_12` 是否稳定保持低 regression。
