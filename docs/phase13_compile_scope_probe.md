# Phase 13 torch.compile Scope Probe

本轮目标：

```text
在整块 action_head.model compile 有速度但有闭环回归之后，收缩 compile scope，定位 drift spike 的来源。
```

## 模块结构

真实 GR00T action head：

```text
policy.model.action_head: FlowmatchingActionHead
policy.model.action_head.model: DiT
DiT transformer blocks: transformer_blocks.0 ... transformer_blocks.15
每个 block:
  attn1: Attention
  ff: FeedForward
```

模块结构输出：

```text
toy_quantvla/results/phase13_action_head_modules_v1.json
```

新增 compile targets：

```text
action_head_dit_blocks_all
action_head_dit_blocks_0_7
action_head_dit_blocks_8_15
action_head_dit_attn_all
action_head_dit_attn_0_7
action_head_dit_attn_8_15
action_head_dit_ff_all
action_head_dit_ff_0_7
action_head_dit_ff_8_15
```

实现：

```text
toy_quantvla/phase13_compile_targets.py
toy_quantvla/timed_fp16_inference_service.py
toy_quantvla/phase13_torch_compile_online_drift.py
```

## Online Drift 结果

边界：

- eager FP16 policy 控制 LIBERO；
- compiled scope 只在同一个 live observation 上旁路算 action；
- eager 与 compiled 使用相同 denoising seed；
- 所以这里测的是同观测 action drift，不是 compiled closed-loop 成功率。

### 两个关键 case

| scope | case | eager success | steps | raw rel RMSE mean | raw rel RMSE max | raw max abs diff | LIBERO RMSE mean | compiled mean |
|---|---|---|---:|---:|---:|---:|---:|---:|
| whole action_head.model | 4:6 | success | 245 | 0.003573 | 0.045184 | 0.142975 | 0.000697 | 0.0902s |
| whole action_head.model | 6:0 | success | 210 | 0.003889 | 0.011721 | 0.011353 | 0.000645 | 0.0929s |
| DiT FFN all | 4:6 | success | 245 | 0.001988 | 0.032053 | 0.107574 | 0.000319 | 0.1710s |
| DiT FFN all | 6:0 | success | 210 | 0.002067 | 0.006646 | 0.004639 | 0.000358 | 0.1648s |
| DiT attention all | 4:6 | success | 245 | 0.001873 | 0.007292 | 0.021942 | 0.000358 | 0.1317s |
| DiT attention all | 6:0 | success | 210 | 0.002064 | 0.007102 | 0.005493 | 0.000336 | 0.1288s |

解读：

- 整块 `action_head.model` 最快，但 drift spike 最大。
- `DiT attention all` 明显压低了 `4:6` 的尖峰：raw max 从 `0.142975` 降到 `0.021942`。
- `DiT FFN all` 的均值 drift 也比整块小，但仍在 `4:6` 出现 `0.107574` 的大尖峰。
- 小 scope 的 online compiled mean 变慢，说明细粒度 compile 会损失整块 DiT 的融合收益。

### FFN 前后半定位

只跑尖峰最明显的 `task4:init6`：

| scope | raw rel RMSE mean | raw rel RMSE max | raw max abs diff | LIBERO RMSE mean | compiled mean |
|---|---:|---:|---:|---:|---:|
| DiT FFN all | 0.001988 | 0.032053 | 0.107574 | 0.000319 | 0.1710s |
| DiT FFN 0-7 | 0.001896 | 0.006738 | 0.018890 | 0.000333 | 0.1537s |
| DiT FFN 8-15 | 0.001987 | 0.034670 | 0.115875 | 0.000345 | 0.1437s |

关键结论：

```text
task4:init6 的大尖峰主要来自后半 DiT FFN，也就是 transformer_blocks.8..15.ff。
```

这说明整块 DiT compile 的风险不是平均误差，而是后半 FFN 在某些接触状态附近产生局部 action spike。

## Closed-Loop 速度/成功率

对 `action_head_dit_attn_all` 跑了 compiled-only 闭环两例：

```text
tag: phase13_compile_scope_attn_all_2case_v1
cases: 4:6,6:0
target: action_head_dit_attn_all
```

| policy | success | calls | client p50 | server p50 | server p90 |
|---|---:|---:|---:|---:|---:|
| attention-only compile | 1/2 | 1232 | 158.1 ms | 153.2 ms | 162.4 ms |

逐 case：

| case | success | calls | client p50 |
|---|---|---:|---:|
| 4:6 | success | 241 | 158.1 ms |
| 6:0 | fail | 991 | 158.1 ms |

对比既有结果：

| policy | success | server p50 | 结论 |
|---|---:|---:|---|
| FP16 eager baseline | 7/15 | 155.3 ms | 稳定参考 |
| whole action_head.model compile | 5/15 | 70.1 ms | 有明显速度，但有闭环回归 |
| attention-only compile | 1/2 | 153.2 ms | 漂移更小，但几乎没有速度收益，且仍会失败 |

## 当前判断

1. `torch.compile` 的速度收益需要足够大的连续图。
   只 compile attention 或 FFN 子模块，会产生很多小图调用，基本吃不到整块 DiT compile 的 p50 加速。

2. 行为风险集中在局部尖峰。
   后半 FFN (`transformer_blocks.8..15.ff`) 对 `4:6` 的大尖峰贡献最大。

3. attention-only 不是好工程路线。
   它降低了同观测 drift spike，但闭环 `6:0` 仍失败，且 server p50 仍是 `153 ms` 量级。

4. 下一步更有价值的方向不是继续小 scope compile，而是做“整块 compile + 局部保护”：

```text
compile whole action_head.model
but keep transformer_blocks.8..15.ff on eager path
```

工程上可以尝试：

- 对 `transformer_blocks.8..15.ff.forward` 加 `torch._dynamo.disable`，让整块 DiT compile 在风险 FFN 周围 graph break；
- 或把 DiT forward 手工拆成两段：大段 compile，后半 FFN eager；
- 或退一步做 CUDA graph 捕获 eager kernel 序列，优先追求数值路径不变。

## 本轮结论

这轮 scope probe 的收益很明确：

```text
整块 DiT compile 给速度。
后半 FFN compile 给尖峰风险。
attention-only 没有足够速度收益。
下一步应尝试 whole-model compile with risky-FFN eager island。
```
