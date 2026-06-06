# Phase 7 Online Drift Report

## 1. 这一步在测什么

这一步不是量化闭环成功率测试。

我们测的是：

```text
teacher-controlled online drift
```

具体做法：

```text
1. FP16 teacher 控制 LIBERO 环境前进。
2. 每个 live observation 同时喂给 teacher 和 FP4-like student。
3. teacher/student 使用同一个 denoising seed。
4. 环境只执行 teacher action。
5. 记录同观测下的 action drift。
```

这样可以先回答一个更干净的问题：

```text
Blackwell FP4-like 数值扰动在真实 simulator observation 上，
会不会直接把 action 输出打爆？
```

它暂时不回答：

```text
student 自己闭环控制时最终 success rate 是多少？
```

## 2. 新增脚本

```text
toy_quantvla/phase7_online_drift.py
```

核心设计：

```text
single model, dual mode
```

模型只加载一次。目标 Linear 被替换成带开关的 wrapper：

```text
state.enabled = False
  跑 FP16 teacher

state.enabled = True
  跑 FP4-like student
```

这样避免加载两份 GR00T 导致显存翻倍，也避免两个模型之间状态不一致。

## 3. FP4-like 数值格式

当前不是 bit-exact CUTLASS kernel，而是 PyTorch fake quant：

```text
quant format: fp4_e2m1_blockscaled_fake
block size: 32
codebook: approximate Float4E2M1 finite values
scale: per-block absmax / 6
```

对 activation：

```text
沿最后一维按 32 分块。
```

对 weight：

```text
沿 input channel / K 维按 32 分块。
```

这一步的意义是验证数值扰动，不验证 kernel performance。

## 4. 已完成的小规模探针

### 4.1 `dit_mlp_only`, 4 cases, 40 steps each

结果文件：

```text
toy_quantvla/results/phase7_online_drift_dit_mlp_40.json
docs/phase7_online_drift_dit_mlp_40.md
```

| task | init | steps | raw rel RMSE mean | raw cosine mean | LIBERO action RMSE mean | max action diff |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 3 | 40 | 0.00764 | 0.999973 | 0.00223 | 0.01282 |
| 4 | 10 | 40 | 0.00748 | 0.999973 | 0.00243 | 0.01282 |
| 8 | 7 | 40 | 0.00621 | 0.999981 | 0.00191 | 0.00847 |
| 8 | 9 | 40 | 0.00554 | 0.999985 | 0.00181 | 0.00916 |

Patch:

```text
patched modules: 32
family: dit_mlp
module input relative RMSE mean: 0.1027
module weight relative RMSE mean: 0.1012
```

Interpretation:

```text
DiT MLP 的 FP4-like 扰动在 live simulator observation 上比较稳。
虽然单层输入/权重 FP4 fake quant 约有 10% relative RMSE，
最终 action chunk 的 relative RMSE 只有约 0.55%-0.76%。
```

### 4.2 `llm_mlp_only`, 2 cases, 20 steps each

结果文件：

```text
toy_quantvla/results/phase7_online_drift_llm_mlp_20.json
docs/phase7_online_drift_llm_mlp_20.md
```

| task | init | steps | raw rel RMSE mean | raw cosine mean | LIBERO action RMSE mean | max action diff |
|---:|---:|---:|---:|---:|---:|---:|
| 8 | 7 | 20 | 0.01241 | 0.999910 | 0.00526 | 0.02380 |
| 8 | 9 | 20 | 0.01140 | 0.999899 | 0.00397 | 0.02609 |

Patch:

```text
patched modules: 36
family: llm_mlp
module input relative RMSE mean: 0.1041
module weight relative RMSE mean: 0.1011
```

Interpretation:

```text
LLM MLP 比 DiT MLP 漂移更大，但仍在同观测 action 层面保持较高 cosine。
这是性能收益最大的区域之一，值得继续扩大验证。
```

### 4.3 `llm_dit_mlp`, 2 cases, 10 steps each

结果文件：

```text
toy_quantvla/results/phase7_online_drift_llm_dit_10.json
docs/phase7_online_drift_llm_dit_10.md
```

| task | init | steps | raw rel RMSE mean | raw cosine mean | LIBERO action RMSE mean | max action diff |
|---:|---:|---:|---:|---:|---:|---:|
| 8 | 7 | 10 | 0.02455 | 0.999688 | 0.00586 | 0.02838 |
| 8 | 9 | 10 | 0.02747 | 0.999562 | 0.00855 | 0.02975 |

Patch:

```text
patched modules: 116
families:
  llm_attn: 48
  llm_mlp: 36
  dit_mlp: 32
```

Interpretation:

```text
full llm_dit_mlp 明显比单独 DiT/LLM MLP scope 更激进。
但在 task8 的前 10 个 live observations 上，没有出现 action 直接崩盘。
```

## 5. 当前判断

### 5.1 好消息

`dit_mlp_only` 的 online same-observation drift 很小：

```text
raw relative RMSE mean: 0.0055 - 0.0076
raw cosine mean: > 0.99997
LIBERO action RMSE mean: 0.0018 - 0.0024
```

这说明 Blackwell FP4-like activation + weight fake quant，不是天然不可用。

### 5.2 需要谨慎的地方

`llm_dit_mlp` 的漂移明显更大：

```text
raw relative RMSE mean: 0.0246 - 0.0275
raw cosine mean: 0.99956 - 0.99969
LIBERO action RMSE mean: 0.0059 - 0.0086
```

这和 Phase 6 的 scope contraction 结论一致：

```text
scope 越大，offline / online drift 都会变大。
```

但这次没有直接否定 full scope，因为：

```text
1. 这只是 task8 前 10 步。
2. online observation 比纯 synthetic observation 更真实。
3. drift 变大但没有立刻数值崩坏。
```

## 6. 和 CUTLASS 性能结果的关系

Phase 7 CUTLASS SM120 benchmark 说明：

```text
Blackwell GeForce 原生 blockscaled FP4 有真实性能收益。
```

这次 online drift 说明：

```text
FP4-like 数值扰动在真实 observation 上至少可以被测量，
并且小 scope 下看起来可控。
```

两者合在一起，下一步路线变得更清楚：

```text
先扩大 online drift，
再决定是否进入 quantized student closed-loop rollout，
最后才做 CUTLASS PyTorch extension。
```

## 7. 下一步建议

### 7.1 扩大 online drift

建议优先跑：

```text
scope:
  dit_mlp_only
  llm_mlp_only
  llm_mlp_dit_mlp

cases:
  8:7, 8:9
  4:10, 4:11
  0:3, 0:7
  6:1, 9:9

max_policy_steps:
  80 或 120
```

暂时不建议直接大跑 full `llm_dit_mlp`，除非我们只是做短窗口诊断。

### 7.2 做 paired closed-loop 小样本

如果扩大 online drift 后 `dit_mlp_only` 或 `llm_mlp_only` 仍然稳定，可以做第二层实验：

```text
same task/init
teacher closed-loop
student closed-loop
paired trace comparison
```

这里才开始看：

```text
repair / regression
trajectory bifurcation
success / failure flip
```

### 7.3 暂不接 CUTLASS extension

当前还不该直接写 PyTorch extension 接 CUTLASS。

原因：

```text
性能后端已经有希望，
但数值格式还没有通过足够 online/closed-loop 行为验证。
```

更稳的 gate 是：

```text
1. online same-observation drift 扩大样本通过
2. paired closed-loop smoke 通过
3. 再开始 CUTLASS extension
```
