# Phase 16 Compile Scope 连续动作漂移筛选

本轮目标承接 Phase 15：

```text
既然闭环回归来自 x/y/z/roll/pitch/yaw 的联合微漂移，
那下一步就先筛 compile scope 的 same-observation continuous drift，
再决定哪些 scope 值得跑闭环。
```

## 实验边界

本轮不是 closed-loop compiled benchmark，而是 same-observation replay：

```text
env 由 eager FP16 action 推进；
每个 live observation 上同时计算 eager action 和 compiled action；
两者使用相同 denoising seed；
只统计 compiled 相对 eager 的动作差异和耗时。
```

这样可以把 simulator 反馈隔离掉，先回答：

```text
这个 compile scope 在同一个 observation 上会不会把连续动作向量推歪？
```

配置：

| item | value |
|---|---|
| cases | `task4:init9`, `task6:init8` |
| max policy steps | 260 |
| seed base | 20260613 |
| compile mode | `reduce-overhead` |
| measured channels | `x/y/z/roll/pitch/yaw` |

说明：`task6:init8` 的 eager baseline 成功需要约 649 步，本轮只截取前 260 policy steps。因此表里的 `window done=1/2` 是正常的，不代表 task6 失败，只表示 260-step 窗口内还没完成。

## 结果总表

| target | valid steps | continuous max p99 | continuous max max | continuous L2 p99 | compiled p50 | eager p50 | p50 speed |
|---|---:|---:|---:|---:|---:|---:|---:|
| `action_head_dit_ff_all` | 484 | 0.002975 | 0.006409 | 0.003216 | 123.5 ms | 90.5 ms | 0.78x |
| `action_head_dit_attn_all` | 484 | 0.003090 | 0.006409 | 0.003223 | 102.9 ms | 92.3 ms | 0.90x |
| `action_head_model_blocks_6_15_eager` | 484 | 0.004120 | 0.054932 | 0.004223 | 93.9 ms | 102.3 ms | 1.09x |
| `action_head_model` | 484 | 0.004349 | 0.054932 | 0.004995 | 63.3 ms | 130.6 ms | 2.06x |
| `action_head_model_blocks_8_15_eager` | 484 | 0.004463 | 0.054932 | 0.004657 | 87.9 ms | 105.7 ms | 1.20x |
| `action_head_dit_blocks_0_7` | 0 | n/a | n/a | n/a | n/a | n/a | n/a |

`action_head_dit_blocks_0_7` 在首个 request 触发 CUDAGraph overwrite：

```text
accessing tensor output of CUDAGraphs that has been overwritten by a subsequent run
```

所以它不是低漂移，而是当前实现不可用。若要继续细粒度 block compile，需要给 online drift 脚本补 `torch.compiler.cudagraph_mark_step_begin()` 支持，或关闭相关 CUDAGraph 路径。

## 关键观察

### 1. 小 scope 漂移小，但没有速度收益

`action_head_dit_ff_all` 和 `action_head_dit_attn_all` 的 continuous drift 最小：

```text
continuous max p99 ~= 0.003
continuous max max ~= 0.0064
```

但它们的 p50 速度都比 eager 慢：

```text
ff_all:   123.5 ms vs eager 90.5 ms
attn_all: 102.9 ms vs eager 92.3 ms
```

这说明细粒度小图 compile 可以改善数值一致性，但吃不到大图融合收益。作为部署加速路线不成立。

### 2. 大图 compile 快，但仍有连续动作尖峰

`action_head_model` 的 p50 很好：

```text
compiled p50 = 63.3 ms
eager p50 = 130.6 ms
约 2.06x
```

但它在 task6 窗口内出现 continuous max spike：

```text
continuous max max = 0.054932
```

分维度看，尖峰主要在：

| target | x max | z max | pitch max |
|---|---:|---:|---:|
| `action_head_model` | 0.054932 | 0.042343 | 0.033631 |
| `action_head_model_blocks_8_15_eager` | 0.054932 | 0.039139 | 0.033631 |
| `action_head_model_blocks_6_15_eager` | 0.054932 | 0.040741 | 0.033924 |

这说明后半 block eager island 并没有消除我们现在关心的 task6 连续动作尖峰。

### 3. `blocks_8_15_eager` 不是最终可用边界

之前 `blocks_8_15_eager` 的优势是：

```text
速度仍较快；
某些闭环集合上能追平 baseline。
```

但 Phase 15/16 合起来说明：

```text
它仍保留了 task6:init8 这类连续动作尖峰；
continuous-action oracle 可以救回病例；
所以这个 compile boundary 还不是 behavior-transparent。
```

这解释了为什么它有速度潜力，但不能直接作为 rollout backend。

### 4. 保护更多后半 block 不单调改善

`blocks_6_15_eager` 比 `blocks_8_15_eager` 保护更多后半 block，但：

```text
continuous max max 一样是 0.054932；
p50 速度从 87.9 ms 变慢到 93.9 ms；
历史闭环里它也没有救回关键 regression。
```

所以继续盲目扩大 eager island 不是好路线。

## 当前判断

这轮把方向收得更窄了：

```text
小 scope：行为更稳，但无加速；
大 scope：有加速，但 task6 仍有连续动作尖峰；
后半 block eager island：没有去掉这个尖峰；
细粒度 block compile：当前会触发 CUDAGraph overwrite，工程上还不能直接用。
```

因此，单靠 torch.compile scope 搜索，短期内很难同时拿到：

```text
约 2x p50 加速
+ continuous action drift 足够小
+ 闭环成功率不回归
```

## 初步 Spike Forensic

对 trace 做 top-k spike 检索后，关键 outlier 非常集中：

```text
case: task6:init8
policy_step: 253
pre EEF pos: [0.179, 0.0364, 0.541]
```

在三个大图类 target 中，这个 step 都出现了同源尖峰：

| target | max dim | x diff | y diff | z diff | pitch diff |
|---|---|---:|---:|---:|---:|
| `action_head_model` | `x` | 0.054932 | -0.005922 | 0.042343 | 0.033631 |
| `action_head_model_blocks_8_15_eager` | `x` | 0.054932 | -0.006895 | 0.039139 | 0.033631 |
| `action_head_model_blocks_6_15_eager` | `x` | 0.054932 | -0.006437 | 0.040741 | 0.033924 |

而小 scope 没有这个尖峰：

| target | task6 max continuous diff | top spike step |
|---|---:|---:|
| `action_head_dit_attn_all` | 0.003319 | 63 |
| `action_head_dit_ff_all` | 0.002747 | 74/82/89 |

这个细节很关键：

```text
task6 的大尖峰不是随机散布的平均漂移；
它是一个特定 observation/step 上的多维联动 outlier。
```

并且 `blocks_8_15_eager`、`blocks_6_15_eager` 都没有消掉它，说明它不像是单纯来自后半 block 内部。更可能是：

1. 编译整块 `action_head.model` 时，前段路径或跨 block 残差路径仍产生了这个 outlier；
2. 或者 graph break/eager island 边界没有改变导致该 outlier 的数值路径；
3. 或者 denoising loop 的重复调用和 compiled graph 缓存/CUDAGraph 行为在该 observation 触发了离散化级别的差异。

## 下一步建议

我建议下一步分两条线：

1. **诊断线：定位 task6 spike 来源**
   - 对 `action_head_model` / `blocks_8_15_eager` 的 task6 trace 做 first spike analysis。
   - 找出 `0.054932` 出现在哪个 policy step、哪个维度、当时 observation/EEF 状态是什么。
   - 如果 spike 集中在少数 step，可以考虑局部 guard 或 fallback。

2. **工程线：不要继续盲目扩大 eager island**
   - 小 scope compile 已经证明速度收益不足。
   - 更保守 block island 也没有去掉 spike。
   - 更值得考虑的是数值守卫、特定高风险 step fallback、或转向 CUDA graph / kernel-level path，保持 eager 数值路径更接近。

短期最务实的下一步是：

```text
Phase 16.5: task6 step253 focused replay
目标：围绕 policy_step=253 的 observation 做重复 replay，
比较 eager / compiled 在相同 seed、相邻 seed、相邻 observation 上的稳定性。
```

如果 spike 是单个 observation 的 outlier，就有机会做轻量 guard 或 fallback；如果它在相邻 observation/seed 上稳定出现，那 `torch.compile` 大图路线要更谨慎，可能需要转向更底层但数值更可控的 kernel/CUDA graph 路线。
