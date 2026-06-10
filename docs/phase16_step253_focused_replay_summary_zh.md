# Phase 16.5 Step253 聚焦复验总结

本阶段目标很窄：

```text
围绕 Phase 16 里发现的 task6:init8 / policy_step=253 连续动作尖峰，
验证它到底是随机噪声、seed 敏感、还是某个 observation 上稳定复现的 graph-level outlier。
```

## 实验边界

这不是 closed-loop rollout，也不是成功率 benchmark。

本轮采用 eager-controlled same-observation replay：

```text
1. simulator 始终由 eager FP16 action 推进；
2. 到达 policy step 248-258 后，冻结同一个 live observation；
3. 在该 observation 上分别跑 eager path 和 compiled path；
4. 两者使用匹配 denoising seed；
5. 每个 step 测 seed offsets = [-2, -1, 0, 1, 2]；
6. 每个 seed 重复 3 次。
```

所以每个 focused step 有：

```text
5 seed offsets * 3 repeats = 15 个 same-observation 样本
```

本轮只看连续动作通道：

```text
x / y / z / roll / pitch / yaw
```

gripper 仍被记录，但它不是本轮判断主轴。

## 配置

| item | value |
|---|---|
| case | `task6:init8` |
| task | `put the white mug on the plate and put the chocolate pudding to the right of the plate` |
| focused policy steps | `248-258` |
| base seed | `20260613` |
| denoising steps | `8` |
| repeats per seed | `3` |
| compile mode | `reduce-overhead` |

测试的 target：

| target | 作用 |
|---|---|
| `action_head_model` | 大图 compile，速度最好，但 Phase 16 发现有 spike |
| `action_head_model_blocks_8_15_eager` | 后半 DiT block 留 eager，之前的候选折中方案 |
| `action_head_dit_attn_all` | 小 scope control，漂移小但速度收益弱 |

## 总结果

| target | top step | top dim | continuous max | repeat span max | compiled p50 mean | eager p50 mean |
|---|---:|---|---:|---:|---:|---:|
| `action_head_model` | 253 | `x` | 0.0549316 | 0 | 62.4 ms | 142.9 ms |
| `action_head_model_blocks_8_15_eager` | 253 | `x` | 0.0549316 | 0 | 83.5 ms | 117.0 ms |
| `action_head_dit_attn_all` | 258 | `x` | 0.0019598 | 0 | 103.0 ms | 107.5 ms |

`repeat span max = 0` 的含义是：

```text
同一个 observation、同一个 denoising seed 下重复调用 3 次，
compiled - eager 的连续动作差异完全一致。
```

这说明 spike 不是随机 runtime 抖动，也不像 CUDA 异步测量噪声；它是一个稳定可复现的数值路径差异。

## Step253 关键样本

### `action_head_model`

在 `policy_step=253`：

| dim | compiled - eager |
|---|---:|
| x | 0.0549316 |
| y | -0.0059223 |
| z | 0.0423431 |
| roll | -0.0034051 |
| pitch | 0.0336312 |
| yaw | -0.0006345 |
| gripper | -0.0019531 |

continuous L2 p99：

```text
0.0773858
```

### `action_head_model_blocks_8_15_eager`

在同一个 `policy_step=253`：

| dim | compiled - eager |
|---|---:|
| x | 0.0549316 |
| y | -0.0068951 |
| z | 0.0391388 |
| roll | -0.0028812 |
| pitch | 0.0336312 |
| yaw | -0.0004532 |
| gripper | -0.0014648 |

continuous L2 p99：

```text
0.0757393
```

这说明 `blocks_8_15_eager` 并没有切断这个 outlier。它和大图 compile 在同一步、同一主维度、相近多维方向上出现尖峰。

### `action_head_dit_attn_all`

同一个窗口下，`attn_all` 没有出现大尖峰。

它在 `policy_step=253` 的 continuous max 只有：

```text
0.00183105
```

全窗口最大值也只有：

```text
0.0019598
```

这说明 step253 不是一个“任何 compile scope 都会爆”的自然病态 observation。更像是：

```text
大图 action_head.model compile 路径
或跨 block / denoising loop 的图级数值路径
在该 observation 上触发了离散级 outlier。
```

## 关键结论

### 1. Spike 是 observation-local，不是慢性漂移

`action_head_model` 在附近 step 的 continuous max：

| step | continuous max |
|---:|---:|
| 248 | 0.0027466 |
| 249 | 0.0020885 |
| 250 | 0.0013733 |
| 251 | 0.0013733 |
| 252 | 0.0016022 |
| 253 | 0.0549316 |
| 254 | 0.0013733 |
| 255 | 0.0009155 |
| 256 | 0.0013733 |
| 257 | 0.0027466 |
| 258 | 0.0042486 |

这不是误差逐步积累，而是在一个局部 observation 上突然跳变。

### 2. Spike 是 deterministic，不是 runtime randomness

所有 target 的 fixed-seed repeat span 都是 0。

因此，后续可以做确定性定位：

```text
同一 observation + 同一 seed + 同一 target
可以稳定复现同一个 action diff。
```

这对工程排查很重要，因为我们可以做 module-level bisect、逐层 hook、局部 fallback，而不是反复跑 rollout 赌概率。

### 3. 后半 block eager island 不够

`blocks_8_15_eager` 保留了后半 DiT block 为 eager，但 step253 spike 仍然存在。

这排除了一个简单解释：

```text
不是只要把后半 block 留 eager 就能解决回归。
```

更可能的问题在：

```text
前半 block、跨 block residual / normalization、
denoising loop 多次调用的 graph/CUDAGraph 行为、
或 action_head.model 外层大图融合后的数值路径。
```

### 4. 小 scope 更透明，但加速不成立

`action_head_dit_attn_all` 的行为透明性更好，但速度几乎没有收益：

```text
compiled p50 mean = 103.0 ms
eager p50 mean    = 107.5 ms
```

这符合 Phase 16 的大判断：

```text
小 scope：更稳，但吃不到大图加速；
大 scope：加速明显，但有行为 outlier。
```

## 工程含义

这轮之后，`torch.compile` 路线的风险不再是泛泛地说“可能精度变”，而是更具体：

```text
少数 contact / near-contact observation 上，
大图 compiled action_head 可能产生确定性多维连续动作 outlier。
```

这种 outlier 对机器人闭环很危险，因为一次 `x/z/pitch` 联动偏移就可能把末端执行器带出可恢复区域。它不需要每步都漂，只要在关键接触阶段出现一次，就可能改变后续轨迹分布。

## 下一步建议

### A. 图内定位

在同一个 `task6:init8 / step253` observation 上做 module-level bisect：

```text
1. 保存 processed_obs 或在同一路径中重放到 step253；
2. 对 action_head.model 内部关键边界加 hook；
3. 比较 eager vs compiled 的 hidden/residual diff；
4. 找到第一个 diff 从 1e-3 级跳到 1e-2 / 1e-1 级的边界。
```

优先候选：

```text
transformer_blocks 0-7
block 间 residual add / norm
denoising loop 的 timestep / noise path
action unnormalization 前后的 head output
```

### B. 局部 fallback / guard

如果 spike 集中在少数 observation 类型，可以考虑工程 guard：

```text
compiled 先出 action；
如果连续动作相对上一帧或内部 margin 超阈值，则 fallback eager；
```

这条路线的前提是 guard 能在没有 eager teacher 的部署场景下触发，因此不能直接用 compiled-vs-eager diff，需要设计 runtime proxy，例如：

```text
action delta norm
EEF-state-conditioned action bound
denoising trajectory internal variance
```

### C. 不要急着扩大 closed-loop rollout

当前已经证明：

```text
大图 compile 的速度收益是真实的；
但 behavior-transparent 还没过关。
```

建议先把 step253 的图内来源定位清楚，再做更大 rollout。否则大规模成功率会把“单点 outlier 的机制”淹没在统计波动里。

## 产物

原始结果：

| file | 内容 |
|---|---|
| `toy_quantvla/results/phase16_step253_focused_replay_v1_summary.json` | 跨 target 汇总 |
| `toy_quantvla/results/phase16_step253_focused_replay_v1_action_head_model.json` | 大图 target 详细结果 |
| `toy_quantvla/results/phase16_step253_focused_replay_v1_action_head_model_blocks_8_15_eager.json` | 后半 block eager target 详细结果 |
| `toy_quantvla/results/phase16_step253_focused_replay_v1_action_head_dit_attn_all.json` | attention 小 scope control 详细结果 |
| `toy_quantvla/results/phase16_step253_focused_replay_v1_*_trace/` | 每个 target 的 per-step trace |

脚本：

| file | 内容 |
|---|---|
| `toy_quantvla/phase16_step_focused_replay.py` | 单 target 聚焦 replay |
| `toy_quantvla/phase16_step_focused_summary.py` | 多 target 汇总 |
| `toy_quantvla/run_phase16_step253_focused_replay.sh` | 5090 一键运行脚本 |
