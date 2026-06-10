# Phase 16.6 Step253 Compile Boundary Bisect 总结

本阶段目标是解释 `task6:init8` 在 `policy_step=253` 附近出现的 `torch.compile` 动作尖峰：它到底来自整个 `action_head.model` 大图，还是来自 DiT 内部某个更小的编译边界。

结论很清楚：尖峰可以被进一步定位到 **DiT transformer block 0 的 compiled path**。只把 block0 保持 eager，其他部分继续走 `action_head.model` 大图编译，就能把 `0.055` 量级的动作差异压回 `0.002` 量级。

## 实验设置

- Case：`task6:init8`
- 聚焦步：`policy_step=253`
- 方法：同一 observation 下做 eager-controlled replay，对比 compiled 输出和 eager 输出
- 随机扰动：`seed_offsets=-2,0,2`
- 每个 seed 重复：`repeats=2`
- 预热：`warmup_repeats=1`
- 编译配置：`torch.compile(mode="reduce-overhead", backend="inductor")`

这里的 `same_seed_repeat_span_max=0` 很关键：说明尖峰不是重复调用时的随机噪声，而是在固定 seed 和固定 observation 下稳定复现的数值路径差异。

## 第一轮：从大范围缩到 block0-3

| target | eager island | cont max | L2 p99 | compiled p50 | eager p50 | 读法 |
|---|---:|---:|---:|---:|---:|---|
| `action_head_model` | 无 | 0.0549316 | 0.0773858 | 54.2 ms | 130.4 ms | 原始大图编译尖峰存在 |
| `action_head_model_blocks_0_15_eager` | block0-15 | 0.000679 | 0.000707 | 101.1 ms | 108.2 ms | 全部 block eager，尖峰消失 |
| `action_head_model_blocks_0_7_eager` | block0-7 | 0.001388 | 0.001826 | 85.2 ms | 117.6 ms | 前半段 eager，尖峰消失 |
| `action_head_model_blocks_0_3_eager` | block0-3 | 0.001831 | 0.002626 | 73.9 ms | 130.4 ms | block0-3 eager，尖峰消失 |
| `action_head_model_blocks_4_7_eager` | block4-7 | 0.0549316 | 0.0769532 | 73.3 ms | 123.1 ms | 后续局部 eager，尖峰仍在 |
| `action_head_model_blocks_8_15_eager` | block8-15 | 0.0549316 | 0.0757393 | 82.0 ms | 114.8 ms | 后半段 eager，尖峰仍在 |
| `action_head_model_attn_all_eager` | 所有 attention | 0.0544739 | 0.0767405 | 96.1 ms | 112.4 ms | 只留 attention eager 不够 |
| `action_head_model_ff_all_eager` | 所有 FFN | 0.0544739 | 0.0754138 | 97.9 ms | 122.3 ms | 只留 FFN eager 不够 |
| `action_head_dit_attn_all` | 只编译 attention | 0.001831 | 0.002542 | 102.7 ms | 103.4 ms | 小 scope 行为稳定，但基本没速度收益 |

第一轮的判断：

- 只要 eager island 覆盖 block0-3，尖峰就消失。
- 只 eager block4-7 或 block8-15，尖峰仍然存在。
- 所以问题大概率在 block0-3，而不是后半段 block。
- 把所有 attention 或所有 FFN 单独留 eager 都不能消掉尖峰，说明它不像是某一个 attention/FFN 子模块的单点问题，更像是 block 级别图边界、residual/norm/attention/FFN 组合路径，或者 block wrapper 被编译后的数值路径问题。

## 第二轮：block0-3 内部细分

| target | eager island | cont max | L2 p99 | compiled p50 | eager p50 | 读法 |
|---|---:|---:|---:|---:|---:|---|
| `action_head_model` | 无 | 0.0549316 | 0.0773858 | 62.9 ms | 139.5 ms | 尖峰基线 |
| `action_head_model_blocks_0_3_eager` | block0-3 | 0.001831 | 0.002626 | 80.3 ms | 121.5 ms | 再次消除尖峰 |
| `action_head_model_blocks_0_1_eager` | block0-1 | 0.001574 | 0.002068 | 75.7 ms | 130.4 ms | 消除尖峰 |
| `action_head_model_blocks_2_3_eager` | block2-3 | 0.0549316 | 0.0765795 | 74.0 ms | 136.4 ms | 尖峰仍在 |
| `action_head_model_blocks_0_0_eager` | block0 | 0.001831 | 0.002533 | 64.8 ms | 137.0 ms | 只留 block0 eager 就够 |
| `action_head_model_blocks_1_1_eager` | block1 | 0.0544739 | 0.0761407 | 67.8 ms | 128.7 ms | 尖峰仍在 |
| `action_head_model_blocks_2_2_eager` | block2 | 0.0549316 | 0.0764477 | 66.9 ms | 131.9 ms | 尖峰仍在 |
| `action_head_model_blocks_3_3_eager` | block3 | 0.0553894 | 0.0771349 | 64.7 ms | 137.8 ms | 尖峰仍在 |

第二轮的关键对照：

- `blocks_0_0_eager`：尖峰消失，最大连续动作差异约 `0.0018`。
- `blocks_1_1_eager`、`blocks_2_2_eager`、`blocks_3_3_eager`：尖峰仍然存在，最大连续动作差异仍是 `0.054` 到 `0.055`。
- `blocks_0_1_eager` 和 `blocks_0_3_eager` 能消除尖峰，主要原因是它们包含 block0。

这基本把责任范围从 `action_head.model` 大图缩到了 **DiT block0 的 compiled path**。

## 原始动作差异对比

`action_head_model` 的尖峰差异：

| dim | diff |
|---|---:|
| x | 0.0549316 |
| y | -0.0059223 |
| z | 0.0423431 |
| roll | -0.0034051 |
| pitch | 0.0336312 |
| yaw | -0.0006345 |
| gripper | -0.0019531 |

`action_head_model_blocks_0_0_eager` 的同一步差异：

| dim | diff |
|---|---:|
| x | 0.0013733 |
| y | -0.0000572 |
| z | 0.0018311 |
| roll | -0.0005239 |
| pitch | -0.0008774 |
| yaw | -0.0003625 |
| gripper | -0.0009766 |

这个对比说明：block0 eager island 不是稍微缓解，而是把 x/z/pitch 三个主要异常维度一起压下去了。

## 工程含义

当前最值得继续验证的工程候选是：

```text
torch_compile_target=action_head_model_blocks_0_0_eager
```

它的优势是：

- 行为上：在 step253 这个已知病态点消除了 `0.055` 量级尖峰。
- 性能上：focused replay 中 compiled p50 约 `64.8 ms`，相比全 eager `137.0 ms` 仍然有明显收益。
- patch scope 上：只需要让第一个 DiT transformer block 留在 eager，其他 `action_head.model` 仍可编译。

但它还不是最终结论。原因是本阶段是同 observation 的 step-focused replay，不是完整 closed-loop rollout。闭环里 observation 会被前一步动作改变，微小差异可能被放大，也可能被系统吸收。因此下一步必须做小规模闭环验证。

## 下一步建议

1. 跑小规模闭环 rollout，对比 `none`、`action_head_model`、`action_head_model_blocks_0_0_eager`。
2. 优先覆盖之前敏感的 matched cases，例如 `task4:init9`、`task6:init8`，再加一个相对稳定 case 做 sanity check。
3. 同时记录 server latency、client latency、显存峰值和 request trace，避免只看成功率。
4. 如果 block0 eager closed-loop 不回归，再扩展到更多 init/task。
5. 如果 closed-loop 仍有异常，再做 block0 内部 hook：分别记录 block0 输入、norm 后、attention 后、FFN 后、residual 后的 diff，进一步判断是 residual/norm/attention/FFN 组合路径还是 block wrapper 编译边界。

## 当前阶段结论

Phase 16.6 的收益很大：我们不再只是知道 “`torch.compile(action_head.model)` 会偶发闭环风险”，而是已经把一个稳定动作尖峰定位到了 **DiT block0 的 compiled path**。这给了一个很具体的折中方向：保留大部分编译加速，只把 block0 留 eager。

下一阶段的核心问题变成：`action_head_model_blocks_0_0_eager` 在真实 closed-loop rollout 中是否既保持成功率，又保留足够推理加速。
