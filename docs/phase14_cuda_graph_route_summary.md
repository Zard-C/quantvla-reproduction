# Phase 14 CUDA Graph DiT 路线小结

这一阶段验证的是一个很具体的问题：如果不改变 eager 数值路径，只把 action head 里重复调用的 DiT forward 用 CUDA graph replay 包起来，能不能获得稳定、无精度回归的推理加速。

结论比较明确：技术上可行，数值上干净，但它不是当前最有价值的主加速路线。

## 实验边界

- 模型：GR00T N1.5 LIBERO long checkpoint。
- 数据：真实 LeRobot observation，不跑 LIBERO simulator。
- 目标边界：`policy.model.action_head.model.forward`。
- denoising steps：`8`。
- 对比方式：同一批 observation、同一批 seed，先跑 eager teacher，再跑 CUDA graph replay student。
- 主要结果文件：
  - `toy_quantvla/results/phase14_cuda_graph_dit_probe_obs4_r3_v3.json`
  - `toy_quantvla/results/phase14_cuda_graph_dit_probe_obs4_r10_v1.json`
  - `docs/phase14_cuda_graph_dit_probe_obs4_r10_v1.md`

这里没有跑闭环 rollout，因为 fixed-observation 已经能回答两个核心问题：有没有 action drift，以及单步 `get_action` 延迟有没有明显下降。

## 关键实现

新增脚本：

- `toy_quantvla/phase14_cuda_graph_dit_probe.py`

脚本做了几件事：

- 读取真实 observation，构造固定 replay cases。
- 用同 seed 跑 eager，得到 teacher actions。
- 把 `action_head.model.forward` 替换成一个 CUDA graph wrapper。
- 按输入 shape 缓存多个 graph，避免不同 `encoder_hidden_states` 长度导致 fallback。
- 禁用 autocast cache，绕过 `make_graphed_callables` 的 capture 限制。
- 输出 latency、drift、graph cache、fallback counter 和显存统计。

## 结果

40 case 稳定版结果如下：

| path | count | mean | p50 | p90 | p99 |
|---|---:|---:|---:|---:|---:|
| eager | 40 | 117.9 ms | 102.8 ms | 168.0 ms | 387.0 ms |
| CUDA graph DiT | 40 | 102.1 ms | 101.0 ms | 111.7 ms | 176.7 ms |

数值漂移：

| comparison | relative RMSE mean | relative RMSE max | max abs diff |
|---|---:|---:|---:|
| eager vs eager | 0 | 0 | 0 |
| eager vs CUDA graph | 0 | 0 | 0 |

CUDA graph counter：

| item | value |
|---|---:|
| graph cache size | 3 |
| capture count | 3 |
| capture time total | 489 ms |
| graph calls | 352 |
| eager fallback calls | 0 |

## 怎么理解

CUDA graph 的好处主要是减少 launch/Python 调度开销，所以它对尾延迟很有效：p90 从约 168 ms 降到 112 ms。

但 p50 几乎没变，只有约 1.02x。说明当前 fixed-observation `get_action` 的主体耗时并不只是 Python launch overhead，而是仍然被真实算子执行、attention/MLP 计算、前后处理、action encoder/decoder、以及 denoising loop 外层开销吃掉。

换句话说，CUDA graph replay 没有改变 kernel 本身，也没有 fuse matmul，没有减少 BF16/FP32 计算量。它只是在“同样的 kernel 序列”外面套了 replay。因此它很干净，但不够猛。

## 和 torch.compile 路线对比

目前两个方向的性格很不一样：

| route | 速度信号 | 数值/行为风险 | 当前判断 |
|---|---:|---:|---|
| CUDA graph DiT | p50 约 1.02x，p90 明显改善 | fixed replay 无 drift | 稳，但不够快 |
| torch.compile block-island | server p50 可到约 76.5 ms | 30-case 从 16/30 掉到 13/30 | 快，但行为风险大 |

所以 CUDA graph 更像一个安全的尾延迟优化工具，不像主加速引擎。主加速仍然要么来自更稳的 compile boundary，要么来自实际低比特 kernel/fused path。

## 后续建议

短期不建议把 CUDA graph DiT 单独推进到 rollout，因为它的 p50 收益太小。更合理的用法是：

- 作为安全基线：如果之后接服务开关，它理论上应当复现 FP16 行为。
- 作为组合优化：可以和更小范围的 compile 或 kernel 替换一起用，专门压尾延迟。
- 作为排查工具：如果某条加速路线有 drift，可以用 CUDA graph 证明“只做 replay 不会引起行为变化”。

下一步仍应回到以速度为导向的主线：继续收缩 torch.compile 的行为风险，或者推进真正改变算子成本的低比特/融合 kernel。
