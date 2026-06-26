# Phase 24: CUDA Graph Service Rollout Smoke

本阶段把 Phase22 已验证过的 DiT forward CUDA Graph replay 接入真实 `RobotInferenceServer` 路径，目标是回答：

1. 真实 LIBERO 闭环 rollout 中是否仍然成功？
2. request-level 延迟是否真正下降？
3. 这个无损加速组合的边界在哪里？

## 1. 接入边界

新增服务端开关：

```text
--cuda-graph-dit-forward
--cuda-graph-max-captures 8
```

当前允许组合：

```text
--lossless-cache-eagle-tokenizer
--lossless-cache-prepare-input-pruning
--lossless-cache-static-normalized-input
--cuda-graph-dit-forward
```

当前禁止组合：

```text
--lossless-cache-action-head-static
--cuda-graph-dit-forward
```

原因是 Phase22 已经测到 `action_head_static_cache + CUDA Graph` 会产生非零 action drift。Phase24 只保留目前可视为无损的组合。

## 2. Prepare-only 检查

设置：

- GPU: RTX 5090
- checkpoint: GR00T N1.5 LIBERO long
- observation index: `115`
- denoising steps: `8`
- capture target: `policy.model.action_head.model.forward`

结果：

| Metric | Value |
| --- | ---: |
| prepare seconds | 12.13 s |
| prewarm requests | 1 |
| prewarm latency | 683.81 ms |
| graph cache size | 1 |
| capture count | 1 |
| capture seconds | 206.89 ms |
| graph calls after prewarm | 8 |
| eager fallback calls | 0 |
| failed | false |

这里的 prewarm latency 包含首次 graph capture，因此不是稳态推理速度。

## 3. 2-case 闭环 smoke

Case list:

```text
4:9,6:8
```

服务端 warmup 额外加入两个 task description，让 CUDA Graph 预先捕获 rollout 中会出现的两种 LLM sequence length：

| Shape item | Value |
| --- | ---: |
| hidden states | `[1, 49, 1536]` |
| encoder hidden states, task 4 | `[1, 566, 2048]` |
| encoder hidden states, task 6 | `[1, 565, 2048]` |
| timestep | `[1]` |

成功率：

| Variant | Episodes | Successes | Success rate |
| --- | ---: | ---: | ---: |
| Phase20 cached server | 2 | 2 | 100% |
| Phase24 cached + CUDA Graph server | 2 | 2 | 100% |

## 4. Latency 对比

Client-side policy latency:

| Variant | Requests | mean | p50 | p90 | p99 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Phase20 cached server | 873 | 148.34 ms | 157.69 ms | 165.78 ms | 184.66 ms |
| Phase24 cached + CUDA Graph server | 873 | 82.23 ms | 71.98 ms | 139.68 ms | 164.20 ms |

Speedup:

| Metric | Speedup |
| --- | ---: |
| client mean | 1.80x |
| client p50 | 2.19x |
| client p90 | 1.19x |
| client p99 | 1.12x |

Server-side `get_action` latency:

| Variant | Requests | mean | p50 | p90 | p99 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Phase20 cached server | 800 | 143.26 ms | 152.51 ms | 160.54 ms | 178.73 ms |
| Phase24 cached + CUDA Graph server | 873 | 77.26 ms | 66.90 ms | 134.57 ms | 160.75 ms |

Phase20 server JSON 只 flush 到 800 条请求，client JSON 有完整 873 条；因此主对比以 client-side 873 请求为准，server-side 作为方向一致的佐证。

## 5. Graph 命中情况

Phase24 2-case 结束时：

| Metric | Value |
| --- | ---: |
| graph cache size | 2 |
| capture count | 2 |
| capture seconds | 516.56 ms |
| graph calls | 7008 |
| eager fallback calls | 0 |
| mismatch calls | 1 |
| failed | false |

`mismatch_calls = 1` 来自第二种 `encoder_hidden_states` shape 首次出现时触发新 capture。它没有造成 eager fallback。完成 capture 后，两条 rollout 都走 graph replay。

## 6. 显存

| Variant | allocated | reserved | max allocated | max reserved |
| --- | ---: | ---: | ---: | ---: |
| Phase20 cached server | 5.48 GB | 5.78 GB | 5.68 GB | 5.78 GB |
| Phase24 cached + CUDA Graph server | 5.51 GB | 5.90 GB | 5.72 GB | 5.90 GB |

CUDA Graph 带来约百 MB 量级的额外 reserved memory，换来 request latency 的明显下降。对 5090 这类显存充足环境，这个 tradeoff 是可以接受的。

## 7. 15-case 对齐验证

为了确认 2-case 不是偶然样本，继续复跑 Phase19 的 15-case FP16 hygiene baseline 子集：

```text
4:6,4:7,4:8,4:9,4:10,
6:0,6:1,6:2,6:3,6:4,
8:6,8:7,8:8,8:9,8:10
```

Success map 完全一致：

| Case group | Phase19 FP16 hygiene | Phase24 cached + CUDA Graph |
| --- | --- | --- |
| task 4, init 6-10 | `1,0,0,1,1` | `1,0,0,1,1` |
| task 6, init 0-4 | `1,1,1,0,0` | `1,1,1,0,0` |
| task 8, init 6-10 | `0,0,0,0,1` | `0,0,0,0,1` |
| total | `7/15` | `7/15` |

Client-side policy latency:

| Variant | Requests | mean | p50 | p90 | p99 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Phase19 FP16 hygiene | 10068 | 148.20 ms | 159.99 ms | 167.13 ms | 188.60 ms |
| Phase24 cached + CUDA Graph | 10068 | 84.95 ms | 73.04 ms | 146.46 ms | 167.63 ms |

Speedup:

| Metric | Speedup |
| --- | ---: |
| client mean | 1.74x |
| client p50 | 2.19x |
| client p90 | 1.14x |
| client p99 | 1.13x |

Server-side `get_action` latency:

| Variant | Requests | mean | p50 | p90 | p99 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Phase19 FP16 hygiene | 10000 | 143.38 ms | 154.82 ms | 162.22 ms | 182.60 ms |
| Phase24 cached + CUDA Graph | 10068 | 80.18 ms | 68.44 ms | 141.72 ms | 163.27 ms |

Phase19 server JSON flush 到 10000 条请求，client JSON 有完整 10068 条；因此完整成功率和主延迟对比仍以 client-side 为准。

15-case 结束时 graph 状态：

| Metric | Value |
| --- | ---: |
| graph cache size | 3 |
| capture count | 3 |
| capture seconds | 737.95 ms |
| graph calls | 80576 |
| eager fallback calls | 0 |
| mismatch calls | 2 |
| failed | false |

缓存命中情况：

| Cache | Hits | Misses | Entries |
| --- | ---: | ---: | ---: |
| Eagle tokenizer | 10069 | 3 | 3 |
| static CUDA tensors | 40280 | 8 | 8 |

显存：

| Variant | allocated | reserved | max allocated | max reserved |
| --- | ---: | ---: | ---: | ---: |
| Phase24 15-case | 5.53 GB | 5.95 GB | 5.73 GB | 5.95 GB |

## 8. 当前结论

这轮结果说明，真正有效的无损加速路径不是继续抠 observation dict 的小开销，而是把 DiT denoising loop 内固定 shape 的 forward kernel launch 序列收进 CUDA Graph。

当前最值得保留的组合是：

```text
tokenizer cache
+ prepare_input pruning
+ static normalized-input CUDA cache
+ DiT forward CUDA Graph
```

它在 2-case 闭环 smoke 上保持 `2/2` 成功，同时把 client p50 从 `157.69 ms` 降到 `71.98 ms`。在更完整的 15-case 子集上，它保持和 FP16 hygiene baseline 完全一致的 `7/15` success map，同时把 client p50 从 `159.99 ms` 降到 `73.04 ms`。

这基本验证了第三档无损加速路线是可走的：它不提高策略本身的成功率，但能在不改变闭环行为的前提下显著降低 FP16 推理延迟。

## 9. 风险和下一步

风险：

- CUDA Graph 对 shape 敏感；新的 language sequence length 会触发新 capture。
- `max_captures` 太小会导致后续 shape fallback 到 eager。
- 不能和 `action_head_static_cache` 直接组合，否则已有非零 action drift 风险。

下一步建议：

1. 记录每个 task description 对应的 `encoder_hidden_states` length，估计真实部署需要多少 graph cache entries。
2. 加一个 warmup manifest，让服务启动时显式预热所有将要评估的 task text，而不是手动传入 `--prewarm-task-description`。
3. 把这条路径作为后续 FP4 / W4A16 实际部署的无损外层加速底座。
4. 如果要继续做无损优化，优先 profiler 看 graph replay 后剩余的 p90/p99 来源，而不是回到每步 observation 小对象构造。
