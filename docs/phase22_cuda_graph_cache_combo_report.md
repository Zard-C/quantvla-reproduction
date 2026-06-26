# Phase 22: CUDA Graph 与 Lossless Cache 组合验证

本阶段对应 Phase20 的第三档：CUDA Graph / fixed-shape replay prototype。

目标不是接 simulator rollout，而是在固定真实 observation 上先回答三个问题：

1. CUDA Graph 是否仍然 bit-exact？
2. 与 Phase20 的 lossless cache 叠加后是否仍然 bit-exact？
3. 对 request-level latency 是否有实际改善？

## 1. 实验设置

- GPU: RTX 5090
- checkpoint: GR00T N1.5 LIBERO long
- observation indices: `115, 462, 632, 1063`
- cases: `4 observations x 10 repeats = 40`
- denoising steps: `8`
- capture target: `policy.model.action_head.model.forward`
- graph max captures: `8`
- graph prewarm actions: `4`

这次 probe 不使用 per-Linear profiler，因此没有 Phase21 那种“每个 Linear 后插入 `torch.cuda.synchronize()`”的测量扰动。这里只在每次 `get_action` 后同步一次，用来统计 request latency。

## 2. 四路对比

| Variant | Cache | Graph | p50 | p90 | mean | Drift max abs |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| baseline eager | none | no | 103.37 ms | 179.84 ms | 122.85 ms | 0 |
| baseline graph | none | yes | 102.16 ms | 110.11 ms | 102.00 ms | 0 |
| prepare-only graph | prepare-input pruning | yes | 97.62 ms | 117.87 ms | 97.69 ms | 0 |
| static-only graph | action-head static cache | yes | 98.42 ms | 129.25 ms | 102.87 ms | 0.03284 |
| full-cache graph | prepare + static | yes | 101.60 ms | 113.59 ms | 99.69 ms | 0.03284 |

与 no-cache eager baseline 相比：

| Variant | p50 speedup | p90 speedup | Drift |
| --- | ---: | ---: | ---: |
| baseline graph | `1.012x` | `1.633x` | 0 |
| prepare-only graph | `1.059x` | `1.526x` | 0 |
| static-only graph | `1.050x` | `1.391x` | 0.03284 |
| full-cache graph | `1.017x` | `1.583x` | 0.03284 |

## 3. 关键发现

### 3.1 CUDA Graph 本身仍然干净

no-cache CUDA Graph replay:

- relative RMSE max: `0`
- max abs diff: `0`
- graph cache size: `3`
- graph calls: `352`
- eager fallback calls: `0`

这复现了 Phase14 的结论：只捕获 DiT forward 的 eager kernel 序列，不改变数值路径，fixed replay 是 bit-exact 的。

### 3.2 Prepare-input pruning 可以和 CUDA Graph 叠加

prepare-only CUDA Graph replay:

- p50: `104.25 ms -> 97.62 ms`
- p90: `172.23 ms -> 117.87 ms`
- relative RMSE max: `0`
- max abs diff: `0`

这是目前第三档最值得保留的安全组合。它比纯 graph p50 更好一些，同时没有 action drift。

### 3.3 Action-head static cache 不能直接和 CUDA Graph 叠加

static-only 和 full-cache 两组都出现同样 drift：

- relative RMSE max: `0.02051`
- max abs diff: `0.03284`
- eager A/A 仍然是 0

这说明问题不是 static cache 单独改变了 eager 行为。更可能是 static cache 与 `torch.cuda.make_graphed_callables` 的输入捕获/重放机制存在交互，尤其可疑的是 denoising loop 内复用的 timestep tensor 或其他静态 tensor view。

因此当前不能把 `action-head static cache + CUDA Graph` 当成无损优化。

## 4. 对 `cuda_sync` 的理解

Phase21 里看起来 `cuda_sync` 开销大，主要有两类：

1. profiler 人为同步  
   per-Linear timing 为了给每个 Linear 计时，在每个 wrapped Linear 后都调用 `torch.cuda.synchronize()`。这不是部署路径。

2. 真实 kernel launch / 调度碎片  
   Phase19 看到单个 request 有大量 CUDA kernel launch，这是真实结构问题。CUDA Graph 的价值正是在不改 kernel 数值路径的前提下，把 launch 调度开销打包成 replay。

Phase22 的结果符合这个判断：p50 小幅提升，p90 明显改善。它更像尾延迟优化，而不是主计算加速。

## 5. 当前工程判断

第三档可以继续，但落地边界应当收窄为：

```text
prepare-input pruning + CUDA Graph DiT forward
```

暂时不要组合：

```text
action-head static cache + CUDA Graph
```

原因：

- prepare-only graph: drift 0，p50/p90 都有改善。
- static-only graph: 有 action drift。
- full-cache graph: drift 与 static-only 相同，说明风险来自 static cache 侧。

## 6. 下一步建议

1. 给 inference service 增加可选 `--cuda-graph-dit-forward`。
2. 默认只允许与 `--lossless-cache-prepare-input-pruning` 组合。
3. 明确禁止或 warn `--lossless-cache-action-head-static + --cuda-graph-dit-forward`。
4. 跑一个 2-case matched rollout smoke，验证 success 与 action trace。
5. 如果 smoke 通过，再跑与 Phase20 相同的 2-case latency 对比。

这个方向不是主加速引擎，但它是当前最干净的无损加速候选：

- p50: 约 `1.06x`
- p90: 约 `1.5x`
- fixed replay drift: `0`

## 7. Artifacts

Code:

- `toy_quantvla/phase14_cuda_graph_dit_probe.py`
- `toy_quantvla/lossless_cache_patches.py`

Results:

- `toy_quantvla/results/phase22_cuda_graph_dit_obs4_r10_baseline.json`
- `toy_quantvla/results/phase22_cuda_graph_dit_obs4_r10_prepare_only.json`
- `toy_quantvla/results/phase22_cuda_graph_dit_obs4_r10_static_only.json`
- `toy_quantvla/results/phase22_cuda_graph_dit_obs4_r10_cached.json`

