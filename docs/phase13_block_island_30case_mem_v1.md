# Phase 13 Blocks 8-15 Eager Island 30-Case Matched Rollout

本报告记录 `action_head_model_blocks_8_15_eager` 在更大 matched set 上的闭环成功率、推理速度和显存结果。

## 配置

```text
tag: phase13_block_island_30case_mem_v1
task suite: libero_10
case list:
  task4 init0..9
  task6 init0..9
  task8 init0..9
baseline: FP16 eager
compiled: torch.compile(action_head.model) + transformer blocks 8..15 eager island
compile target: action_head_model_blocks_8_15_eager
compile mode: reduce-overhead
compile backend: inductor
policy seed base: 20260613
```

结果文件：

```text
toy_quantvla/results/phase13_block_island_30case_mem_v1_baseline_client_latency.json
toy_quantvla/results/phase13_block_island_30case_mem_v1_baseline_server_latency.json
toy_quantvla/results/phase13_block_island_30case_mem_v1_baseline_server_prepare.json
toy_quantvla/results/phase13_block_island_30case_mem_v1_compiled_client_latency.json
toy_quantvla/results/phase13_block_island_30case_mem_v1_compiled_server_latency.json
toy_quantvla/results/phase13_block_island_30case_mem_v1_compiled_server_prepare.json
```

## 总表

| policy | success | calls | client mean | client p50 | client p90 | server mean | server p50 | server p90 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| FP16 baseline | 16/30 | 20050 | 153.9 ms | 161.1 ms | 166.5 ms | 149.0 ms | 156.2 ms | 161.5 ms |
| blocks 8-15 eager island | 13/30 | 21316 | 108.0 ms | 81.0 ms | 150.3 ms | 103.3 ms | 76.5 ms | 145.5 ms |

速度收益：

| metric | speedup |
|---|---:|
| server mean | 1.44x |
| server p50 | 2.04x |
| server p90 | 1.11x |
| client mean | 1.43x |
| client p50 | 1.99x |
| client p90 | 1.11x |

关键点：

- server p50 从 `156.2 ms` 降到 `76.5 ms`，速度收益仍然很强。
- client p50 从 `161.1 ms` 降到 `81.0 ms`，说明收益能穿透到 eval 端看到。
- p90 收益只有约 `1.1x`，因为 compiled 仍有 150 ms 左右的长尾。
- compiled 成功率从 baseline `16/30` 降到 `13/30`，不能视为透明替换。
- compiled calls 更多，`21316` vs `20050`，主要来自更多失败 episode 跑到 horizon。

## 分任务成功率

| task | baseline | blocks 8-15 eager island | delta |
|---|---:|---:|---:|
| task4 | 6/10 | 5/10 | -1 |
| task6 | 6/10 | 5/10 | -1 |
| task8 | 4/10 | 3/10 | -1 |
| total | 16/30 | 13/30 | -3 |

逐 case 翻转：

| case | baseline | blocks 8-15 eager island | direction |
|---|---|---|---|
| 4:9 | success | fail | regression |
| 6:8 | success | fail | regression |
| 8:4 | fail | success | rescue |
| 8:7 | success | fail | regression |
| 8:8 | success | fail | regression |

task8 很有代表性：compile 后既有 `8:4` 的失败转成功，也有 `8:7`、`8:8` 的成功转失败。这说明 eager-island backend 不是单向破坏，而是在闭环里重新分配轨迹；不过本轮净效果仍然是负的。

## 显存与预热

| policy | allocated | reserved | max allocated | max reserved |
|---|---:|---:|---:|---:|
| FP16 baseline | 5223.1 MiB | 5512.0 MiB | 5419.4 MiB | 5512.0 MiB |
| blocks 8-15 eager island | 5223.1 MiB | 5538.0 MiB | 5419.4 MiB | 5538.0 MiB |

显存结论：

- compiled reserved memory 多约 `26 MiB`。
- allocated 基本一致。
- 这条路线的主要成本不是 steady-state 显存，而是 compile/prewarm 时间和闭环成功率风险。

预热：

| policy | prewarm mean | prewarm p50 | prewarm max |
|---|---:|---:|---:|
| FP16 baseline | 188.7 ms | 89.8 ms | 488.7 ms |
| blocks 8-15 eager island | 1693.4 ms | 1316.5 ms | 4069.0 ms |

compiled 有明显 cold compile/prewarm 成本。部署时必须做 shape/task-description 覆盖充分的预热，不能把 cold compile 留给在线请求。

## 和 15-Case 的关系

15-case matched set 中，blocks 8-15 eager island 是：

```text
FP16 baseline: 7/15
blocks 8-15 eager island: 7/15
server p50: about 72.5 ms
```

30-case 后，结论变得更谨慎：

```text
FP16 baseline: 16/30
blocks 8-15 eager island: 13/30
server p50: 76.5 ms
```

因此 15-case 的“追平 baseline”不能外推为行为等价。扩大 init 覆盖后，边界样本暴露出净成功率回归。

## 当前判断

`action_head_model_blocks_8_15_eager` 是一条非常有价值的速度路线，但还不是可直接替换的生产 backend。

可以保留的结论：

1. 编译前半 DiT、保护后半 blocks 的策略能稳定获得约 `2x` p50 server speedup。
2. steady-state 显存基本不变，compile 后 reserved memory 只小幅增加。
3. 端到端 client latency 也明显下降，说明收益不只存在于局部 microbench。

必须承认的风险：

1. 闭环成功率从 `16/30` 降到 `13/30`。
2. 行为差异不是简单随机噪声，集中在接触/放置边界样本。
3. 仍存在长尾 latency，p90 speedup 远弱于 p50。
4. compiled 路径会改变轨迹分布：既有回归，也有救回失败的样本。

## 下一步

下一轮不建议继续扩大 rollout 前盲目上线这个 target。更好的方向是把速度收益拆开，寻找更保守的数值路径：

1. 手工拆 DiT denoising loop：
   - blocks 0..7 compile；
   - blocks 8..15 eager；
   - block boundary、final norm、action decoder 分别控制。
2. 测 CUDA graph 捕获 eager denoising loop：
   - 目标是尽量保持 eager 数值路径；
   - 看能否拿到一部分 launch/Python 调度收益。
3. 对 flip cases 做 trace：
   - regression: `4:9`, `6:8`, `8:7`, `8:8`；
   - rescue: `8:4`；
   - 重点看 first divergence、接触阶段、raw action spike。
4. 如果继续走 compile backend，需要用更大的 matched set 固化判据：
   - 至少 3 tasks x 10 init；
   - 最好加入 previously rescued 和 regressed cases；
   - 成功率必须和 FP16 baseline 在统计上接近，不能只看速度。
