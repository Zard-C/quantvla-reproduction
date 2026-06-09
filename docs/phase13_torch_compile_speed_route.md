# Phase 13 torch.compile Speed Route

本阶段目标是先验证一个朴素但关键的问题：在不引入 FP4、不改策略数值路径的前提下，`torch.compile` 能不能让 GR00T server 侧 `policy.get_action` 整体变快。

## 为什么不直接 compile `policy.get_action`

官方 `Gr00tPolicy.get_action` 外层包含大量非 tensor 逻辑：

- observation dict copy；
- batch 维度判断；
- numpy 转换；
- modality transform；
- action `.cpu()`；
- action unnormalize。

这些逻辑很容易触发 Dynamo graph break。直接 compile 整个 `policy.get_action` 大概率会得到很多碎图，收益也会被 Python 数据处理淹没。

更合适的目标是内部 tensor-heavy 子图：

```text
policy.model.get_action
  backbone(...)
  action_head.get_action(...)
    process_backbone_output
    state_encoder
    for denoising step in 0..7:
      action_encoder
      action_head.model(...)
      action_decoder
      Euler update
```

其中第一阶段只 compile：

```text
policy.model.action_head.model
```

也就是 DiT transformer 主体。它的输入主要是 tensors，且每个 `get_action` 会在 8-step denoising loop 中重复调用，比较适合作为第一块实验对象。

## 已加代码

`toy_quantvla/timed_fp16_inference_service.py` 新增：

- `--torch-compile-target`
  - `none`
  - `backbone`
  - `action_head_model`
  - `backbone_action_head_model`
- `--torch-compile-backend`
- `--torch-compile-mode`
- `--torch-compile-fullgraph`
- `--torch-compile-dynamic`
- server prewarm 参数：
  - `--prewarm-observations`
  - `--prewarm-indices`
  - `--prewarm-task-description`
- request trace 参数：
  - `--server-request-trace-jsonl`
  - `--server-request-trace-min-seconds`
  - `--server-request-trace-cuda-sync`

新增 runner：

```bash
bash toy_quantvla/run_phase13_torch_compile_matched_set.sh
```

默认行为：

```text
CASE_LIST=6:1
RUN_BASELINE=1
RUN_COMPILED=1
COMPILE_TARGET=action_head_model
COMPILE_MODE=reduce-overhead
COMPILE_BACKEND=inductor
DETERMINISTIC_POLICY_SEEDS=1
POLICY_SEED_BASE=20260613
```

runner 会先跑 FP16 baseline，再跑 FP16 + `torch.compile(action_head.model)`。两边都做相同的 warmdesc prewarm：

```text
task4 description
task6 description
task8 description
```

这样可以把常见 LLM sequence length 和 DiT shape 的 cold compile 尽量移到正式 eval 之前。

## 判断标准

第一轮只看速度，不做成功率结论：

- server `get_action` p50 / p90 是否下降；
- client policy latency 是否同步下降；
- prewarm/cold compile 是否可接受；
- request trace 是否还有长尾；
- compiled 路径是否能完成同一个 `task:init`。

如果 `action_head_model` 有正收益，再扩大到：

```text
CASE_LIST=4:6,4:7,4:8,4:9,4:10,6:0,6:1,6:2,6:3,6:4,8:6,8:7,8:8,8:9,8:10
```

如果没有正收益，则下一步不该继续把 FP4 压进 DiT MLP，而应该改做：

1. 更细粒度 profile：`backbone`、`action_head.process_backbone_output`、DiT loop、action decoder 分段；
2. CUDA graph 版本的 tensor-only denoise loop；
3. FP4 packed `up_proj` 的 module-level profile，确认 activation pack / wrapper / GEMM 占比。

## 风险

`torch.compile` 对这类大模型推理有几个现实风险：

- 首次编译非常慢，必须依赖 prewarm；
- 不同 task description 可能产生不同 sequence length，导致多次编译；
- `fullgraph=True` 很可能失败，默认不用；
- compile 可能增加显存；
- 如果只 compile DiT transformer，而瓶颈主要在 backbone 或 Python 调度，收益会有限。

因此 Phase 13 的正确读法是：**先用 FP16 compile 找整体加速的天花板和真实瓶颈，再决定 FP4 工程优化该压在哪一段。**

## 2026-06-09 初轮结果

配置：

```text
COMPILE_TARGET=action_head_model
COMPILE_MODE=reduce-overhead
COMPILE_BACKEND=inductor
DETERMINISTIC_POLICY_SEEDS=1
POLICY_SEED_BASE=20260613
```

### 单例 smoke：task6:init1

| policy | success | calls | server mean | server p50 | server p90 | client p50 | client p90 |
|---|---:|---:|---:|---:|---:|---:|---:|
| FP16 baseline | 1/1 | 589 | 0.1509s | 0.1561s | 0.1623s | 0.1616s | 0.1680s |
| FP16 + compile action_head.model | 1/1 | 236 | 0.0868s | 0.0693s | 0.1476s | 0.0746s | 0.1528s |

编译路径有明显单步速度收益，但 calls 从 589 降到 236，说明闭环轨迹也变了，不能把 episode wall time 当作纯速度对比。

### 三例复验：task6:init0,1,2

| policy | successes | calls | server mean | server p50 | server p90 | client p50 | client p90 |
|---|---:|---:|---:|---:|---:|---:|---:|
| FP16 baseline | 3/3 | 1074 | 0.1385s | 0.1538s | 0.1625s | 0.1590s | 0.1674s |
| FP16 + compile action_head.model | 2/3 | 1455 | 0.0846s | 0.0699s | 0.1460s | 0.0746s | 0.1509s |

per-init：

| init | baseline | compiled |
|---:|---|---|
| 0 | success, 210 calls | fail, 991 calls |
| 1 | success, 589 calls | success, 236 calls |
| 2 | success, 275 calls | success, 228 calls |

这说明 `torch.compile(action_head.model)` 是真实的速度正信号，但当前还不能视为行为等价替换。它改变了数值路径，闭环里可能放大为成功/失败切换。

### Cold / Prewarm Cost

| policy | prepare | prewarm total | prewarm p50 | prewarm max |
|---|---:|---:|---:|---:|
| baseline single | 11.09s | 0.85s | 0.087s | 0.514s |
| compiled single | 34.31s | 21.57s | 5.10s | 11.21s |
| baseline 3-case | 12.38s | 0.96s | 0.096s | 0.550s |
| compiled 3-case | 19.66s | 7.48s | 1.52s | 4.27s |

编译开销可以通过 prewarm 前置，但 shape/task description 覆盖仍然要谨慎设计。

### 同观测 Action Drift

用同一个 real observation、同一个 task6 description、同一个 policy seed 比较 FP16 eager 和 compiled action：

| metric | value |
|---|---:|
| teacher get_action | 0.0890s |
| first compiled call | 10.38s |
| second compiled call | 0.1280s |
| action abs RMSE | 0.000994 |
| action relative RMSE | 0.00211 |
| max abs diff | 0.003906 |
| cosine | 0.9999978 |

动作误差本身很小，但在 LIBERO 接触动力学里足以导致闭环轨迹重分配。这个现象和前面量化噪声实验一致。

## 当前结论

1. `torch.compile(action_head.model)` 是目前看到的最强整体速度信号。
   server p50 从大约 154-156 ms 降到大约 69-70 ms，client p50 从大约 159-162 ms 降到大约 75 ms。

2. 速度收益不是来自通信或 client。
   client preprocess/postprocess 仍是 0.1 ms 量级；收益发生在 server `policy.get_action` 内部。

3. compile 路径不是严格数值等价。
   同观测漂移很小，但闭环可能从成功变失败。因此后续需要把它当成一种“加速且有微扰”的 inference backend，而不是透明替换。

4. 这条路比继续压低单个 FP4 GEMM 更值得优先推进。
   它直接作用于 DiT denoising 主体，覆盖面远大于 `up_proj` 或 `dit_mlp_only` 的单层替换。

## RNG 与 Fixed-Observation Replay 复查

上一节 rollout 对比还不能证明“同一策略透明加速”，因为 baseline 和 compiled 是两条独立闭环轨迹。即使每个 request 都设置了 deterministic policy seed，只要第一步动作有微小差异，下一帧 observation 就会不同，后面就不再是严格 paired comparison。

为此新增固定 observation replay：

```text
toy_quantvla/phase13_torch_compile_replay.py
```

它做两件事：

1. eager/eager A/A：同一个 observation、同一个 seed 连跑两次 eager。
2. eager/compiled A/B：先记录 eager action，再 compile `action_head.model`，用同一个 observation 和 seed 跑 compiled。

### Replay 结果：原始数据集 task description

输出：

```text
docs/phase13_torch_compile_replay_obs4_v1.md
toy_quantvla/results/phase13_torch_compile_replay_obs4_v1.json
```

| comparison | rel RMSE mean | rel RMSE max | cosine mean | max abs diff |
|---|---:|---:|---:|---:|
| eager/eager A/A | 0 | 0 | 0.999999999 | 0 |
| eager/compiled | 0.002649 | 0.003966 | 0.999996402 | 0.005493 |

同步离线 latency：

| path | p50 | p90 |
|---|---:|---:|
| eager warm | 0.0877s | 0.0935s |
| compiled warm | 0.1469s | 0.3080s |

### Replay 结果：强制 task6 description

输出：

```text
docs/phase13_torch_compile_replay_obs4_task6desc_v1.md
toy_quantvla/results/phase13_torch_compile_replay_obs4_task6desc_v1.json
```

| comparison | rel RMSE mean | rel RMSE max | cosine mean | max abs diff |
|---|---:|---:|---:|---:|
| eager/eager A/A | 0 | 0 | 0.999999999 | 0 |
| eager/compiled | 0.002694 | 0.004666 | 0.999995924 | 0.004578 |

同步离线 latency：

| path | p50 | p90 |
|---|---:|---:|
| eager warm | 0.0954s | 0.1038s |
| compiled warm | 0.0837s | 0.1008s |

### Replay 解读

1. RNG 控制是有效的。
   eager/eager 同 observation、同 seed 的 action drift 为 0，说明之前测试不是因为 seed 漏控。

2. compile 的确引入了小幅数值扰动。
   `rel RMSE ~= 0.0026-0.0027`，cosine 仍接近 1，但这类微扰足以在 LIBERO 接触动力学里触发闭环轨迹重分配。

3. replay 不是用来否定 rollout 速度信号的。
   三例复验已经给出关键的真实 server/eval 路径信号：compiled 单步 p50 稳定在约 75 ms，baseline 约 159 ms。fixed-observation replay 的作用是把 RNG 和 simulator feedback 拆开，证明差异不是 seed 漏控，而是 compile backend 带来的小幅数值漂移。

4. 因此当前正确结论是：
   `torch.compile(action_head.model)` 是强候选路线，但还不是透明替换。它需要用更大 matched rollout 同时看速度收益和成功率回归。

## Online Drift Replay

fixed-observation replay 只验证了少量静态 observation。为了观察真实成功轨迹上 drift 是否会在接触阶段冒尖，又新增了在线 replay：

```text
script: toy_quantvla/phase13_torch_compile_online_drift.py
tag: phase13_torch_compile_online_drift_4_6_6_0_v1
cases: 4:6,6:0
boundary: eager controls LIBERO; compiled only computes side-by-side actions on the same live observations
```

输出：

```text
docs/phase13_torch_compile_online_drift_4_6_6_0_v1.md
toy_quantvla/results/phase13_torch_compile_online_drift_4_6_6_0_v1.json
toy_quantvla/results/phase13_torch_compile_online_drift_4_6_6_0_v1_trace/
```

结果：

| case | eager success | steps | raw rel RMSE mean | raw rel RMSE max | raw max abs diff | LIBERO action RMSE mean |
|---|---|---:|---:|---:|---:|---:|
| 4:6 | success | 245 | 0.003573 | 0.045184 | 0.142975 | 0.000697 |
| 6:0 | success | 210 | 0.003889 | 0.011721 | 0.011353 | 0.000645 |

关键点：

- eager 控制轨迹在两个稳定回归 case 上都成功，说明这次 replay 的 teacher trajectory 是健康的。
- compiled action 在同一个 observation 上仍有 `0.36-0.39%` 平均 raw relative RMSE。
- `4:6` 在 policy step 114 出现明显尖峰：raw max abs diff `0.142975`，LIBERO action RMSE `0.004808`。
- 因此闭环失败更像是局部数值尖峰进入接触动力学后被放大，而不是整体均匀噪声或 seed 没控住。

## Compile Scope Probe

进一步收缩 compile scope，结果记录在：

```text
docs/phase13_compile_scope_probe.md
toy_quantvla/phase13_compile_targets.py
```

核心结果：

- `action_head_dit_attn_all` 明显降低同观测 drift spike，但 closed-loop server p50 仍是 `153.2 ms`，几乎没有速度收益。
- `action_head_dit_ff_all` 在 `task4:init6` 仍出现大尖峰，raw max abs diff `0.107574`。
- 继续切半后，`action_head_dit_ff_0_7` raw max 只有 `0.018890`，但 `action_head_dit_ff_8_15` raw max 达到 `0.115875`。

当前判断：

```text
速度收益需要整块 DiT 级别的大图。
行为风险主要来自后半 DiT FFN 的局部尖峰。
下一步应尝试 whole action_head.model compile + transformer_blocks.8..15.ff eager island。
```

### Eager Island 复验

已尝试：

```text
target: action_head_model_ff_8_15_eager
compile: whole action_head.model
eager island: transformer_blocks.8..15.ff
report: docs/phase13_eager_island_probe.md
```

结果：

| target | online `4:6` raw max | closed-loop success | server p50 |
|---|---:|---:|---:|
| whole action_head.model | 0.142975 | 5/15 或 flip-case 0/4 | 70.1 ms |
| action_head + FFN 8-15 eager island | 0.129303 | 1/2 | 70.4 ms |
| action_head + blocks 8-15 eager island | 0.134369 | 2/2 | 72.6 ms |

结论：

- 速度保住了，说明 graph break/eager island 没有破坏大图 compile 的主要 p50 收益。
- FFN-only eager island 行为没有救回来，`4:6` 仍然失败，且同观测大尖峰仍接近 whole compile。
- Blocks 8-15 eager island 救回了 `4:6` 和 `6:0` 两个稳定回归 case，且 p50 仍在 `72.6 ms`。
- 风险不只是 FFN forward 内部，更可能包含后半 block 的 residual/norm/add/attention/FFN 整段数据路径。

## 下一步

短期建议：

0. 阅读回归病例分析。
   15-case 与 flip-case 复验已经补充到 `docs/phase13_compile_regression_trace_analysis.md`。当前判断是 compile 速度收益稳定，但 `4:6` 与 `6:0` 存在稳定闭环回归。

1. 扩大 blocks 8-15 eager island 到 15-case matched set。
   当前 `action_head_model_blocks_8_15_eager` 在两个稳定回归 case 上 `2/2`，下一步需要和 baseline、whole compile 在同一 15-case 上对齐比较。

2. 统一统计 rollout 级指标。
   对每个 case 记录 success、action calls、server/client p50/p90、episode wall time。速度收益要和成功率一起读，不能只看单步 latency。

3. 若 15-case 仍有明显回归，再尝试更保守的数值路径。
   手工拆 DiT forward、扩大 eager block 范围，或控制 matmul/reduction 精度，看是否能压低 action spike，同时保留主要速度收益。

4. 研究 CUDA graph / tensor-only denoise loop。
   如果 compile 的行为扰动来自 Inductor 重排，可以尝试 CUDA graph 捕获 eager kernel 序列，理论上可能更接近数值等价。
