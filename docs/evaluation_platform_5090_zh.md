# 5090 评测基准与运行背景

本文中的闭环 rollout 与延迟统计都以同一类远端单卡 5090 环境为基准。这个基准的作用是保证不同 tactic 在相同硬件、相同仿真和相同 server/client 边界下比较，而不是宣称这是最终部署平台。

## 硬件与软件边界

| 项目 | 设置 |
| --- | --- |
| GPU | 单卡 NVIDIA RTX 5090 |
| 仿真 | LIBERO / robosuite headless EGL |
| 模型 | GR00T N1.5 LIBERO long-horizon checkpoint；N1.7 用于小规模 held-out transfer check |
| 推理形态 | policy inference server + LIBERO evaluator client，同机运行 |
| 主要速度指标 | server-side warm `get_action` p50 |
| 不计入 p50 的部分 | model loading、prewarm、cold `torch.compile` / Inductor 编译、simulator stepping、client/RPC 包装、video I/O |
| 代表性软件栈 | PyTorch `2.8.0+cu128`，CUDA `12.8` |

## FP16 hygiene baseline

Phase19 用官方 FP16 `Gr00tPolicy` 建立过一个无加速 baseline，用于理解 5090 上的稳定推理速度和显存。

运行边界：

- 模型：`gr00t-n1.5-libero-long-posttrain`
- task suite：`libero_10`
- denoising steps：`8`
- deterministic policy seed：base `20260613`
- workload：15 个混合难度 task/init case
- server timing：server 内部 `get_action`

核心结果：

| 指标 | 数值 |
| --- | ---: |
| server `get_action` count | 10000 |
| server `get_action` p50 | 154.82 ms |
| server `get_action` p90 | 162.22 ms |
| server `get_action` p99 | 182.60 ms |
| CUDA allocated after prewarm | 5.48 GB |
| CUDA reserved after prewarm | 5.78 GB |
| max allocated | 5.68 GB |
| max reserved | 5.78 GB |

这个 baseline 说明：在 5090 上，FP16 GR00T/LIBERO 的主要耗时在 server 内部模型推理，而不是 client 侧包装或显式同步。后续所有 speedup 都应理解为在这个同机 server/client、headless EGL、warm serving 边界下的相对比较。

## 解释口径

- `server p50` 是主速度指标，因为它最接近 policy serving path 的稳态延迟。
- `mean` 和 `max` 会受到 cold compile、prewarm、长尾 episode 和 profiler 边界影响，不直接作为部署速度主指标。
- profile 模式会引入额外同步和采样开销，只用于热点归因，不作为最终 latency。
- 量化或 compile tactic 的 success 必须和同一批 task/init 上的 FP16 结果做 paired repair/regression 对比，不能只看 aggregate success。
