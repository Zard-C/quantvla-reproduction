# Phase 19 交接记录

## 当前状态

- 日期：2026-06-25
- 远端机器：`ssh -p 33519 root@connect.westd.seetacloud.com`
- GPU：NVIDIA GeForce RTX 5090，空闲时约 `0 MiB / 32607 MiB`
- Python 环境：`/root/autodl-tmp/envs/gr00t-libero-py310/bin/python`
- PyTorch：`2.8.0+cu128`
- 新同步目录：`/root/autodl-tmp/quantvla-reproduction-ad2f9fb-20260625`
- 代码基线：`main@ad2f9fb add phase18 lossless acceleration audit`
- 旧目录：`/root/autodl-tmp/quantvla-reproduction` 保持原样，没有清理或覆盖已有实验现场。

新目录来自本机 `main` 的 Git bundle，并额外叠加了本机尚未提交的 Phase 17 闭环敏感性离线分析脚本与结果：

- `toy_quantvla/phase17_closed_loop_sensitivity_mining.py`
- `toy_quantvla/phase17_first_divergence_branching.py`
- `toy_quantvla/phase17_action_channel_sensitivity.py`
- `toy_quantvla/phase17_module_layer_sensitivity_map.py`
- `docs/phase17_*`
- `toy_quantvla/results/phase17_*.json`

## 已知结论

Phase 18 的核心判断是：先不要急着写更多 kernel。当前 FP16 rollout 的主瓶颈仍在 server 侧 `policy.get_action`，client preprocess/postprocess 只有约 0.1 ms，通信/序列化差额约 4.7 ms。`dit_mlp_only` 覆盖面约 8.4%，单独优化它的端到端收益上限很低。

Phase 17 的核心判断是：闭环风险不是平均 action drift 决定的，而是由 `module scope -> continuous drift -> first divergence -> outcome flip` 这条链条决定。当前最值得继续验证的局部候选是 `action_head_model_blocks_0_0_eager`，因为它能消除已定位的 step253 尖峰，同时保留较多 compile 速度潜力。

## 明天推荐顺序

1. Phase19A：FP16 无损 profiler / hygiene baseline。
   - 不量化，不 compile。
   - 关闭 request trace、关闭显式 per-request CUDA sync。
   - 记录 server latency、client latency、显存。
   - 用 profiler 查隐藏 `cudaMemcpyDtoH`、`aten::item`、`aten::_local_scalar_dense`、`cudaDeviceSynchronize`。

2. Phase19B：敏感性导向 compile 小闭环。
   - 优先测 `action_head_model_blocks_0_0_eager`。
   - matched cases 先覆盖 `task4:init9` 与 `task6:init8`，再扩到 15 或 30 case。
   - 验收指标必须同时包括 success、server p50/p90、first-divergence、continuous action p99/max。

3. Phase19C：FP4/CUTLASS 真实收益门槛。
   - 先只测真实 layer shape 的 steady-state kernel。
   - 把 quant pack、warmup/compile、kernel runtime 分开计时。
   - 只有 kernel 端有真实收益且 same-observation drift 过关，才进入 rollout。

## Phase19A 命令骨架

进入目录：

```bash
cd /root/autodl-tmp/quantvla-reproduction-ad2f9fb-20260625
PYTHON_BIN=/root/autodl-tmp/envs/gr00t-libero-py310/bin/python
```

启动 FP16 server：

```bash
${PYTHON_BIN} toy_quantvla/timed_fp16_inference_service.py \
  --port 5555 \
  --prewarm-observations 1 \
  --prewarm-indices 115 \
  --server-latency-json toy_quantvla/results/phase19_fp16_hygiene_server_latency.json
```

跑小规模 matched set：

```bash
${PYTHON_BIN} toy_quantvla/libero_eval_init_range.py \
  --task-suite-name libero_10 \
  --case-list 4:9,6:8 \
  --headless \
  --port 5555 \
  --deterministic-policy-seeds \
  --policy-seed-base 20260613 \
  --latency-json toy_quantvla/results/phase19_fp16_hygiene_client_latency.json \
  --trace-dir toy_quantvla/results/phase19_fp16_hygiene_trace
```

如果 2-case 正常，再扩到已有 15-case matched set：

```bash
${PYTHON_BIN} toy_quantvla/libero_eval_init_range.py \
  --task-suite-name libero_10 \
  --case-list 4:6,4:7,4:8,4:9,4:10,6:0,6:1,6:2,6:3,6:4,8:6,8:7,8:8,8:9,8:10 \
  --headless \
  --port 5555 \
  --deterministic-policy-seeds \
  --policy-seed-base 20260613 \
  --latency-json toy_quantvla/results/phase19_fp16_hygiene_15case_client_latency.json
```

## 注意

- 今晚没有启动长实验。
- 明天先确认 5090 上没有遗留 server/eval 进程，再跑 Phase19A。
- 如果 FP16 hygiene 没有发现隐藏同步或 D2H 问题，就不要继续在无损 Torch 层死磕，主线应转到 `block0 eager compile guard` 或 FP4 kernel 真实收益验证。
