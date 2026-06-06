# Phase 8 CUTLASS Dense Backend Smoke

日期：2026-06-06

## 目标

这一阶段先不直接声称已经完成真实 FP4 量化 backend，而是验证一条更底层的工程路径：

1. 能否把真实 `torch.Tensor` 传入 CUTLASS CuTe compiled op；
2. 能否把 `nn.Linear` 替换成 CUTLASS backend module；
3. 替换后能否跑通 GR00T `get_action`；
4. 替换后能否驱动 LIBERO simulator 完成小规模 rollout；
5. 初步观察推理速度是否有工程收益。

当前实现是 dense fp16/bf16 GEMM backend。它验证的是接入链路，不是最终的 blockscaled FP4 量化 kernel。

## 新增代码

- `toy_quantvla/cutlass_dense_linear.py`
  - `CutlassDenseLinear`：可替换 `nn.Linear` 的 CUTLASS dense wrapper。
  - `CutlassDenseLinearCache`：按 `(M,K,N,dtype,tile)` 缓存 JIT compiled GEMM，避免每层重复编译。
- `toy_quantvla/phase8_cutlass_dense_real_shape_bench.py`
  - 抓取真实 GR00T selected Linear 的 runtime shape，比较 torch fp16 与 CUTLASS dense warm runtime。
- `toy_quantvla/phase8_cutlass_dense_forward_smoke.py`
  - patch GR00T policy，比较 FP teacher 与 CUTLASS dense student 的 `get_action` 输出与耗时。
- `toy_quantvla/cutlass_dense_inference_service.py`
  - 起一个 CUTLASS dense patched GR00T server，用现有 `libero_eval_init_range.py` 跑小规模 simulator。

5090 环境额外安装了：

```bash
pip install apache-tvm-ffi
```

原因：CUTLASS 的 `--enable-tvm-ffi` 路径需要 `tvm_ffi`，否则无法让 compiled function 直接接收 `torch.Tensor`。

## 结果 1：真实 shape microbench

命令边界：

```bash
python toy_quantvla/phase8_cutlass_dense_real_shape_bench.py \
  --scope llm_dit_mlp \
  --denoising-steps 1 \
  --num-observations 1 \
  --max-cases 3 \
  --repeats 30 \
  --warmup 5 \
  --output-json toy_quantvla/results/phase8_cutlass_dense_real_shape_bench_small.json
```

结果：

| group | cases | torch fp16 mean ms | CUTLASS dense mean ms | speedup vs torch | rel RMSE |
| --- | ---: | ---: | ---: | ---: | ---: |
| `dit_mlp` | 2 | 0.0229 | 0.0528 | 0.448x | 2.89e-4 |
| `llm_attn` | 1 | 0.0168 | 0.0274 | 0.611x | 0 |

JIT compile 约 15s/shape。这个开销是 cold-start，不应该算进 steady-state latency，但服务启动后第一次遇到新 shape 会吃到它。

解读：

- 数值正确，说明真实 tensor 进入 CUTLASS dense op 的路径是通的。
- dense fp16 单层 warm runtime 没有打过 torch/cuBLAS。
- 这不否定 CUTLASS 路线，因为 Phase 7 的 blockscaled FP4 microbench 才是有速度收益的 kernel；dense fp16 这里只是接入骨架。

## 结果 2：GR00T get_action smoke

### `dit_mlp_only`

命令边界：

```bash
python toy_quantvla/phase8_cutlass_dense_forward_smoke.py \
  --scope dit_mlp_only \
  --denoising-steps 8 \
  --num-observations 2 \
  --synthetic-variants zero,noise \
  --output-json toy_quantvla/results/phase8_cutlass_dense_forward_dit_mlp_d8_n2.json
```

结果：

- patched modules：32
- compiled CUTLASS shapes：1
  - `(M=49, K=6144, N=1536, dtype=bf16)`
- action cosine mean：0.9999978
- action relative RMSE mean：0.00204
- max action diff mean：0.00214
- cold student：16.15s
- warm student：0.200s
- teacher：0.755s

注意：当前 dense wrapper 只支持 fp16/bf16。`dit_mlp_only` 里部分 activation 是 float32，因此这些层会走 fallback，不会触发 CUTLASS。真实 FP4 backend 本来就会量化 activation，所以后续应该支持显式 compute dtype / activation packing，而不是继续走 fallback。

### `llm_dit_mlp`

命令边界：

```bash
python toy_quantvla/phase8_cutlass_dense_forward_smoke.py \
  --scope llm_dit_mlp \
  --denoising-steps 8 \
  --num-observations 1 \
  --synthetic-variants zero \
  --output-json toy_quantvla/results/phase8_cutlass_dense_forward_llm_dit_d8_n1.json
```

结果：

- patched modules：116
- compiled CUTLASS shapes：5
  - `(551, 2048, 2048, bf16)`
  - `(551, 2048, 1024, bf16)`
  - `(551, 2048, 6144, bf16)`
  - `(551, 6144, 2048, bf16)`
  - `(49, 6144, 1536, bf16)`
- action cosine：0.9999955
- action relative RMSE：0.00303
- max action diff：0.00269
- cold student：75.28s
- warm student：0.098s
- teacher：0.521s

warm student 比 teacher 快这一点只能作为 smoke 信号，不能当最终性能结论。原因是 synthetic observation、重复调用、GR00T 内部缓存和 Python 调度都会影响端到端秒数。可信的单层 microbench 反而显示 dense fp16 CUTLASS 慢于 torch。

## 结果 3：小规模 LIBERO 成功率 smoke

server：

```bash
python toy_quantvla/cutlass_dense_inference_service.py \
  --scope llm_dit_mlp \
  --denoising-steps 8 \
  --port 5568 \
  --output-json toy_quantvla/results/phase8_cutlass_dense_server_prepare_llm_dit.json
```

eval：

```bash
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl python toy_quantvla/libero_eval_init_range.py \
  --task-suite-name libero_10 \
  --case-list 6:1 \
  --headless \
  --port 5568 \
  --trace-dir /tmp/phase8_cutlass_dense_rollout_task6_init1 \
  --log-file toy_quantvla/results/phase8_cutlass_dense_rollout_task6_init1.log
```

结果：

- task id：6
- init index：1
- task：`put the white mug on the plate and put the chocolate pudding to the right of the plate`
- success：1/1
- server patched modules：116
  - DiT MLP：32
  - LLM attention：48
  - LLM MLP：36
- eval exit code：0

退出时有 robosuite EGL destructor warning，但不影响本次成功率结果；server/eval 进程已清理。

## 当前结论

可以确认：

1. CUTLASS CuTe + TVM FFI 可以作为 PyTorch callable backend 接真实 tensor。
2. `nn.Linear -> CutlassDenseLinear` 的替换链路可以穿过 GR00T `get_action`。
3. `llm_dit_mlp` 范围可以 patch 116 个模块，并在 simulator 中完成至少一个成功 episode。
4. shape-level JIT cache 是必要的；当前每个新 shape 约 15s cold compile。

还不能确认：

1. dense fp16 CUTLASS backend 本身不能证明推理加速，真实 shape microbench 目前慢于 torch fp16。
2. 当前不是实际 FP4 quant backend；没有真实 blockscaled FP4 activation/weight packing。
3. 当前 dense wrapper 对 float32 activation 走 fallback，覆盖率还不等于最终量化覆盖率。
4. 1/1 simulator success 只能证明工程路径能跑通，不是成功率统计。

## 下一步

优先级最高的是把 Phase 7 已经有速度收益的 SM120 blockscaled FP4 kernel 接到同一套 wrapper：

1. 实现 real tensor 的 FP4 block packing：
   - activation：runtime blockscale packing；
   - weight：offline pack once，作为 module buffer；
   - scale tensor：匹配 CUTLASS SM120 blockscaled layout。
2. 在 `CutlassDenseLinear` 的接口下新增 `CutlassBlockscaledFP4Linear`，复用 patch/smoke/server 脚本。
3. 先做 `dit_mlp_only` 和 `llm_mlp_only` 的 get_action drift，再做 `llm_dit_mlp`。
4. 小规模 simulator 使用相同 case list 对比：
   - FP16 baseline；
   - fake FP4 + ATM/OHB；
   - CUTLASS real FP4 backend。

工程判断：CUTLASS 路线是可继续的，但 dense fp16 不是目标形态。真正值得投入的是 blockscaled FP4 packing + kernel wrapper，而不是继续优化 dense fp16 wrapper。
