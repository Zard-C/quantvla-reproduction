# Phase 7 CUTLASS SM120 FP4 Backend Report

## 1. 结论先行

在 RTX 5090 上，CUTLASS CuTe 的 Blackwell GeForce `sm_120` 原生
block-scaled FP4 GEMM 跑通了，并且第一次在真实 GR00T 代表 shape 上
跑出了稳定速度收益。

这条结果和前面的 naive Triton / bitsandbytes 不一样：

```text
naive Triton W4A16:
  正确，但比 torch fp16 慢

bitsandbytes Linear4bit:
  能跑，但 6 个真实 shape 全部比 torch fp16 慢

CUTLASS CuTe SM120 blockscaled FP4:
  6 个真实 shape 全部比 torch fp16 快
```

最重要的限制也要同时写清楚：

```text
当前 CUTLASS benchmark 不是严格的 W4A16 weight-only。
它使用 Blackwell-native blockscaled FP4 GEMM，A/B 输入都为 Float4E2M1FN。
```

所以它不能直接等价替换 Phase 6 的 W4A16 Triton kernel，但它给出了一个
很强的工程信号：

```text
如果我们愿意走 Blackwell 原生 FP4 / NVFP4 / MXFP4 路线，
llm_dit_mlp 的性能目标重新变得现实。
```

## 2. 环境与入口

远端环境：

```text
GPU: RTX 5090
compute capability: sm_120
torch: 2.8.0+cu128
CUDA runtime: 12.8
nvcc: CUDA 13.0
nvidia-cutlass-dsl[cu13]: 4.5.2
CUTLASS checkout: /root/autodl-tmp/cutlass
```

最开始尝试的 SM100 示例失败了：

```text
examples/python/CuTeDSL/cute_ext/blackwell/dense_gemm.py

error:
  expects arch to be one of sm_100/sm_101/sm_103/sm_110,
  but got sm_120a
```

这说明 RTX 5090 不能直接拿 datacenter Blackwell SM100 示例硬套。
正确入口是 CUTLASS 的 Blackwell GeForce 目录：

```text
examples/python/CuTeDSL/cute/blackwell_geforce/kernel/dense_gemm/dense_gemm.py
examples/python/CuTeDSL/cute/blackwell_geforce/kernel/blockscaled_gemm/dense_blockscaled_gemm_persistent_pingpong.py
```

我们新增了可复现脚本：

```text
toy_quantvla/phase7_cutlass_sm120_blockscaled_bench.py
```

输出：

```text
toy_quantvla/results/phase7_cutlass_sm120_blockscaled_bench.json
```

## 3. 官方示例烟测

SM120 dense fp16 示例通过：

| case | dtype | time |
|---|---|---:|
| M=128, N=128, K=256 | fp16/fp16 -> fp16 | 10.416 us |

SM120 blockscaled FP4 示例通过：

| case | dtype | time |
|---|---|---:|
| M=128, N=128, K=256 | FP4/FP4 + SF8 -> fp16 | 9.963 us |

这一步确认的是：

```text
5090 + CUDA 13 + nvidia-cutlass-dsl[cu13] + CUTLASS GeForce examples
可以正常 JIT 编译和运行。
```

## 4. 真实 GR00T Shape Sweep

设置：

```text
backend: CUTLASS CuTe SM120 Blackwell GeForce blockscaled FP4
a_dtype: Float4E2M1FN
b_dtype: Float4E2M1FN
sf_dtype: Float8E8M0FNU
sf_vec_size: 32
c_dtype: Float16
acc_dtype: Float32
warmup: 10
iterations: 50
tile sweep:
  (128, 128, 128)
  (128, 128, 256)
epilogue tile sweep:
  (128, 128)
  (64, 32)
```

### 4.1 Best Per Shape

| family | M | K | N | torch fp16 ms | bnb NF4 ms | CUTLASS FP4 ms | speed vs torch | speed vs bnb NF4 | best tile | best epi |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| DiT MLP | 49 | 1536 | 6144 | 0.0153 | 0.1070 | 0.00958 | 1.60x | 11.17x | 128,128,128 | 64,32 |
| DiT MLP | 49 | 6144 | 1536 | 0.0313 | 0.1052 | 0.02105 | 1.49x | 5.00x | 128,128,256 | 64,32 |
| LLM attn | 551 | 2048 | 1024 | 0.0167 | 0.0985 | 0.01021 | 1.64x | 9.65x | 128,128,256 | 64,32 |
| LLM attn | 551 | 2048 | 2048 | 0.0270 | 0.0996 | 0.01014 | 2.66x | 9.83x | 128,128,128 | 128,128 |
| LLM MLP | 551 | 2048 | 6144 | 0.0762 | 0.1016 | 0.01847 | 4.12x | 5.50x | 128,128,128 | 64,32 |
| LLM MLP | 551 | 6144 | 2048 | 0.0722 | 0.1003 | 0.02258 | 3.20x | 4.44x | 128,128,128 | 64,32 |

### 4.2 Group Summary

| family | cases | mean speed vs torch fp16 | min | max |
|---|---:|---:|---:|---:|
| DiT MLP | 2 | 1.54x | 1.49x | 1.60x |
| LLM attn | 2 | 2.15x | 1.64x | 2.66x |
| LLM MLP | 2 | 3.66x | 3.20x | 4.12x |
| overall | 6 | 2.45x | 1.49x | 4.12x |

整体相对 bnb NF4 的平均速度约为 `7.60x`。

## 5. 怎么理解这个结果

### 5.1 性能上

这个结果说明，之前“4-bit 没有加速”并不是量化路线本身的问题，而是后端问题。

```text
bitsandbytes 慢:
  对这些小 batch / 特定 GR00T shape 没有打到 5090 的高效路径

naive Triton 慢:
  Python/Triton 朴素解包 + matmul 没有用到 Blackwell 原生 FP4 MMA

CUTLASS SM120 快:
  走的是 5090 对应的 Blackwell GeForce FP4 blockscaled MMA 路线
```

尤其是 LLM MLP 两个大矩阵：

```text
M=551, K=2048, N=6144: 4.12x vs torch fp16
M=551, K=6144, N=2048: 3.20x vs torch fp16
```

这说明如果我们最终能把 LLM MLP / DiT MLP 子图稳定接入低精度 GEMM，
工程收益是值得继续做的。

### 5.2 精度/算法上

当前 benchmark 只是性能烟测，还没有证明行为可用。

原因：

```text
Phase 5 fake quant:
  我们主要验证的是 llm_dit_mlp 量化扰动在闭环 rollout 中可接受，
  并观察到有 repair / regression 的轨迹重分配。

Phase 6 W4A16:
  是权重量化 + fp16 activation 的工程原型。

Phase 7 CUTLASS SM120 FP4:
  官方示例是 activation 和 weight 都进入 blockscaled FP4 GEMM。
```

所以后续不能直接说：

```text
把 Phase 5 的 llm_dit_mlp fake quant 替换成这个 kernel 就能上线。
```

更准确的说法是：

```text
我们已经找到了一个真实能快的 5090 低精度 GEMM 后端；
下一步要把量化数值格式和 GR00T 的行为稳定性重新对齐。
```

## 6. 下一步建议

建议把 Phase 7 后半段拆成两个分支。

### 分支 A：Blackwell 原生 FP4 路线

目标：

```text
验证 activation FP4 / blockscaled FP4 对 GR00T action drift 和 rollout 的影响。
```

动作：

```text
1. 做离线 forward drift：
   scope: dit_mlp_only, llm_mlp_only, llm_dit_mlp
   quant format: blockscaled FP4-like activation + weight

2. 如果 drift 可控：
   做少量 rollout smoke

3. 如果 rollout 不崩：
   再封装 CUTLASS PyTorch extension
```

优点：

```text
速度收益最大，最贴近 5090 原生能力。
```

风险：

```text
activation FP4 可能比 Phase 5 的 fake quant 更激进。
行为稳定性需要重新验证。
```

### 分支 B：真正 W4A16 / Weight-Only 路线

目标：

```text
保留 fp16 activation，只压缩 weight。
```

动作：

```text
1. 继续查 CUTLASS 是否有 SM120 fp16 x int4/fp4 weight-only GEMM 路线。
2. 同时评估 TensorRT INT4 weight-only quantization 是否能编译较大子图。
3. 避免单层 TensorRT engine，因为 per-Linear 调用开销很可能吞掉收益。
```

优点：

```text
更接近 Phase 6 W4A16 设定，数值风险较小。
```

风险：

```text
不一定有现成 SM120 原生 fast path；
如果只能做手写 dequant + fp16 matmul，速度可能又回到 Triton 的问题。
```

## 7. 当前阶段判断

Phase 7 到这里可以更新判断：

```text
性能后端问题已经出现突破口。
```

但工程路线要从“朴素 W4A16 Triton”调整为：

```text
优先研究 Blackwell GeForce SM120 原生 blockscaled FP4。
```

对 `llm_dit_mlp_only` 的期望也要更精确：

```text
短期：
  不急着接 rollout。
  先证明 FP4/blockscaled 数值格式在 offline forward drift 上可接受。

中期：
  如果 drift 可控，再做 PyTorch extension 和少量 rollout。

长期：
  再决定是走 Blackwell-native FP4，还是回到 TensorRT/CUTLASS weight-only。
```

Follow-up:

```text
docs/phase7_online_drift_report.md
```

The first online same-observation drift probe has been completed with
teacher-controlled LIBERO rollouts and FP4-like student side evaluation.
