# Phase 13 torch.compile Online Drift Replay

本实验用于拆分两个因素：

```text
compile backend 数值漂移
closed-loop simulator feedback 放大
```

实验边界：

- LIBERO 环境只执行 eager FP16 policy 的 action。
- compiled `action_head.model` 只在同一个 live observation 上旁路计算 action。
- eager 和 compiled 使用相同 denoising seed。
- 因此这里不是 compiled closed-loop benchmark，而是同观测 action drift 诊断。

## 配置

```text
tag: phase13_torch_compile_online_drift_4_6_6_0_v1
cases: 4:6,6:0
task suite: libero_10
denoising steps: 8
compile target: action_head_model
compile backend: inductor
compile mode: reduce-overhead
base seed: 20260613
```

输出文件：

```text
toy_quantvla/results/phase13_torch_compile_online_drift_4_6_6_0_v1.json
toy_quantvla/results/phase13_torch_compile_online_drift_4_6_6_0_v1_trace/
```

## 汇总

| case | eager success | steps | raw rel RMSE mean | raw rel RMSE max | raw max abs diff | LIBERO RMSE mean | LIBERO max abs diff |
|---|---|---:|---:|---:|---:|---:|---:|
| 4:6 | success | 245 | 0.003573 | 0.045184 | 0.142975 | 0.000697 | 0.009155 |
| 6:0 | success | 210 | 0.003889 | 0.011721 | 0.011353 | 0.000645 | 0.005493 |

## 窗口演化

| case | first 5 raw rel RMSE | first 50 raw rel RMSE | first 100 raw rel RMSE | full raw rel RMSE | first 100 raw max | full raw max |
|---|---:|---:|---:|---:|---:|---:|
| 4:6 | 0.002709 | 0.003359 | 0.003925 | 0.003573 | 0.005608 | 0.142975 |
| 6:0 | 0.002657 | 0.003467 | 0.004046 | 0.003889 | 0.005615 | 0.011353 |

## 尖峰位置

| case | metric | top policy step | value |
|---|---|---:|---:|
| 4:6 | raw relative RMSE | 114 | 0.045184 |
| 4:6 | raw max abs diff | 114 | 0.142975 |
| 4:6 | LIBERO action RMSE | 114 | 0.004808 |
| 6:0 | raw relative RMSE | 45 | 0.011721 |
| 6:0 | raw max abs diff | 108 | 0.011353 |
| 6:0 | LIBERO action RMSE | 66 | 0.002115 |

## 延迟

这里的延迟不是最终部署速度，因为每个 env step 都运行了 eager 和 compiled 两条路径，并且统计包含首次 compiled call 的编译开销。不过它仍能给出方向性信号。

| case | eager mean | compiled mean | compiled/eager mean |
|---|---:|---:|---:|
| 4:6 | 0.1414s | 0.0902s | 0.596 |
| 6:0 | 0.1359s | 0.0929s | 0.702 |

## 结论

1. 同观测、同 seed 下，compiled action-head path 与 eager FP16 path 不严格等价。
   平均 raw relative RMSE 约 `0.36-0.39%`，cosine 仍接近 1，但这已经不是 bitwise transparent backend。

2. 危险主要来自局部尖峰。
   `4:6` 在 policy step 114 出现 `0.143` 的 raw max abs diff，这类局部动作尖峰比全局均值更能解释闭环失败。

3. 这次 replay 支持前面的闭环病例判断。
   前面 closed-loop compiled rollout 的失败不是初始状态错配，也不是 RNG 漏控；更合理的解释是 compile backend 的小数值漂移在接触/抓取附近被闭环动力学放大。

4. 后续不应该把 `torch.compile(action_head.model)` 当作透明替换。
   如果继续走 compile 加速，需要收缩 compile scope 或换成更接近 eager 数值路径的 CUDA graph。
