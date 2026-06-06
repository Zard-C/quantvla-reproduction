# Phase 7 Online ATM/OHB Drift Update

## 1. 这次补了什么

上一版 `phase7_online_drift.py` 只测：

```text
FP16 teacher
vs
FP4-like student
```

这次补上了 Phase 5 对照里的模式：

```text
none
ohb
atm_ohb
```

实现上没有直接复用全局 attention processor，因为 online drift 是单模型双模式：

```text
state.enabled = False
  teacher path

state.enabled = True
  student path
```

所以新增了 student-only ATM/OHB processor：

```text
teacher path:
  不应用 alpha / beta

student path:
  按 mode 应用 ATM / OHB scale
```

这样 teacher baseline 不会被 ATM/OHB 污染。

## 2. 校准方式

当前 ATM/OHB scale 来自 synthetic calibration observations：

```text
variants: zero, midgray, noise
num calibration observations: 3
log clamp: 0.3
```

这和 Phase 5 的真实 LeRobot calibration 不完全一样。它适合快速判断机制是否能跑通，但还不能作为最终校准策略。

## 3. 对照实验

设置：

```text
scope: llm_dit_mlp
cases: task8 init7, task8 init9
max policy steps: 10
teacher controls env
student sees same live observation
matched denoising seed
```

结果文件：

```text
toy_quantvla/results/phase7_online_drift_llm_dit_none_10_v2.json
toy_quantvla/results/phase7_online_drift_llm_dit_ohb_10_v2.json
toy_quantvla/results/phase7_online_drift_llm_dit_atm_ohb_10_v2.json
```

## 4. Drift 对比

| mode | task | init | raw rel RMSE mean | action RMSE mean | max action diff | student/teacher time |
|---|---:|---:|---:|---:|---:|---:|
| none | 8 | 7 | 0.02455 | 0.00586 | 0.02838 | 8.51x |
| none | 8 | 9 | 0.02788 | 0.00844 | 0.02975 | 8.77x |
| ohb | 8 | 7 | 0.02544 | 0.00665 | 0.03113 | 5.56x |
| ohb | 8 | 9 | 0.02924 | 0.00838 | 0.02655 | 4.99x |
| atm_ohb | 8 | 7 | 0.02235 | 0.00562 | 0.02655 | 5.35x |
| atm_ohb | 8 | 9 | 0.02979 | 0.00845 | 0.02655 | 6.27x |

ATM/OHB scale summary:

```text
attention processors: 16
alpha mean/min/max: 1.0275 / 0.9921 / 1.0601
beta mean/min/max: 1.0064 / 0.9690 / 1.0356
```

## 5. 怎么理解

### 5.1 ATM/OHB 已经成功接入 online drift

这一步的工程目标完成了：

```text
1. student-only ATM/OHB 能跑
2. teacher path 不被污染
3. live observation 上能记录 drift 和 timing
```

### 5.2 但当前校准下，ATM/OHB 不是单调收益

在 task8/init7：

```text
none raw rel RMSE:    0.02455
atm_ohb raw rel RMSE: 0.02235
```

ATM+OHB 有改善。

在 task8/init9：

```text
none raw rel RMSE:    0.02788
atm_ohb raw rel RMSE: 0.02979
```

ATM+OHB 略变差。

所以当前结论是：

```text
ATM/OHB mechanism can be tested online,
but synthetic calibration does not yet provide robust drift reduction
for FP4-like llm_dit_mlp.
```

这和 Phase 5 的闭环现象并不矛盾。闭环成功率不是 action drift 的单调函数，小扰动可能 repair，也可能 regression。

## 6. Timing 怎么看

当前 `student/teacher time` 很大，通常约 `5x-9x`。

这不是 CUTLASS 性能结论。

原因：

```text
student path 是 PyTorch fake quant：
  每次 Linear forward 都动态 block quantize activation 和 weight
  没有 packed weight
  没有 CUTLASS FP4 kernel
```

所以 timing 只能说明：

```text
fake quant path 不适合真实加速。
```

真正的速度路线仍然是：

```text
CUTLASS SM120 blockscaled FP4 backend
```

Phase 7 的性能 microbench 已经证明真实 kernel 在代表 GEMM shape 上快于 torch fp16。

## 7. 大规模测试准备

新增批量脚本：

```text
toy_quantvla/run_phase7_online_drift_sweep.sh
```

默认会跑：

```text
scopes:
  dit_mlp_only
  llm_mlp_only
  llm_mlp_dit_mlp
  llm_dit_mlp

modes:
  none
  ohb
  atm_ohb

cases:
  8:7, 8:9
  4:10, 4:11
  0:3, 0:7
  6:1, 9:9

max policy steps:
  80
```

运行方式：

```bash
cd /root/autodl-tmp/quantvla-reproduction
RUN_ID=phase7_online_big_001 \
CASE_LIST="8:7,8:9,4:10,4:11,0:3,0:7,6:1,9:9" \
SCOPES="dit_mlp_only llm_mlp_only llm_mlp_dit_mlp" \
MODES="none ohb atm_ohb" \
MAX_POLICY_STEPS=80 \
bash toy_quantvla/run_phase7_online_drift_sweep.sh
```

## 8. 下一步 gate

建议按这个顺序推进：

```text
1. 大规模 online drift sweep
   先筛出 scope/mode 的稳定组合。

2. paired closed-loop 小样本
   用筛出的组合做 teacher/student 闭环对照。

3. CUTLASS PyTorch extension
   只给通过行为 gate 的 scope 写真实 backend。

4. inference speed benchmark
   比较 FP16 get_action vs CUTLASS-backed student get_action。
```

当前不要把 PyTorch fake quant 的 `student/teacher time` 当成最终速度判断。
