# Phase 17: first-divergence / trajectory branching 分析

## 结论摘要

这轮分析不重新跑推理，只复用已经落盘的 trace。核心结论是：当前看到的失败不是简单的同一 observation 上夹爪阈值抖动，而是闭环轨迹被早期连续动作误差逐步推到不同状态区域，随后在接触或夹爪相位附近发生分叉。

最重要的证据有四条：

- `task4:init9` 在闭环里第 46 步出现夹爪相位分叉，最大末端位置分离达到 172.6 mm；但 same-observation replay 的夹爪二值 mismatch 是 0。
- `task6:init8` 的 same-observation replay 在第 253 步有连续动作 outlier，最大差异 0.054932，但夹爪仍不翻转；闭环中真正的夹爪相位分叉到第 244 步才出现。
- Step253 outlier 的 fixed-seed repeat span 是 0，说明它是确定性的数值路径差异，不是 runtime 随机噪声。
- compile boundary bisect 把该 outlier 定位到 DiT block0 的 compiled path：只让 block0 eager 后，连续动作最大差异从约 0.055 降到约 0.0018。

## 分析定义

为了避免把所有误差混在一起，这里把分叉过程拆成三类事件：

- first action divergence：第一次连续动作差异超过阈值，例如 `max(|delta x,y,z,roll,pitch,yaw|) >= 0.005`。
- first pose divergence：第一次末端执行器位置差异超过阈值，例如 `0.5mm / 1mm / 5mm / 10mm`。
- first phase split：第一次离散控制相位不同，目前主要看夹爪二值动作是否不同。

这三个事件的顺序很关键。若同一 observation 已经夹爪翻转，问题更像阈值 jitter；若同一 observation 不翻转、闭环后才翻转，问题更像轨迹 branching。

## 闭环分叉时间线

| case | FP16 | compiled | same-obs gripper flips | first cont >=0.005 | first pose >=0.5mm | first pose >=1mm | first gripper phase split | max pose split |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| task4_init9 | success | fail | 0 | 8 | 15 (0.531 mm) | 49 (1.113 mm) | 46 (0.609 mm) | 172.6 mm @ 208 |
| task6_init8 | success | fail | 0 | 40 | 59 (0.593 mm) | 63 (1.03 mm) | 244 (1.214 mm) | 169.0 mm @ 648 |

读法：

- `task4:init9` 是较快分叉：第 8 步已有连续动作差异超过 0.005，第 46 步夹爪相位分裂，随后轨迹分离放大到厘米级。
- `task6:init8` 是长潜伏分叉：第 63 步末端位置刚超过 1mm，但第 244 步才出现夹爪相位分裂，第 391 步才超过 5mm。这说明早期误差未必马上失败，但会改变之后进入接触阶段时的状态分布。

## same-observation 对照

| case | same-obs success | max action diff | argmax dim | gripper flips | max raw gripper diff | compiled margin at max gripper diff |
| --- | --- | --- | --- | --- | --- | --- |
| task4_init9 | True | 0.005093 | z | 0 | 0.003174 | 0.4984 |
| task6_init8 | True | 0.054932 | x | 0 | -0.004883 | 0.4955 |

same-observation 对照说明：在固定 observation 上，compiled 和 eager 的夹爪二值输出没有发生翻转。尤其 `task6:init8` 的第 253 步连续动作差异很大，但夹爪 margin 仍离阈值很远。因此闭环失败不能简单归因于 `gripper > 0.5` 这种阈值附近的瞬时抖动。

更合理的解释是：连续动作误差先改变下一帧 observation，新的 observation 再改变后续去噪轨迹和接触几何，最终在夹爪开合、物体接触或恢复路径附近出现相位分叉。

## Step253 局部 outlier

| target | top step | dim | cont max | L2 p99 | repeat span | compiled p50 | eager p50 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| action_head_model | 253 | x | 0.054932 | 0.077386 | 0 | 61.3 ms | 150.8 ms |
| action_head_model_blocks_8_15_eager | 253 | x | 0.054932 | 0.075739 | 0 | 78.1 ms | 110.3 ms |
| action_head_dit_attn_all | 258 | x | 0.00196 | 0.002979 | 0 | 104.0 ms | 106.8 ms |

`action_head_model` 在 248-258 窗口内的连续动作差异如下：

| step | cont max | L2 p99 | dim |
| --- | --- | --- | --- |
| 248 | 0.002747 | 0.003576 | z |
| 249 | 0.002089 | 0.002783 | z |
| 250 | 0.001373 | 0.001385 | y |
| 251 | 0.001373 | 0.002113 | x |
| 252 | 0.001602 | 0.001726 | z |
| 253 | 0.054932 | 0.077386 | x |
| 254 | 0.001373 | 0.00169 | y |
| 255 | 0.000916 | 0.001201 | x |
| 256 | 0.001373 | 0.001649 | x |
| 257 | 0.002747 | 0.002888 | z |
| 258 | 0.004249 | 0.006245 | x |

这张表说明 step253 不是慢性累积误差，而是局部 observation 触发的确定性尖峰。附近大部分 step 在 0.001 到 0.004 量级，只有 step253 跳到 0.0549。

## compile boundary 定位

| target | cont max | L2 p99 | repeat span | compiled p50 | speedup | spike removed |
| --- | --- | --- | --- | --- | --- | --- |
| action_head_model | 0.054932 | 0.077386 | 0 | 54.2 ms | 2.41x | False |
| action_head_model_blocks_0_15_eager | 0.000679 | 0.000707 | 0 | 101.1 ms | 1.07x | True |
| action_head_model_blocks_0_7_eager | 0.001388 | 0.001826 | 0 | 85.2 ms | 1.38x | True |
| action_head_model_blocks_0_3_eager | 0.001831 | 0.002626 | 0 | 73.9 ms | 1.76x | True |
| action_head_model_blocks_0_1_eager | 0.001574 | 0.002068 | 0 | 75.7 ms | 1.72x | True |
| action_head_model_blocks_0_0_eager | 0.001831 | 0.002533 | 0 | 64.8 ms | 2.11x | True |
| action_head_model_blocks_2_3_eager | 0.054932 | 0.07658 | 0 | 74 ms | 1.84x | False |
| action_head_model_blocks_1_1_eager | 0.054474 | 0.076141 | 0 | 67.8 ms | 1.9x | False |
| action_head_model_blocks_2_2_eager | 0.054932 | 0.076448 | 0 | 66.9 ms | 1.97x | False |
| action_head_model_blocks_3_3_eager | 0.055389 | 0.077135 | 0 | 64.7 ms | 2.13x | False |
| action_head_model_blocks_4_7_eager | 0.054932 | 0.076953 | 0 | 73.3 ms | 1.68x | False |
| action_head_model_blocks_8_15_eager | 0.054932 | 0.075739 | 0 | 82 ms | 1.4x | False |
| action_head_dit_attn_all | 0.001831 | 0.002542 | 0 | 102.7 ms | 1.01x | True |

定位结果很直接：

- 让 block0-15、block0-7、block0-3 eager 都能消掉尖峰。
- 只让 block0 eager 也能消掉尖峰。
- 只让 block1、block2、block3、block4-7 或 block8-15 eager，尖峰仍在。
- 所以这个 step253 outlier 的最小已知责任范围是 `DiT block0 compiled path`。

## 对闭环误差修正的含义

这组数据把问题从“平均误差大不大”推进到“哪些扰动会触发轨迹分叉”。后续闭环修正应该优先围绕分叉事件做，而不是继续只优化全局 MSE：

- 在线监控：记录 `first action divergence`、`first pose divergence` 和 `phase split`，把它们作为闭环风险指标。
- 局部 fallback：对已知高风险编译边界，例如 block0 compiled path，做 eager island 或风险触发 fallback。
- 轨迹级校准：校准目标不只看同一 observation 的动作误差，还要看误差是否会把 rollout 推入不同接触区域。
- 反事实 replay：固定某一步 observation，对不同 compile/quant path 做 repeated same-seed replay，区分随机噪声和确定性 outlier。

## 限制

当前结论来自已有 trace，不是新 benchmark。闭环 paired case 只有 `task4:init9` 和 `task6:init8` 两个失败病例，所以它适合解释机制和指导下一步工程验证，还不能当成全任务统计结论。真正要形成强 claim，需要继续在更多 task/init 上自动挖 `first divergence -> phase split -> outcome flip` 的链路。

## 产物

- JSON: `toy_quantvla/results/phase17_first_divergence_branching.json`
- Report: `docs/phase17_branching/first_divergence_trajectory_branching_zh.md`
