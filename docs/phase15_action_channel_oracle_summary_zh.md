# Phase 15 动作通道 Oracle 归因实验

本轮目标是回答一个更细的问题：

```text
compiled action_head_model_blocks_8_15_eager 的失败，到底是哪类动作通道把闭环轨迹带偏？
```

我们沿用双服务端诊断框架：

```text
primary server: compiled action_head_model_blocks_8_15_eager
oracle server: eager FP16
env action: 部分动作维度来自 oracle，其余维度来自 primary
```

这不是部署方案。每步要访问两个 server，所以延迟约等于 compiled + eager。它只用于行为归因。

## 代码变化

`toy_quantvla/libero_eval_init_range.py` 现在支持：

```text
--gripper-oracle-port <port>
--oracle-action-keys gripper
--oracle-action-keys translation
--oracle-action-keys rotation
--oracle-action-keys translation,rotation
--oracle-action-keys all
```

`toy_quantvla/run_phase15_gripper_oracle_dual_server.sh` 现在可通过环境变量切换：

```text
ORACLE_ACTION_KEYS=gripper
ORACLE_ACTION_KEYS=translation
ORACLE_ACTION_KEYS=rotation
ORACLE_ACTION_KEYS=translation,rotation
ORACLE_ACTION_KEYS=all
```

其中：

| group | action channels |
|---|---|
| `translation` | `x/y/z` |
| `rotation` | `roll/pitch/yaw` |
| `continuous` | `x/y/z/roll/pitch/yaw` |
| `all` | `x/y/z/roll/pitch/yaw/gripper` |

## 核心结果

两个重点病例：

```text
task4:init9
task6:init8
```

这两个 case 在 FP16/eager 路径上可成功，但在 compiled 相关路径上容易失败，因此适合作为归因样本。

| setting | oracle action keys | success | task4:init9 | task6:init8 | client policy p50 |
|---|---|---:|---|---|---:|
| gripper-only oracle | `gripper` | 0/2 | fail, 990 steps | fail, 990 steps | 251.2 ms |
| full-action oracle | `all` | 2/2 | success, 224 steps | success, 649 steps | 252.4 ms |
| translation-only oracle | `x/y/z` | 0/2 | fail, 990 steps | fail, 990 steps | 252.4 ms |
| rotation-only oracle | `roll/pitch/yaw` | 0/2 | fail, 990 steps | fail, 990 steps | 252.0 ms |
| continuous-action oracle | `x/y/z/roll/pitch/yaw` | 2/2 | success, 224 steps | success, 649 steps | 252.7 ms |

这个矩阵很有信息量：

1. `all` 能 2/2 成功，说明双服务端 oracle 框架是通的。
2. `gripper` 单独替换仍 0/2，说明 gripper 不是底层根因。
3. `translation` 单独替换仍 0/2，说明位置通道单独不够。
4. `rotation` 单独替换仍 0/2，说明姿态通道单独不够。
5. `translation + rotation` 替换后 2/2 成功，而且 gripper 仍来自 compiled primary，说明这两个病例的恢复不需要替换 gripper。

## 同观测动作差异

在 `continuous-action oracle` 成功路径中，同一个 observation 上 primary compiled 与 eager oracle 的动作差异并不大：

| channel | mean abs diff | p50 abs diff | p99 abs diff | max abs diff |
|---|---:|---:|---:|---:|
| `x` | 0.000679 | 0.000458 | 0.002747 | 0.054932 |
| `y` | 0.000638 | 0.000458 | 0.003319 | 0.006895 |
| `z` | 0.000730 | 0.000479 | 0.002975 | 0.039139 |
| `roll` | 0.000160 | 0.000131 | 0.000524 | 0.002881 |
| `pitch` | 0.000202 | 0.000146 | 0.000585 | 0.033631 |
| `yaw` | 0.000175 | 0.000181 | 0.000725 | 0.001088 |

这支持一个更细的解释：

```text
不是某一个动作维度出现巨大错误；
而是多个连续动作维度同时带有很小的数值偏移。
这些偏移进入 simulator 闭环后，会通过接触、视觉、EEF pose 反馈被逐步放大。
```

单独修 `x/y/z` 或单独修 `roll/pitch/yaw` 都救不回来，说明失败不是一个可分离的一维问题。连续 6 维一起修正后立刻恢复到 224/649 步成功路径，说明它更像一个多维闭环轨迹重分配问题。

## 和 gripper 现象的关系

Phase 14 看到 baseline 和 compiled 在若干 step 出现 gripper phase mismatch。Phase 15 的结果说明：

```text
gripper mismatch 是轨迹分叉后的表征，不是最底层触发源。
```

gripper-only oracle 在同一个 compiled observation 上把 gripper 换成 eager 后仍然失败；而 continuous-action oracle 不替换 gripper 却成功。这说明：

```text
只要连续动作把轨迹拉回 eager-like 区域，compiled gripper 本身可以跟着成功；
如果连续动作已经把 observation 带偏，eager gripper 也不能单独把系统拉回来。
```

## 当前结论

`action_head_model_blocks_8_15_eager` 的风险不在某个单独动作通道，而在连续动作向量的联合微扰。

更准确地说：

```text
compiled 对同观测动作的单步误差很小；
但这些误差分布在 translation 与 rotation 多个维度；
闭环系统对这种多维小偏移敏感；
一旦进入不同接触/视觉状态，后续 gripper phase mismatch 会成为失败标记和放大器。
```

这也解释了为什么 fake quant / FP4 / compile 类实验里会出现“有时量化扰动反而更好”的现象：扰动不是单调损害，它会重新分配闭环轨迹。有些 init 会被推离失败盆地，有些 init 会被推离成功盆地。

## 工程含义

双服务端 oracle 的延迟约 252 ms/step，只能做诊断，不能部署。

部署上更重要的是：

1. 保住 continuous action 的数值一致性，而不是只调 gripper threshold。
2. 对 compile scope 做更细的数值守卫：同观测 `x/y/z/roll/pitch/yaw` 的 max/p99 diff 需要被约束。
3. 后续如果继续追 compile 加速，应优先比较不同 compile scope 的 same-observation continuous-action diff，再跑少量闭环病例。
4. 如果后续追 FP4，也要把 fake quant 的结论视为前期筛选，而不是性能/行为最终结论。

当前最有价值的下一步不是继续调 gripper，而是做：

```text
compile scope 收缩 + same-observation continuous diff + 小规模闭环复验
```

目标是找到一个既保留 compile 速度收益，又不会把连续动作向量推过闭环敏感边界的 scope。
