# Phase 15 Gripper Oracle 双服务端实验

本轮目标是验证 Phase 14 的一个关键猜想：

```text
compiled rollout 失败是否主要由 gripper 二值动作错位触发？
```

做法是跑一个诊断性 oracle：

```text
主策略：compiled action_head_model_blocks_8_15_eager
oracle：独立 eager FP16 server
动作组合：x/y/z/roll/pitch/yaw 来自主策略，action.gripper 来自 eager oracle
```

这不是部署方案，因为每步要访问两个 server。它只用来归因。

## 实验文件

代码：

```text
toy_quantvla/libero_eval_init_range.py
toy_quantvla/run_phase15_gripper_oracle_dual_server.sh
```

结果：

```text
toy_quantvla/results/phase15_gripper_oracle_dual_server_4_9_6_8_v1_compiled_gripper_oracle_client_latency.json
toy_quantvla/results/phase15_gripper_oracle_dual_server_4_9_6_8_v1_compiled_server_latency.json
toy_quantvla/results/phase15_gripper_oracle_dual_server_4_9_6_8_v1_gripper_oracle_server_latency.json
toy_quantvla/results/phase15_gripper_oracle_dual_server_task04_init09_trace.json
toy_quantvla/results/phase15_gripper_oracle_dual_server_task06_init08_trace.json
toy_quantvla/results/phase15_gripper_oracle_dual_server_4_9_6_8_analysis.json
docs/phase15_gripper_oracle_dual_server_4_9_6_8.md
```

## 先排除掉的方案

一开始尝试过单服务端 oracle：

```text
同一个 policy 对象里切 compiled/eager module；
compiled 输出连续动作，eager 输出 gripper。
```

这条路在首个真实 sim request 上触发了很长的 Inductor CPU 编译，6 分多仍没有第一条 request trace，所以停止。后续采用双服务端方案，避免在单进程里反复切 module。

## 成功率结果

对比对象：

| setting | cases | success |
|---|---:|---:|
| FP16 baseline | 5 | 3/5 |
| compiled blocks 8-15 eager island | 5 | 1/5 |
| compiled + eager gripper oracle | 2 | 0/2 |

本轮两个重点 case：

| case | FP16 baseline | compiled | compiled + eager gripper oracle |
|---|---:|---:|---:|
| `task4:init9` | success | fail | fail |
| `task6:init8` | success | task6:init8 在 5-case compiled 里 success，但单独 rerun fail | fail |

两个 oracle rollout 都跑到 `executing action in terminated episode`，也就是失败后跑满/接近 horizon。

## 延迟

因为每步访问两个 server，oracle 总延迟约等于 compiled server + eager server：

| metric | p50 |
|---|---:|
| compiled primary remote | 133.3 ms |
| eager gripper oracle remote | 116.2 ms |
| combined client policy total | 251.2 ms |

server 侧单独看：

| server | p50 |
|---|---:|
| compiled primary | 127.6 ms |
| eager gripper oracle | 111.0 ms |

所以这轮不看加速，只看行为归因。

## 最关键发现

在 oracle trace 内，比较同一个 observation 上：

```text
primary compiled gripper
eager oracle gripper
```

结果：

| case | primary vs oracle gripper bin mismatch | raw abs diff mean | raw abs diff max |
|---|---:|---:|---:|
| `task4:init9` | 0 | 0.00101 | 0.0610 |
| `task6:init8` | 0 | 0.000946 | 0.0135 |

也就是说，在 compiled 已经走偏后的 observation 上，compiled 和 eager 对 gripper 二值动作的判断是一致的。

这句话很重要：

```text
Phase 14 看到的 gripper phase mismatch 不是 compiled gripper channel 自己算错；
它更像是连续动作/视觉状态已经把闭环轨迹带到了另一个局部状态，
然后 eager 和 compiled 都会在那个 observation 上输出同样的 gripper phase。
```

## 和 Phase 14 的关系

Phase 14 发现：

```text
baseline 和 compiled 轨迹对齐后，gripper 在 step 46 或 step 244 出现二值错位。
```

Phase 15 进一步说明：

```text
如果在 compiled 轨迹上把 gripper 换成 eager same-observation gripper，离散 gripper bin 仍然没有改变。
```

所以 gripper mismatch 是一个很好的分叉标记，但不是最底层根因。

更底层的问题仍然是：

1. compiled 和 eager 在连续动作维度存在小差异；
2. 小差异经过闭环环境反馈，改变图像、EEF pose、接触状态；
3. 一旦 observation 进入另一个局部区域，gripper phase 自然也会随之变掉；
4. 此后离散 gripper 会放大失败，但单独替换 gripper 不足以把轨迹拉回。

## 当前判断

`action_head_model_blocks_8_15_eager` 这条路的速度潜力仍然存在，但现在的失败不是一个简单的 gripper 后处理问题。

工程上要继续追这条路，下一步应该定位连续动作维度：

1. 做 translation oracle：`x/y/z` 用 eager，其他维度用 compiled。
2. 做 rotation oracle：`roll/pitch/yaw` 用 eager，其他维度用 compiled。
3. 做 full-action oracle sanity：所有动作维度用 eager，确认双服务端组合链路能复现 FP16 success。
4. 如果 translation oracle 能救回，说明主要风险在末端位置控制；如果 full-action oracle 才能救回，说明误差在多个维度共同闭环放大。

这会比继续调 gripper threshold 或 hysteresis 更有信息量。
