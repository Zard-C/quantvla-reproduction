# Phase 5 中文报告：LLM/DiT MLP 消融实验与闭环轨迹重分配分析

## 1. 这一阶段为什么重要

这一阶段的收益很大，因为它把问题从“量化后误差会不会变大”推进到了更关键的行为层面：

```text
量化后的策略在真实闭环仿真里，到底还能不能完成任务？
如果成功率变化了，是整体变好/变坏，还是成功样本被重新分配？
ATM/OHB 是稳定器，还是也可能破坏某些本来能成功的轨迹？
```

实验结果显示，`llm_dit_mlp` selective W4A8 fake quantization 没有把 GR00T/LIBERO 策略搞崩。相反，它在 `libero_10` 的 init `0..14` 上整体保持甚至略高于 FP16。

最重要的结论不是“量化一定更好”，而是：

```text
量化会改变闭环轨迹落入哪个成功/失败 basin。

它会救活一部分 FP16 失败样本，
也会破坏一部分 FP16 成功样本。

最终成功率取决于：

救活数量 - 破坏数量
```

这和我们的视频抽帧病例分析高度一致。

## 2. 实验设置

本轮实验固定配置为：

```text
config: llm_dit_mlp
task suite: libero_10
init indices: 0..14
episodes per mode: 10 tasks * 15 init states = 150
denoising steps: 8
fake quantization: W4A8
activation scale: absmax
quantized modules: 116
quantized groups: dit_mlp_selected, llm_selected
calibration observations: 16
machine: RTX 5090
```

对比模式：

```text
none:
  只做 selective W4A8 fake quantization，不加 ATM/OHB。

atm:
  在 attention logits 层面做 temperature-style correction。

ohb:
  在 attention output 能量层面做 output balancing。

atm_ohb:
  同时使用 ATM 和 OHB。
```

注意：这仍然是行为层 fake quant 验证，不是 packed integer kernel 的速度/显存验证。

## 3. 总体结果

完整 init `0..14`、共 `150` 个 episode 的结果如下：

| policy | successes | success rate | delta vs FP16 |
|---|---:|---:|---:|
| FP16 | 108/150 | 72.0% | +0 |
| `llm_dit_mlp + none` | 113/150 | 75.3% | +5 |
| `llm_dit_mlp + atm` | 114/150 | 76.0% | +6 |
| `llm_dit_mlp + ohb` | 116/150 | 77.3% | +8 |
| `llm_dit_mlp + atm_ohb` | 114/150 | 76.0% | +6 |

几个直接观察：

```text
1. none 已经超过 FP16：
   113/150 vs 108/150

2. OHB 单独最好：
   116/150

3. ATM 和 ATM+OHB 打平：
   都是 114/150

4. ATM+OHB 没有超过 OHB：
   说明两个 correction 不一定线性叠加。
```

因此，这一阶段不能被简单理解成：

```text
ATM/OHB 修好了一个坏掉的量化模型
```

更准确的说法是：

```text
selective W4A8 本身已经行为稳健。
ATM/OHB 进一步改变成功/失败样本分布。
OHB 在当前 ablation 中是最好的单独 correction。
```

## 4. Per-task 结果

| task id | FP16 | none | atm | ohb | atm_ohb | 主要现象 |
|---:|---:|---:|---:|---:|---:|---|
| 0 | 13/15 | 10/15 | 11/15 | 13/15 | 13/15 | none/atm 退化，ohb/atm_ohb 恢复到 FP16 |
| 1 | 11/15 | 13/15 | 13/15 | 14/15 | 14/15 | 量化整体有益 |
| 2 | 12/15 | 13/15 | 14/15 | 15/15 | 13/15 | ohb 最好 |
| 3 | 15/15 | 15/15 | 13/15 | 13/15 | 15/15 | atm/ohb 单独破坏，组合恢复 |
| 4 | 8/15 | 8/15 | 13/15 | 12/15 | 12/15 | ATM/OHB 明显修复 |
| 5 | 15/15 | 14/15 | 14/15 | 15/15 | 15/15 | 基本稳定 |
| 6 | 6/15 | 9/15 | 10/15 | 8/15 | 7/15 | atm 最好 |
| 7 | 11/15 | 8/15 | 8/15 | 8/15 | 10/15 | 量化整体困难 |
| 8 | 3/15 | 9/15 | 4/15 | 6/15 | 3/15 | none 最好，ATM/OHB 削弱有益扰动 |
| 9 | 14/15 | 14/15 | 14/15 | 12/15 | 12/15 | ohb/atm_ohb 退化 |

最关键的两个 task：

```text
task4:
  none: 8/15
  atm:  13/15
  ohb:  12/15
  atm_ohb: 12/15

  这里 ATM/OHB 像稳定器。

task8:
  none: 9/15
  atm:  4/15
  ohb:  6/15
  atm_ohb: 3/15

  这里 raw quant 的扰动反而最有益，
  ATM/OHB 把一部分有益扰动削弱掉了。
```

这两个 task 共同说明：

```text
同一种 correction 在不同 task/init 上可能帮忙，也可能帮倒忙。
```

## 5. Paired outcome：为什么 aggregate 不够看

只看总成功率会隐藏很多信息。更有价值的是 paired transition：

```text
同一个 task/init 上，
FP16 成功还是失败？
量化版本成功还是失败？
```

对于每个量化版本，可以分成四类：

```text
same success:
  FP16 成功，量化也成功

repaired failure:
  FP16 失败，量化成功

new regression:
  FP16 成功，量化失败

same failure:
  FP16 失败，量化也失败
```

相对 FP16 的 paired 结果：

| policy | repaired failures | new regressions | same success | same failure | net |
|---|---:|---:|---:|---:|---:|
| none | 21 | 16 | 92 | 21 | +5 |
| atm | 20 | 14 | 94 | 22 | +6 |
| ohb | 22 | 14 | 94 | 20 | +8 |
| atm_ohb | 19 | 13 | 95 | 23 | +6 |

这张表是整个阶段最重要的统计之一。

它说明：

```text
none 的 +5 不是每个 episode 都变好，
而是：

21 个 FP16 失败被救活
16 个 FP16 成功被弄坏
净收益 +5

ohb 的 +8 也是一样：

22 个 FP16 失败被救活
14 个 FP16 成功被弄坏
净收益 +8
```

所以量化后的成功率变化可以写成：

```text
成功率变化
= 被量化救活的失败样本
- 被量化弄坏的成功样本
```

这就是“闭环轨迹重分配”的经验形式。

## 6. 数学解释：量化诱导的闭环轨迹重分配

本节使用 GitHub Markdown 可渲染的数学公式。同时，每个关键公式前都保留一行“文字公式版”，这样即使本地 Markdown 预览器不支持数学渲染，也能顺畅阅读。

核心目标是把我们观察到的现象写成一个闭环动力系统问题：

```text
量化误差不是一次性输出误差，
而是会进入环境反馈，
最终改变整条轨迹落入哪个成功/失败 basin。
```

### 6.1 闭环系统

机器人执行不是一次性预测，而是闭环：

```text
观察环境 -> 输出动作 -> 环境变化 -> 新观察 -> 新动作 -> ...
```

FP16 策略：

文字公式：

```text
当前动作 = FP16策略(当前观测)
```

$$
a_t = \pi_\theta(o_t)
$$

环境更新：

文字公式：

```text
下一状态 = 环境动力学(当前状态, 当前动作)
```

$$
s_{t+1} = F(s_t, a_t)
$$

合起来：

文字公式：

```text
下一状态 = 环境动力学(当前状态, FP16策略(当前观测))
```

$$
s_{t+1} = F(s_t, \pi_\theta(o_t))
$$

量化后策略可以写成：

文字公式：

```text
量化策略(观测) = FP16策略(观测) + 量化误差(观测)
```

$$
\pi_q(o_t) = \pi_\theta(o_t) + \delta\pi(o_t)
$$

所以量化闭环变成：

文字公式：

```text
量化后的下一状态
= 环境动力学(量化后的当前状态, FP16动作 + 量化误差)
```

$$
s'_{t+1}
=
F\left(s'_t,\ \pi_\theta(o'_t) + \delta\pi(o'_t)\right)
$$

关键点：

```text
量化误差不是只影响当前动作。

当前动作变一点，
环境状态就变一点，
下一帧观察也变一点，
后续动作继续变化。

小误差会进入闭环反馈。
```

### 6.2 误差传播

定义状态差异：

文字公式：

```text
状态差异 = 量化轨迹状态 - FP16轨迹状态
```

$$
e_t = s'_t - s_t
$$

一阶近似可以写成：

文字公式：

```text
下一步状态差异
≈ 闭环反馈增益 * 当前状态差异
  + 动作敏感方向 * 当前量化动作误差
```

$$
e_{t+1}
\approx
A_t e_t + B_t \delta a_t
$$

也就是：

文字公式：

```text
当前量化动作误差 = 量化策略误差(当前观测)
```

$$
\delta a_t = \delta\pi(o_t)
$$

其中：

$A_t$ 是闭环反馈增益，包含环境动力学、观测函数、策略对观测变化的敏感性；$B_t$ 是动作对环境状态的影响方向。

更展开一点：

文字公式：

```text
A_t = 环境对状态的敏感性
      + 环境对动作的敏感性 * 策略对观测的敏感性 * 观测对状态的敏感性
```

$$
A_t
=
\frac{\partial F}{\partial s}
+
\frac{\partial F}{\partial a}
\frac{\partial \pi_\theta}{\partial o}
\frac{\partial o}{\partial s}
$$

文字公式：

```text
B_t = 环境对动作的敏感性
```

$$
B_t
=
\frac{\partial F}{\partial a}
$$

递推展开后：

文字公式：

```text
最终状态误差
≈ 每一步量化动作误差
  经过后续闭环增益传播后的累加
```

$$
e_T
\approx
\sum_{k=0}^{T-1}
\left(
A_{T-1} A_{T-2} \cdots A_{k+1}
\right)
B_k \delta a_k
$$

所以离线 action MSE 小，不代表 rollout 一定一样。

因为真正决定成功的是：

```text
误差经过闭环反馈之后，
有没有把轨迹推到另一个 basin。
```

### 6.3 成功/失败是 basin 判断

LIBERO 成功不是平滑连续分数，更像一个事件：

文字公式：

```text
如果轨迹进入成功 basin，则任务成功。
```

$$
\tau \in \mathcal{G}
\quad \Rightarrow \quad
\text{success}
$$

文字公式：

```text
如果轨迹没有进入成功 basin，则任务失败。
```

$$
\tau \notin \mathcal{G}
\quad \Rightarrow \quad
\text{failure}
$$

设：

$\mathcal{G}$ 表示成功 basin，$\partial \mathcal{G}$ 表示成功/失败边界。

如果 FP16 轨迹靠近边界：

文字公式：

```text
FP16轨迹 到 成功/失败边界 的距离 很小。
```

$$
\operatorname{dist}(\tau_\theta,\ \partial\mathcal{G})
\approx 0
$$

那么一点量化扰动就可能造成：

文字公式：

```text
FP16轨迹在失败 basin，
量化轨迹进入成功 basin。
```

$$
\tau_\theta \notin \mathcal{G},
\quad
\tau_q \in \mathcal{G}
$$

也就是：

```text
FP16 failure -> quant success
```

也可能造成：

文字公式：

```text
FP16轨迹在成功 basin，
量化轨迹掉到失败 basin。
```

$$
\tau_\theta \in \mathcal{G},
\quad
\tau_q \notin \mathcal{G}
$$

也就是：

```text
FP16 success -> quant failure
```

这解释了我们看到的两类现象：

```text
task8:
  none 把很多 FP16/ATM-OHB 失败轨迹推成成功

task0/task7/task9:
  量化也会把一些 FP16 成功轨迹推坏
```

### 6.4 成功率变化公式

对于一批固定 init：

文字公式：

```text
成功率变化
= P(FP16失败 且 量化成功)
  - P(FP16成功 且 量化失败)
```

$$
\Delta P
=
P(\text{FP16 fail},\ \text{quant success})
-
P(\text{FP16 success},\ \text{quant fail})
$$

对应样本计数：

文字公式：

```text
净收益 = 被量化救活的失败样本数 - 被量化弄坏的成功样本数
```

$$
\Delta N
=
N_{\text{repaired failures}}
-
N_{\text{new regressions}}
$$

例如 OHB：

文字公式：

```text
OHB净收益 = 22 - 14 = 8
```

$$
\Delta N_{\text{OHB}}
=
22 - 14
=
8
$$

所以 OHB 的 116/150，不代表它每个地方都更好，而是它修复的样本比破坏的样本更多。

## 7. ATM 和 OHB 的机制解释

### 7.1 ATM：attention temperature correction

ATM 作用在 attention logits 的尺度上。

直觉上：

```text
如果 student attention logits 太软或太尖，
ATM 用 teacher/student 的统计比例修正 query 尺度，
让 attention sharpness 更接近 teacher。
```

可以简化写成：

文字公式：

```text
attention logits = Q 和 K 的点积，再除以 sqrt(d)
```

$$
L
=
\frac{QK^\top}{\sqrt{d}}
$$

ATM 可以理解为：

文字公式：

```text
ATM 把 query 按 alpha 缩放。
```

$$
Q \leftarrow \alpha Q
$$

因此：

文字公式：

```text
query 被缩放后，attention logits 也近似按 alpha 缩放。
```

$$
L \leftarrow \alpha L
$$

其中一个简化的校准系数写法是：

文字公式：

```text
alpha = teacher attention logits 的标准差 / student attention logits 的标准差
```

$$
\alpha
=
\frac{\operatorname{std}(L_{\text{teacher}})}
       {\operatorname{std}(L_{\text{student}})}
$$

这会直接改变 softmax 之前的 attention 分布。

优点：

```text
可以修正 attention routing 的 sharpness。
```

风险：

```text
它直接改变“注意力看哪里”。
如果 calibration 只有 16 个 observations，
一个固定 alpha 不可能适合所有 task/init。
```

这解释了 ATM 的高方差：

```text
task4:
  none 8/15 -> atm 13/15
  强修复

task8:
  none 9/15 -> atm 4/15
  强退化
```

ATM 像一个更激进的轨迹重分配器。

### 7.2 OHB：output energy balancing

OHB 作用在 attention output 的能量上。

直觉上：

```text
attention 已经算完以后，
OHB 调整 attention output 进入 residual stream 的强度。
```

简化写成：

文字公式：

```text
OHB 把 attention output 按 beta 缩放。
```

$$
Y_{\text{attn}}
\leftarrow
\beta Y_{\text{attn}}
$$

其中一个简化的校准系数写法是：

文字公式：

```text
beta = teacher attention output 的 RMS / student attention output 的 RMS
```

$$
\beta
=
\frac{\operatorname{rms}(Y_{\text{teacher}})}
       {\operatorname{rms}(Y_{\text{student}})}
$$

它不直接改变 softmax routing，而是改变 attention branch 的输出能量。

所以 OHB 更保守：

```text
ATM:
  改“看哪里”

OHB:
  改“看见的东西以多大强度进入后续网络”
```

这和实验结果一致：

```text
OHB 总体最好：
  116/150

OHB 在 task8 上比 ATM 伤害小：
  ATM 4/15
  OHB 6/15
  none 9/15
```

OHB 更像 residual-energy stabilizer。

### 7.3 为什么 ATM+OHB 没有超过 OHB

ATM 和 OHB 不是简单相加。

因为：

```text
ATM 先改变 attention 分布。
OHB 再缩放 attention output。
```

但 OHB 缩放的 output 已经是被 ATM 改过 routing 后的结果。

所以两者是耦合的：

```text
ATM changes where attention looks.
OHB changes how loudly that attention enters residual stream.
```

组合后可能：

```text
1. 修正更多错误
2. 互相抵消
3. 过度修正
4. 推到另一个失败 basin
```

本轮结果：

```text
OHB:     116/150
ATM+OHB: 114/150
```

说明至少在这组 calibration 和 init `0..14` 上，组合没有线性收益。

## 8. 和视频病例分析的对应关系

视频抽帧分析支持上面的数学解释。

我们已经整理了 25 张 contact sheets：

```text
analysis_keyframes/
```

单独视频病例文档：

```text
docs/phase5_video_case_analysis.md
```

主要观察：

```text
1. 很多成功/失败差异不是最后一帧才出现，
   而是早期轨迹就分叉。

2. task8 的 none 修复样本中，
   FP16/ATM-OHB 往往跑满 991 帧失败，
   none 通常在 388~652 帧之间成功。

3. task4 的 ATM/OHB 修复样本中，
   ATM/OHB 往往 220~255 帧快速成功，
   FP16/none 跑满失败。

4. 退化样本也支持 basin 解释：
   none 可能造成早期物体关系漂移；
   ATM/OHB 可能稳定但推进不足。
```

这说明消融结果不是随机数字，而是能在视频行为层看到对应机制。

## 9. 当前最强结论

这一阶段可以比较有底气地总结为：

```text
1. selective W4A8 over LLM/DiT MLP layers 是行为可行的。

2. none 已经达到 113/150，
   说明选中的量化层没有破坏 GR00T 在 LIBERO 上的闭环能力。

3. OHB 是当前最好的单独 compensation：
   116/150。

4. ATM/OHB 不是单调提升机制，
   而是任务相关的闭环轨迹重分配机制。

5. 成功率提升来自：
   修复失败样本数量 > 新增失败样本数量。

6. task4 和 task8 是最有解释价值的对照：
   task4 显示 ATM/OHB 的稳定作用；
   task8 显示 raw quant 扰动有时比 correction 更有利。
```

一句话版：

```text
这轮实验真正证明的不是“量化误差更小所以更好”，
而是“selective W4A8 足够稳，而 ATM/OHB 会重新分配闭环轨迹 basin”。
```

## 10. 下一步建议

短期不建议重跑大规模 benchmark。更有价值的是小规模 trace 重跑。

建议选择：

```text
task8 init7:
  none 修复 FP16/ATM-OHB

task8 init9:
  ATM/OHB 修复 FP16/none

task4 init10:
  ATM/OHB 稳定修复

task0 init3:
  none 退化

task6 init1:
  ATM/OHB 退化

task9 init9:
  ATM/OHB 退化

task8 init0:
  FP16 成功，但两个量化版本失败
```

每个 episode 保存：

```text
step
task_id
init_index
success / done / exception
action.x
action.y
action.z
action.roll
action.pitch
action.yaw
action.gripper
robot0_eef_pos
robot0_gripper_qpos
```

这样可以把目前的“视频行为归因”升级成：

```text
视频关键帧
+ action trace
+ end-effector trajectory
+ gripper state
```

到那一步，我们就能更定量地回答：

```text
量化到底是在更早闭合 gripper？
还是改变了 x/y/z 的推进方向？
还是让 action norm 更大/更小？
还是让轨迹在接触前就分叉？
```

这会是下一阶段最值得做的机制验证。
