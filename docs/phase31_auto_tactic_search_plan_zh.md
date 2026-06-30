# Phase 31: 自动寻找 Closed-Loop Tactic 的冲刺计划

Phase31 的目标是把目前的经验 guide 升级成一个轻量算法：

> Sensitivity-Guided Tactic Search 自动寻找类似 `0-120` 的 acceleration tactic。

这里的重点不是找到一个固定 universal tactic。固定的 `0-120`、`blocks0-3` 很可能依赖 checkpoint、任务分布和成功边界。更合理的目标是一个可迁移的 **tactic-search procedure**：换模型、换权重、换任务后，重新估计 sensitivity，再自动选 tactic。

## 类比

```text
TensorRT tactic search:
candidate kernels
-> microbenchmark latency
-> local numerical check
-> fastest valid tactic

Closed-loop VLA tactic search:
candidate acceleration policies
-> cheap sensitivity proxy
-> small matched rollout probe
-> held-out validation
-> best speed / behavior tactic
```

区别是闭环 rollout 很贵，所以不能暴力枚举。算法必须先用便宜 proxy 剪枝，再把 rollout budget 用在少量候选上。

## 问题定义

输入：

- reference policy `pi_theta`
- candidate tactics `T`
- cheap calibration/probe observations `C`
- fragile rollout probe set `P`
- held-out validation set `H`
- rollout budget `B`
- latency budget or trade-off coefficient `lambda`

输出：

- 一个 fixed tactic，例如 `duration 0-120`
- 或一个 conditional tactic selector，例如：
  - early-duration dominated -> duration fallback
  - layer-boundary dominated -> layer eager island
  - gripper/channel dominated -> action-channel correction

## Candidate Space

候选 tactic 可以来自：

| family | examples |
| --- | --- |
| compile | speed-only compile, compile submodule, CUDA graph |
| layer protection | block0, blocks0-3, blocks8-15 eager |
| duration protection | 0-120, 0-180, 80-240, post-grasp window |
| quantization scope | llm only, dit mlp only, llm_dit_mlp |
| precision guard | attention fp16, gripper fp16, high-risk module fp16 |
| residual correction | action residual, gripper margin correction |
| kernel replacement | packed int4/fp4 kernel, CUTLASS backend |

## Two-Level Objective

便宜 proxy 阶段可以用：

```text
score_proxy(tau)
  = latency_gain(tau)
  - alpha * open_loop_drift(tau)
  - beta  * action_channel_risk(tau)
  - gamma * routing_or_contact_risk(tau)
```

闭环 rollout 阶段用 paired outcome：

```text
score_rollout(tau)
  = success_rate(tau)
  - lambda * latency(tau)
  - beta   * regression_count(tau)
  - gamma  * worst_slice_risk(tau)
```

其中 `worst_slice_risk` 可以用 task-level 或 case-cluster 的 worst-k / CVaR 近似，避免平均值好看但打爆某一类脆弱任务。

## Algorithm Sketch

```text
Input: policy pi, candidate tactics T, cheap set C, probe set P, held-out set H

1. For each tactic tau in T:
      estimate latency_gain(tau)
      estimate open_loop_drift(tau) on C
      estimate action/layer/duration sensitivity proxy

2. Keep top-K candidates by score_proxy,
      with diversity constraints across tactic families.

3. Run small matched rollouts on P.
      compute repair/regress vs FP16 and speed-only
      compute p50/p90 latency and eager fraction
      compute first-divergence or EEF drift for flips

4. Select top-M candidates by score_rollout.

5. Optionally refine locally:
      split a duration window
      merge layer and duration guard
      shrink broad fallback to smallest useful region

6. Validate final candidate(s) on H.

Return: best tactic or tactic selector.
```

## 最小实验路线

先不新增大量 GPU 工作，优先复用已有数据：

1. **Retrospective search on Phase28/29**
   - 把 Phase29 的 duration windows 和 Phase28C 的 layer candidates 当作 candidate pool。
   - 用 success、latency、repair/regress 计算 `score_rollout`。
   - 验证算法是否会选出 `0-120` 或 `blocks0-3` 这类高 trade-off 候选。

2. **Held-out validation with Phase30**
   - Phase30 已经固定跑 FP16、speed-only、blocks0-3、0-120。
   - 如果 retrospective search 选 `0-120`，Phase30 就是它的 held-out validation。
   - 如果 Phase30 显示 `blocks0-3` 更稳，则算法输出应改成 layer tactic，而不是硬保 `0-120`。

3. **Paper integration**
   - ICRA 主文只放 algorithm sketch 和一张小表。
   - arXiv 长版放完整 scoring、candidate pool、retrospective search 表。

## 成功标准

最小成功：

- 算法在 Phase28/29 candidate pool 中选出的 tactic 不弱于 speed-only；
- 在 Phase30 held-out set 上不明显退化；
- 论文叙事从“我们手工发现 0-120”升级为“我们用 sensitivity-guided search 找到了 0-120 这类 tactic”。

强成功：

- 算法能解释为什么 `0-250` 太宽、`0-220` 更差、`0-120` 更好；
- 算法能在 held-out set 上自动偏向 `0-120` 或 `blocks0-3`；
- 结果能支持未来换 checkpoint/模型时重新搜索 tactic，而不是直接迁移固定窗口。

## 风险

| 风险 | 处理 |
| --- | --- |
| 算法看起来像事后拟合 | 明确 Phase30 是 held-out sanity；不要过度宣称自动最优。 |
| cheap proxy 不够强 | 先把算法定位成 two-stage search，cheap proxy 只负责剪枝，不负责最终选择。 |
| candidate pool 太小 | 承认这是 prototype tactic search，未来可加入真实 packed kernel / quant scopes。 |
| 0-120 held-out 不稳 | 输出从 fixed duration tactic 降级为 tactic-search framework，强调需要重新估计 sensitivity。 |
