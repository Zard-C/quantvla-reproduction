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

## 第一轮离线结果

已经完成 retrospective scorer：

- script: [`toy_quantvla/phase31_tactic_search_retrospective.py`](../toy_quantvla/phase31_tactic_search_retrospective.py)
- report: [`docs/phase31_tactic_search_retrospective_zh.md`](phase31_tactic_search_retrospective_zh.md)
- json: [`toy_quantvla/results/phase31_tactic_search_retrospective.json`](../toy_quantvla/results/phase31_tactic_search_retrospective.json)

当前 scorer 使用：

```text
score = 100 * success_rate
      + 4 * speedup
      + repair_vs_baseline
      - 2.5 * regress_vs_baseline
      - 10 * baseline_success_gap
      - 4 * worst_task_gap
```

这个分数不是理论最优目标，而是一个启发式 rollout triage。它把闭环成功率放在第一优先级，用 speedup 做次级收益，并用 regression count 和 worst-task risk 避免平均值掩盖脆弱任务。

第一轮结果：

| stage | selected tactic | evidence |
| --- | --- | --- |
| Phase29 probe | `window_0_120` | `19/33`, p50 `69.66 ms`, score 最高 |
| Phase30 held-out | `speed_only` | `25/30`, p50 `50.72 ms`, score 最高 |

因此当前结论是：

- `window_0_120` 是 probe set 上的好 diagnostic，但不是 universal tactic。
- `speed_only` 是 Phase30 这个 held-out slice 上的 winner，但还不能称为稳定 incumbent。
- Phase31 的价值不是证明某个固定窗口永远正确，而是证明 tactic search 必须包含 held-out closed-loop validation。

## 第二轮 held-out 结果

Phase32 已完成，使用新的 held-out init set `18/19/20`：

| run | success | p50 | paired vs FP16 |
| --- | ---: | ---: | --- |
| FP16 baseline | 25/30 | 154.72 ms | - |
| speed-only compile | 20/30 | 68.01 ms | 0 repair / 5 regress |
| window 0-120 | 25/30 | 78.26 ms | 1 repair / 1 regress |
| blocks0-3 + window 0-120 | 25/30 | 88.47 ms | 0 repair / 0 regress |

Phase32 改变了 Phase31 第一轮的判断：

- `speed_only` 不是稳定 incumbent。它在 Phase30 上赢，但在 Phase32 上出现 5 个 FP16 regression。
- `window_0_120` 在 Phase32 上恢复 FP16 aggregate success，但仍有 1 个 repair / 1 个 regression。
- `blocks0-3 + window_0_120` 在 Phase32 上逐 case 保持 FP16 outcome，且仍有 `1.75x` p50 speedup。

因此当前更准确的结论是：

> held-out slice selection is part of the tactic-search problem.

单个 held-out set 不足以确定最终 tactic。我们应该使用多个 held-out slices 或者交叉验证式的 task/init split 来选择 tactic，尤其要关注 paired regression，而不是只看 aggregate success 或单一 slice winner。

## 下一轮 GPU 工作建议

下一轮最小动作是补齐 combo 的验证面：

1. 在 Phase30 的 init `15/16/17` 上追加 `blocks0-3 + window 0-120`。
2. 如果 combo 在 Phase30 和 Phase32 都不差于 FP16 paired outcome，则可作为当前 behavior-preserving tactic。
3. 如果 combo 在 Phase30 退化，则进入更系统的 cross-validation search：把 init slices 拆成 probe / validation folds，用平均 paired regression 和 worst-fold risk 排序。

## 成功标准

最小成功：

- 算法在 Phase28/29 candidate pool 中选出的 tactic 不弱于 speed-only；
- 在多个 held-out slices 上不明显退化；
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
