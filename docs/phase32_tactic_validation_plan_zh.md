# Phase 32: Held-Out Tactic Validation v2

Phase32 的目标是验证 Phase31 的 retrospective search 结论是否稳定：

> 在新的 held-out init set 上，当前 incumbent `speed_only compile` 是否仍然优于手工 duration/layer protection？

## 为什么做这个阶段

Phase31 已经给出一个重要信号：

- Phase29 probe set 会选择 `window_0_120`；
- Phase30 held-out set 会选择 `speed_only`；
- 所以固定窗口不能写成 universal tactic，必须经过 held-out closed-loop validation。

Phase32 继续沿着这个思路，把 held-out set 换成 init `18/19/20`。如果 `speed_only` 继续胜出，它就是当前 checkpoint/task distribution 下更稳的工程 incumbent；如果组合候选胜出，则说明 sensitivity-guided refinement 仍有机会超过 naive compile。

## 实验设计

- task suite: `libero_10`
- case list: all 10 tasks x init `18/19/20` = 30 episodes
- deterministic policy seed base: `20260702`
- runner: [`toy_quantvla/run_phase32_tactic_validation.sh`](../toy_quantvla/run_phase32_tactic_validation.sh)
- summary: [`toy_quantvla/phase32_tactic_validation_summary.py`](../toy_quantvla/phase32_tactic_validation_summary.py)

## 候选

| candidate | purpose |
| --- | --- |
| FP16 baseline | reference closed-loop behavior |
| speed-only compile | Phase31 current incumbent |
| duration 0-120 | Phase29 probe winner, failed Phase30 held-out |
| blocks0-3 + duration 0-120 | layer x duration combination, tests whether refinement can beat speed-only |

## 判读标准

优先级：

1. success rate / paired net repair-regression
2. regressions vs FP16 and vs speed-only
3. p50/p90 latency
4. worst-task behavior

如果 `speed_only` 胜出：

- 论文应把 `speed_only` 写成当前 empirical incumbent。
- `0-120` 和组合策略写成 search candidates / diagnostics。

如果组合候选胜出：

- 论文可以强调 sensitivity-guided refinement 能够超过 naive compile。
- 下一步应扩大 layer x duration candidate pool。

如果结果混乱：

- 结论仍然是 tactic search procedure，而不是固定 tactic。
- 下一步应增加 held-out seeds 或引入更细的 event/contact proxy。
