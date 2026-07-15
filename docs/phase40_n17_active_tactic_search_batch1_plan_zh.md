# Phase40: N1.7 active tactic search batch-1 计划

## 目的

本阶段把前面讨论的 active tactic search 落成第一批仿真 probe。它不是完整 Bayesian optimization，而是一个可执行的 warm-start batch：

1. 用 Phase36/37 已经发现的敏感窗口作为先验。
2. 在新的 held-out init 上重新评估固定 tactics。
3. 用 paired repair/regression、latency 和 per-task outcome 为下一轮 active selection / routing 提供数据。

这对应论文中的思想：

```text
A1/A2 证明闭环边界存在且复杂；
Phase40 开始把这种敏感性转成主动 tactic 搜索流程。
```

## 运行配置

- Remote session: `phase40_n17_active_batch1`
- Tag prefix: `phase40_n17_active_tactic_search_batch1_15case_v1`
- Model: GR00T N1.7 LIBERO-10
- Cases:

```text
tasks 0/1/4/6/8 × init 27/28/29 = 15 cases
```

- Tactics:

```text
fp16
speed_only
window_0_20
window_5_15
window_8_18
window_10_20
```

## 候选选择理由

| tactic | 角色 |
| --- | --- |
| `fp16` | reference baseline，用于 paired repair/regression |
| `speed_only` | aggressive speed-first anchor |
| `window_0_20` | Phase37B behavior-first 单 tactic，在 held-out 上最稳 |
| `window_5_15` | Phase37A/37B 的高速折中点，也是 routed policy 的组成部分 |
| `window_8_18` | Phase37A 中较强的细窗口候选，测试边界迁移 |
| `window_10_20` | 另一个短窗口候选，用来区分 timing shift |

## 判读方式

这轮看三件事：

1. **是否存在跨 init 稳定的全局 tactic。**
   如果 `window_0_20` 或 `window_5_15` 在 init 27/28/29 仍然强，说明它们有一定泛化性。

2. **speed-only 的 regression 是否稳定出现。**
   如果 `speed_only` 继续产生 paired regressions，说明纯速度目标风险稳定存在。

3. **是否能为下一轮 active selection 提供 task-level posterior。**
   如果不同 task 的最佳 tactic 不同，下一步应转向 routed / task-conditioned active search；如果某个 fixed window 统一占优，则先扩大 fixed tactic held-out。

## 预期产物

- Summary JSON:

```text
toy_quantvla/results/phase40_n17_active_tactic_search_batch1_15case_v1_summary.json
```

- Report:

```text
docs/phase40_n17_active_tactic_search_batch1_report_zh.md
```

## 当前状态

已在 5090 上启动，tmux session:

```text
phase40_n17_active_batch1
```

启动时间：

```text
2026-07-15 18:03 CST
```
