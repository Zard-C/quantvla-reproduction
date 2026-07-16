# Phase49 计划：N1.5 task-conditioned tactic expansion

## 目的

Phase48 显示 N1.5 上不同 tactic 修复不同 case：

- `speed_only` 修复 `1:23, 8:23`
- `window_0_120` 和 `combo_blocks0_3_window_0_120` 修复 `1:23, 4:22`
- task 6 在当前 slice 全失败

Phase49 因此不再做全任务平均 sweep，而是聚焦 task 4/6/8，扩展 duration window 和 layer-duration 组合，检查 task-conditioned tactic policy 是否能被更稳定地识别。

## 默认设置

脚本：

```bash
bash toy_quantvla/run_phase49_n15_task_conditioned_expansion.sh
```

cases：

```text
tasks 4/6/8 x init 21/22/23/24/25 = 15 cases
```

tactics：

```text
fp16
speed_only
window_0_60
window_0_120
window_60_180
window_120_260
window_0_240
blocks0_3
combo_blocks0_3_window_0_120
combo_blocks0_3_window_120_260
```

总规模：

```text
10 tactics x 15 cases = 150 episodes
```

## 要回答的问题

1. task 4 是否稳定偏好 early duration protection？
2. task 8 是否更适合 aggressive `speed_only`，而不是 window protection？
3. task 6 是否需要跳出当前 duration/layer candidate pool？
4. 能否构造一个 N1.5 routed policy，例如 task 4 用窗口、task 8 用 speed-only、task 6 fallback 到更保守实现？

## 判读方式

优先看 paired repair/regression，而不是只看 aggregate success。

如果不同 task 的 winner 明显不同，就支持：

> CLSG-TS should output a task-conditioned tactic policy, not a single global tactic.

如果 task 6 所有候选仍然失败，则下一步应围绕 task 6 单独扩展搜索维度，例如更晚 duration window、动作通道敏感性、或者直接保守 fallback。
