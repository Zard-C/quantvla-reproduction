# Phase49 计划：N1.5 all-task confirmation fold

## 目的

Phase48 在 N1.5 checkpoint 的 15-case transfer probe 上得到一个很好的信号：

```text
FP16: 7/15
speed_only: 9/15, 2 repairs / 0 regressions
window_0_120: 9/15, 2 repairs / 0 regressions
combo_blocks0_3_window_0_120: 9/15, 2 repairs / 0 regressions
```

但 Phase48 只覆盖 `tasks 0/1/4/6/8`，还不能说明这个信号跨全任务保持。Phase49 因此扩展到 LIBERO-10 全任务 confirmation fold。

## 设置

脚本：

```bash
bash toy_quantvla/run_phase49_n15_alltask_confirmation.sh
```

默认 cases：

```text
tasks 0-9 x init 24/25/26 = 30 cases
```

默认 tactics：

```text
fp16
speed_only
window_0_120
combo_blocks0_3_window_0_120
```

## 要回答的问题

1. Phase48 的 `0` paired regression 是否能跨 all-task fold 保持？
2. `speed_only` 的高速度是否会在新增 task 上重新暴露 regression？
3. `window_0_120` 是否仍然是更均衡的 speed-risk candidate？
4. `combo_blocks0_3_window_0_120` 是否仍然更偏 behavior-first，还是速度收益太小？
5. 是否继续支持方法论结论：迁移的是 CLSG-TS 的搜索/验证流程，而不是固定 tactic。

## 当前运行状态

已在 5090 上启动：

```text
tmux session: phase49_n15_alltask_confirmation
repo: /root/autodl-tmp/quantvla-reproduction-current
tag prefix: phase49_n15_alltask_confirmation_30case_v1
```

预计耗时约 `5.5-7h`，取决于 horizon failure 的数量。
