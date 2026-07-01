# Phase 35: Final Held-Out Tactic Validation Plan

## 目标

Phase35 用一个新的 held-out slice 验证 Phase34 得出的 tactic selection 规则是否继续成立。

这不是新的 tactic search，也不是重新调参。它只验证最终候选：

| candidate | role |
| --- | --- |
| FP16 baseline | reference outcome |
| speed_only | speed upper bound / high-risk candidate |
| window_0_120 | speed-constrained candidate |
| combo_blocks0_3_window_0_120 | behavior-first candidate |

## Slice

- tasks: LIBERO-10 task 0-9
- init states: `21/22/23`
- total: 30 matched episodes per candidate
- deterministic policy seed base: `20260704`

## 运行方式

```bash
bash toy_quantvla/run_phase35_final_validation.sh
```

该 wrapper 复用 Phase32 的 tactic validation 管线，并生成：

- `toy_quantvla/results/phase35_final_validation_30case_v1_summary.json`
- `docs/phase35_final_validation_report_zh.md`

## 判读规则

Phase35 的结论不应该被解读为“第三个 slice 的 winner 就是最终 winner”。它应该进入 multi-fold selector：

```bash
python toy_quantvla/phase34_multifold_tactic_selection.py \
  --fold toy_quantvla/results/phase33_combo_backtest_phase30_30case_v1_summary.json \
  --fold toy_quantvla/results/phase32_tactic_validation_30case_v1_summary.json \
  --fold toy_quantvla/results/phase35_final_validation_30case_v1_summary.json \
  --out-json toy_quantvla/results/phase35_multifold_with_final_validation.json \
  --out-md docs/phase35_multifold_with_final_validation_zh.md
```

重点看：

1. `speed_only` 是否继续有高 regression。
2. `window_0_120` 是否继续保持较高 speedup 和较低 regression。
3. `combo` 是否继续最低 regression，但速度成本更高。
4. behavior-first 和 speed-constrained selector 是否仍然选不同 tactic。

## 预期贡献

如果 Phase35 继续支持 Phase34 的模式，论文可以更强地写成：

> Single-slice tactic winners are unstable. A closed-loop acceleration tactic should be selected by robust multi-fold validation under explicit speed-risk constraints.

如果 Phase35 出现反例，也同样有价值：

> The tactic frontier itself is distribution-sensitive, so deployment should re-estimate the frontier on the target task/init distribution rather than assuming one universal tactic.
