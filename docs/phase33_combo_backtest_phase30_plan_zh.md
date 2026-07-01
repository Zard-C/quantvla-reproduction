# Phase 33: Combo Backtest on Phase30 Slice

Phase33 的目标是回测 Phase32 上最稳的候选：

```text
blocks0-3 eager + window 0-120 eager fallback
```

到 Phase30 的 held-out slice，也就是 init `15/16/17`。

## 为什么做这个阶段

Phase32 给了一个很好的结果：

| run | success | p50 | paired vs FP16 |
| --- | ---: | ---: | --- |
| speed-only compile | 20/30 | 68.01 ms | 0 repair / 5 regress |
| window 0-120 | 25/30 | 78.26 ms | 1 repair / 1 regress |
| blocks0-3 + window 0-120 | 25/30 | 88.47 ms | 0 repair / 0 regress |

但这还不能说明 combo 是最终 tactic，因为 Phase30 和 Phase32 已经证明：不同 held-out slice 会改变 tactic ranking。Phase33 因此只跑 combo，把它和 Phase30 已有的 FP16 / speed-only / blocks0-3 / window0-120 结果合并比较。

## 实验设计

- case list: all 10 tasks x init `15/16/17` = 30 episodes
- deterministic policy seed base: `20260701`
- reference summary: `toy_quantvla/results/phase30_heldout_sanity_30case_v1_summary.json`
- runner: [`toy_quantvla/run_phase33_combo_backtest_phase30.sh`](../toy_quantvla/run_phase33_combo_backtest_phase30.sh)
- summary: [`toy_quantvla/phase33_combo_backtest_summary.py`](../toy_quantvla/phase33_combo_backtest_summary.py)

## 判读标准

如果 combo 在 Phase30 上：

- success 不低于 FP16；
- paired regressions 接近 0；
- latency 仍明显快于 FP16；

则 combo 可以成为当前最强 behavior-preserving acceleration candidate。

如果 combo 在 Phase30 明显退化：

- 说明 combo 也只是 Phase32 slice winner；
- 最终路线应转向 multi-fold tactic selection，而不是固定 tactic。
