# Phase 30: Held-Out Sanity Set 计划

Phase30 用来支撑 ICRA 投稿版的一个关键风险点：

> `0-120` duration proxy 是否只是 Phase29 33-case 上的偶然调参结果？

## Case List

使用全 10 个 LIBERO-10 task 的新 init index `15,16,17`，共 30 cases：

```text
0:15,0:16,0:17,
1:15,1:16,1:17,
2:15,2:16,2:17,
3:15,3:16,3:17,
4:15,4:16,4:17,
5:15,5:16,5:17,
6:15,6:16,6:17,
7:15,7:16,7:17,
8:15,8:16,8:17,
9:15,9:16,9:17
```

它和 Phase29 的 `task 4/6/8, init 0-10` 完全不重叠，并且覆盖更多 task slice。

## Candidates

只跑四个最小对照：

| run | 目的 |
| --- | --- |
| FP16 baseline | 参考闭环行为和 baseline latency |
| speed-only compile | 速度上限，同时暴露 compile-induced regressions |
| blocks0-3 eager | layer proxy sanity |
| duration 0-120 | Phase29 最强 duration proxy |

## Command

```bash
bash toy_quantvla/run_phase30_heldout_sanity.sh
```

默认 tag:

```text
phase30_heldout_sanity_30case_v1
```

默认 deterministic policy seed base:

```text
20260701
```

## Expected Outputs

- `toy_quantvla/results/phase30_heldout_sanity_30case_v1_summary.json`
- `docs/phase30_heldout_sanity_report_zh.md`

## 判读

- 如果 `window_0_120` 在 held-out set 上仍然不低于 speed-only，并且 latency 接近 speed-only，则它支持 ICRA 版的 sensitivity-guided acceleration claim。
- 如果 `window_0_120` 明显退化，则论文应把 `0-120` 降级为 33-case diagnostic，而不是泛化候选。
- 如果 `blocks0_3` 比 `window_0_120` 更稳，则下一步应进入 layer x duration 的组合保护。
