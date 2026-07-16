# Phase48: N1.5 Methodology Transfer Probe

Phase48 切回 GR00T N1.5 LIBERO long-horizon checkpoint，在一个新的 15-case slice 上复用 CLSG-TS v2 的 tactic set。目标不是寻找最终 N1.5 最优 tactic，而是检验“迁移的是搜索/验证流程，不是固定窗口或固定 layer island”这一方法论判断。

本轮结果有一个很有价值的现象：三个加速 tactic 都没有相对 FP16 引入 paired regression，并且都把 FP16 的 `7/15` 提高到 `9/15`。但它们修复的是不同 case：

- `speed_only` 修复 `1:23` 和 `8:23`；
- `window_0_120` 修复 `1:23` 和 `4:22`；
- `combo_blocks0_3_window_0_120` 同样修复 `1:23` 和 `4:22`。

这说明 N1.5 的这个 slice 上，后端扰动仍然在重新分配闭环轨迹 basin；不同 tactic 的 repair profile 不同，支持 task-conditioned tactic search 的必要性。

## 实验设计

- tag prefix: `phase48_n15_methodology_probe_15case_v1`
- case list: `0:21,0:22,0:23,1:21,1:22,1:23,4:21,4:22,4:23,6:21,6:22,6:23,8:21,8:22,8:23`
- deterministic policy seed base: `20260716`
- 候选: FP16 baseline, speed-only compile, duration 0-120, blocks0-3 + duration 0-120。

## 汇总

| run | policy | status | success | p50 ms | p90 ms | speedup | eager frac | reserved MiB | max reserved MiB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| FP16 baseline | no compile | complete | 7/15 | 149.28 | 176.22 | 1.00x | - | 4918.00 | 4918.00 |
| Speed-only compile | compile action_head.model | complete | 9/15 | 75.40 | 155.99 | 1.98x | - | 4960.00 | 5122.00 |
| Duration 0-120 | compile action_head.model; eager fallback for policy steps [0, 120) | complete | 9/15 | 79.38 | 167.60 | 1.88x | 0.20 | 4964.00 | 5126.00 |
| Blocks0-3 + duration 0-120 | compile action_head.model with blocks0-3 eager; eager fallback for policy steps [0, 120) | complete | 9/15 | 119.07 | 183.39 | 1.25x | 0.20 | 4944.00 | 5106.00 |

## Paired repair/regression

| comparison | common | repair | regress | net | repair cases | regress cases |
| --- | --- | --- | --- | --- | --- | --- |
| speed_only vs baseline | 15 | 2 | 0 | 2 | 1:23, 8:23 | - |
| window_0_120 vs baseline | 15 | 2 | 0 | 2 | 1:23, 4:22 | - |
| combo_blocks0_3_window_0_120 vs baseline | 15 | 2 | 0 | 2 | 1:23, 4:22 | - |
| window_0_120 vs speed_only | 15 | 1 | 1 | 0 | 4:22 | 8:23 |
| combo_blocks0_3_window_0_120 vs speed_only | 15 | 1 | 1 | 0 | 4:22 | 8:23 |

## 方法论判读

1. `speed_only` 是本轮最快候选：`1.98x` p50 speedup，且没有 FP16 regression。在这个小 slice 上它可以作为 speed-first incumbent。
2. `window_0_120` 速度略慢：`1.88x`，但它修复的是 `4:22`，而不是 `speed_only` 修复的 `8:23`。这说明 duration protection 改变的是 task-specific repair profile。
3. `combo_blocks0_3_window_0_120` 与 `window_0_120` outcome 完全一致，但 p50 只有 `1.25x`，说明在这个 slice 上额外的 early-block eager island 没带来行为收益，反而牺牲速度。
4. task 6 和 task 8 仍然是弱 slice：task 6 全部失败；task 8 只有 `speed_only` 修复了 `8:23`。这提示下一轮若做 task-conditioned search，应优先围绕 task 6/8 扩 candidate。
5. Phase48 没有推翻 Phase47，反而补了一种互补情况：有时 global speed tactic 在某个 slice 上确实最好，但其他 tactic 仍然修复不同 basin。因此最终方法仍应是 speed-risk frontier + task-aware held-out validation，而不是固定相信某个 tactic。

## Per-case outcomes

| case | baseline | speed_only | window_0_120 | combo_blocks0_3_window_0_120 |
| --- | --- | --- | --- | --- |
| 0:21 | S284 | S282 | S284 | S282 |
| 0:22 | S274 | S282 | S278 | S277 |
| 0:23 | S256 | S254 | S256 | S256 |
| 1:21 | S344 | S380 | S363 | S363 |
| 1:22 | S226 | S224 | S226 | S226 |
| 1:23 | F991 | S426 | S472 | S440 |
| 4:21 | S199 | S196 | S199 | S199 |
| 4:22 | F991 | F991 | S643 | S689 |
| 4:23 | S235 | S220 | S247 | S236 |
| 6:21 | F991 | F991 | F991 | F991 |
| 6:22 | F991 | F991 | F991 | F991 |
| 6:23 | F991 | F991 | F991 | F991 |
| 8:21 | F991 | F991 | F991 | F991 |
| 8:22 | F991 | F991 | F991 | F991 |
| 8:23 | F991 | S463 | F991 | F991 |

## 判读方式

- 如果 `speed_only` 继续在更大 held-out set 上胜出，它可以作为当前 checkpoint/task distribution 的 speed-first incumbent。
- 如果 `window_0_120` 或 combo 在更多 task-4-like slice 上稳定修复 `speed_only` 失败，则它们适合作为 routed/task-conditioned candidate，而不是全局替代。
- 如果下一轮出现 regression，应优先用 paired repair/regression 和 per-task profile 调整 selector，而不是只按 aggregate success 选 tactic。

## 产物

- JSON: `toy_quantvla/results/phase48_n15_methodology_probe_15case_v1_summary.json`
- Report: `docs/phase48_n15_methodology_probe_report_zh.md`
