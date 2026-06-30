# Phase 29: Finer Duration Proxy

Phase29 是 Phase28D 之后的独立 finer duration proxy 搜索。它不改变已有 Phase28D tag；默认窗口围绕 `0-250` 拆细，用来判断收益是否来自接近、接触/抓取、初始搬起，还是后续搬运阶段。

## Runs

| run | policy | status | success | server p50 ms | speedup | eager frac |
| --- | --- | --- | --- | --- | --- | --- |
| baseline | no compile | complete | 19/33 | 156.22 | 1.00x | - |
| speed_only | compile action_head.model | complete | 16/33 | 70.20 | 2.23x | - |
| phase28d_window_0_250 | existing Phase28D broad early fallback reference | complete | 18/33 | 88.75 | 1.76x | 0.38 |
| window_0_120 | compile action_head.model; eager fallback for policy steps [0, 120) | complete | 19/33 | 69.66 | 2.24x | 0.19 |
| window_0_180 | compile action_head.model; eager fallback for policy steps [0, 180) | complete | 18/33 | 77.55 | 2.01x | 0.28 |
| window_0_220 | compile action_head.model; eager fallback for policy steps [0, 220) | complete | 16/33 | 90.49 | 1.73x | 0.33 |
| window_80_240 | compile action_head.model; eager fallback for policy steps [80, 240) | complete | 14/33 | 75.98 | 2.06x | 0.21 |
| window_120_280 | compile action_head.model; eager fallback for policy steps [120, 280) | complete | 16/33 | 73.48 | 2.13x | 0.22 |
| window_160_240 | compile action_head.model; eager fallback for policy steps [160, 240) | complete | 15/33 | 69.29 | 2.25x | 0.11 |
| window_240_320 | compile action_head.model; eager fallback for policy steps [240, 320) | complete | 17/33 | 66.71 | 2.34x | 0.08 |

## Paired Repair / Regress

| pair | repair | regress | net | repair cases | regress cases |
| --- | --- | --- | --- | --- | --- |
| phase28d_window_0_250 vs baseline | 1 | 2 | -1 | 8:9 | 8:3, 8:10 |
| window_0_120 vs baseline | 2 | 2 | 0 | 6:3, 8:4 | 6:10, 8:3 |
| window_0_180 vs baseline | 2 | 3 | -1 | 8:6, 8:9 | 6:8, 8:7, 8:10 |
| window_0_220 vs baseline | 0 | 3 | -3 | - | 6:8, 8:3, 8:7 |
| window_80_240 vs baseline | 1 | 6 | -5 | 8:9 | 4:6, 4:10, 6:10, 8:7, 8:8, 8:10 |
| window_120_280 vs baseline | 2 | 5 | -3 | 6:7, 8:9 | 4:6, 4:10, 6:0, 8:3, 8:7 |
| window_160_240 vs baseline | 1 | 5 | -4 | 6:7 | 4:6, 4:10, 6:0, 8:7, 8:10 |
| window_240_320 vs baseline | 2 | 4 | -2 | 6:7, 6:9 | 4:6, 6:0, 6:6, 8:7 |
| phase28d_window_0_250 vs speed_only | 4 | 2 | 2 | 4:6, 6:0, 6:6, 8:7 | 6:7, 6:9 |
| window_0_120 vs speed_only | 7 | 4 | 3 | 4:6, 6:0, 6:3, 6:6, 8:4, 8:7, 8:10 | 6:7, 6:9, 6:10, 8:9 |
| window_0_180 vs speed_only | 5 | 3 | 2 | 4:6, 6:0, 6:6, 8:3, 8:6 | 6:7, 6:8, 6:9 |
| window_0_220 vs speed_only | 4 | 4 | 0 | 4:6, 6:0, 6:6, 8:10 | 6:7, 6:8, 6:9, 8:9 |
| window_80_240 vs speed_only | 3 | 5 | -2 | 6:0, 6:6, 8:3 | 4:10, 6:7, 6:9, 6:10, 8:8 |
| window_120_280 vs speed_only | 2 | 2 | 0 | 6:6, 8:10 | 4:10, 6:9 |
| window_160_240 vs speed_only | 2 | 3 | -1 | 6:6, 8:3 | 4:10, 6:9, 8:9 |
| window_240_320 vs speed_only | 2 | 1 | 1 | 8:3, 8:10 | 8:9 |

## 判读方式

- 如果某个窄窗口接近 `phase28d_window_0_250` 的成功率但 p50 更低，它就是更好的 duration proxy。
- 如果 prefix 窗口有效而中后段 atomic bucket 无效，说明敏感性主要来自 grasp-critical prefix。
- 如果不同 case 被不同窗口修复，就需要进入 layer x duration 或 task-conditioned duration guard。
