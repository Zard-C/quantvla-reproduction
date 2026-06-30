# Phase28D Duration-Window 结果分析

数据源: `toy_quantvla/results/phase28D_duration_window_33case_v1_summary.json`。

这份报告只分析 Phase28D 的 duration-window fallback 结果，不启动新的 rollout。它的目的有两个：第一，把 `repair/regress` 和执行步数差异说清楚；第二，为下一轮 finer duration proxy 独立出候选设计。

## 总览

| run | success | server p50 ms | speedup | server eager frac | mean steps | median success steps | median failure steps |
| --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | 19/33 | 156.22 | 1.00x | - | 643.79 | 264 | 991.00 |
| speed_only | 16/33 | 70.20 | 2.23x | - | 674.30 | 242.50 | 991 |
| window_80_220 | 14/33 | 75.11 | 2.08x | 0.20 | 695.09 | 228.00 | 991 |
| window_120_320 | 16/33 | 78.09 | 2.00x | 0.26 | 671.33 | 241.50 | 991 |
| window_180_420 | 16/33 | 78.82 | 1.98x | 0.26 | 667.88 | 238.00 | 991 |
| window_0_250 | 18/33 | 88.75 | 1.76x | 0.38 | 641.58 | 254.50 | 991 |

失败 episode 基本都是 `991` 步 horizon failure，所以成功率变化会直接改变平均执行步数。`window_0_250` 的平均步数接近 baseline，是因为它把几个 speed-only horizon failure 拉回了成功；但它的 eager 请求比例也最高，server p50 从 speed-only 的约 `70 ms` 升到约 `89 ms`。

## Repair / Regress

| pair | repair | regress | net | repair cases | regress cases |
| --- | --- | --- | --- | --- | --- |
| window_80_220 vs baseline | 0 | 5 | -5 | - | 4:6 (S245->F991), 4:10 (S212->F991), 6:10 (S525->F991), 8:7 (S702->F991), 8:10 (S458->F991) |
| window_120_320 vs baseline | 2 | 5 | -3 | 6:7 (F991->S297), 8:9 (F991->S472) | 4:6 (S245->F991), 4:10 (S212->F991), 6:0 (S210->F991), 8:7 (S702->F991), 8:10 (S458->F991) |
| window_180_420 vs baseline | 2 | 5 | -3 | 6:9 (F991->S244), 8:9 (F991->S740) | 4:6 (S245->F991), 6:0 (S210->F991), 6:6 (S224->F991), 8:3 (S986->F991), 8:7 (S702->F991) |
| window_0_250 vs baseline | 1 | 2 | -1 | 8:9 (F991->S534) | 8:3 (S986->F991), 8:10 (S458->F991) |
| window_80_220 vs speed_only | 3 | 5 | -2 | 6:0 (F991->S218), 6:6 (F991->S212), 8:3 (F991->S678) | 4:10 (S244->F991), 6:7 (S261->F991), 6:9 (S241->F991), 6:10 (S225->F991), 8:9 (S624->F991) |
| window_120_320 vs speed_only | 2 | 2 | 0 | 6:6 (F991->S398), 8:3 (F991->S733) | 4:10 (S244->F991), 6:9 (S241->F991) |
| window_180_420 vs speed_only | 1 | 1 | 0 | 8:10 (F991->S569) | 6:7 (S261->F991) |
| window_0_250 vs speed_only | 4 | 2 | 2 | 4:6 (F991->S245), 6:0 (F991->S210), 6:6 (F991->S224), 8:7 (F991->S687) | 6:7 (S261->F991), 6:9 (S241->F991) |

最重要的对照是 `speed_only`。`window_0_250` 相比 speed-only 修复 `4:6, 6:0, 6:6, 8:7`，同时打坏 `6:7, 6:9`，净收益 `+2`。这说明 early duration fallback 有信息量，但它不是单调安全操作。

一个合理解释是：`0-250` 覆盖了接近物体、对齐、接触、夹爪闭合和初始搬起这些高敏感阶段。对 manipulation policy 来说，这些步骤决定了后续轨迹是否进入正确 basin；一旦物体已经稳定抓起，后续很多 step 更接近带约束的搬运/放置，误差敏感性可能下降。因此 `0-250` 的收益不一定来自“早期”本身，而可能来自它刚好覆盖了 grasp-critical duration。

相比 FP16 baseline，`window_0_250` 只差 `-1`：它保留了 speed-only 在 `8:9` 上的 beneficial branch，但仍然打坏 `8:3, 8:10`。这提醒我们，目标不应该只是“复制 baseline”，而应该区分 harmful regression 和 beneficial branch。

## 执行步数视角

| case | category | baseline | speed | 80-220 | 120-320 | 180-420 | 0-250 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 4:6 | harmful_speed_regression | S245 | F991 | F991 | F991 | F991 | S245 |
| 4:10 | both_success | S212 | S244 | F991 | F991 | S205 | S212 |
| 6:0 | harmful_speed_regression | S210 | F991 | S218 | F991 | F991 | S210 |
| 6:6 | harmful_speed_regression | S224 | F991 | S212 | S398 | F991 | S224 |
| 6:7 | beneficial_speed_branch | F991 | S261 | F991 | S297 | F991 | F991 |
| 6:9 | beneficial_speed_branch | F991 | S241 | F991 | F991 | S244 | F991 |
| 6:10 | both_success | S525 | S225 | F991 | S227 | S229 | S522 |
| 8:3 | harmful_speed_regression | S986 | F991 | S678 | S733 | F991 | F991 |
| 8:7 | harmful_speed_regression | S702 | F991 | F991 | F991 | F991 | S687 |
| 8:9 | beneficial_speed_branch | F991 | S624 | F991 | S472 | S740 | S534 |
| 8:10 | harmful_speed_regression | S458 | F991 | F991 | F991 | S569 | F991 |

几个关键模式：

- `4:6, 6:0, 6:6, 8:7` 是 speed-only 的 harmful regression。`window_0_250` 能修复它们，说明早期控制段对这些失败 basin 很关键。
- `8:3` 很反直觉：`80-220` 和 `120-320` 能救，`0-250` 反而失败。这说明不是越早、越宽越好；过早 eager 会改变它进入的 branch。
- `8:10` 只被 `180-420` 修复，说明 late manipulation / settling 段也有独立敏感性。
- `6:7, 6:9, 8:9` 是 speed-only 的 beneficial branch。一个只追求 baseline mimicry 的 proxy 会倾向于把它们拉回 baseline failure，这是不对的。

## Window 命中比例

| window | server eager frac | approx step frac all | approx step frac success | approx step frac failure |
| --- | --- | --- | --- | --- |
| window_80_220 | 0.20 | 0.30 | 0.52 | 0.14 |
| window_120_320 | 0.26 | 0.32 | 0.46 | 0.20 |
| window_180_420 | 0.26 | 0.27 | 0.29 | 0.24 |
| window_0_250 | 0.38 | 0.54 | 0.79 | 0.25 |

`window_0_250` 对成功 episode 的平均覆盖比例接近 `0.79`，所以它更像“早期大段 eager policy”，不是轻量 guard。`80-220/120-320/180-420` 的 server eager fraction 约 `0.20-0.26`，更便宜，但目前 repair/regress 平衡不够好。

## 对 finer duration proxy 的启发

- Harmful speed regression set: `4:6, 6:0, 6:6, 8:3, 8:7, 8:10`
- Beneficial speed branch set: `6:7, 6:9, 8:9`
- Stable-success risk set: `4:10, 6:10`

下一轮不应该只问“哪个窗口更像 FP16”，而应该问“哪个窗口能修复 harmful regressions，同时保留 beneficial branches”。一个更合适的 proxy score 可以写成：

$$
Score(w)=|H_{repair}(w)|-|B_{regress}(w)|-\alpha |S_{regress}(w)|-\lambda \Delta latency(w)
$$

其中 `H` 是 baseline 成功但 speed-only 失败的 harmful set，`B` 是 baseline 失败但 speed-only 成功的 beneficial set，`S` 是 baseline 和 speed-only 都成功的稳定成功集合。`Delta latency` 用 server p50 或 eager fraction 近似。

建议把 finer duration proxy 独立成 Phase29：先跑少量 atomic windows 定位敏感时间桶，再跑组合窗口。优先级如下：

1. Atomic buckets: `0:80,80:160,160:240,240:320,320:500`。用来判断敏感性到底来自 initial approach、pre-grasp/contact、grasp/lift，还是 late transport/settling。
2. Grasp-centered prefixes/combinations: `0:120,0:180,0:220,0:250,80:240,120:280`。用来验证 broad `0:250` 的收益是否能用更窄、更贴近抓取阶段的窗口保留。
3. 如果 Phase29 仍然存在强烈 case-specific 交叉，就进入 layer × duration 的二阶 proxy，而不是继续只调一维 duration。

对应独立 runner: `toy_quantvla/run_phase29_finer_duration_proxy.sh`。
