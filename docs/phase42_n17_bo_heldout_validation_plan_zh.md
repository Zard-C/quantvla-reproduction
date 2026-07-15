# Phase42: N1.7 CLSG-BO held-out validation plan

## 目的

Phase40/41 已经把 N1.7 的早期 duration-window search 从人工候选推进到 BO/active search。Phase42 不继续在同一批 init `27/28/29` 上调参，而是切到新的 held-out init `30/31/32`，验证两个问题：

1. Phase40 的 behavior-first 候选 `window_0_20` 是否仍然低 regression。
2. Phase41/42 selector 找到的 speed-constrained 早期窄窗口，尤其 `window_2_12`、`window_4_9`、`window_6_11`，能否在新 init 上保留速度收益且不显著增加 closed-loop regression。

## 运行配置

- Tag prefix: `phase42_n17_bo_heldout_validation_15case_v1`
- Cases:

```text
tasks 0/1/4/6/8 x init 30/31/32 = 15 cases
```

- Tactics:

```text
fp16
speed_only
window_0_20
window_2_12
window_4_9
window_6_11
```

## 候选角色

| tactic | 角色 |
| --- | --- |
| `fp16` | reference baseline |
| `speed_only` | aggressive speed anchor |
| `window_0_20` | Phase40 behavior-first winner: `14/15`, `0` regression vs FP16 |
| `window_2_12` | Phase41 observed speed-constrained winner: `14/15`, `1.27x`, `2` repairs / `1` regression |
| `window_4_9` | corrected combined BO top speed-constrained proposal |
| `window_6_11` | corrected combined BO top behavior/speed proposal |

## 判读标准

Phase42 是 held-out validation，不是继续调参。判读顺序：

1. 先看 paired regressions vs FP16。
2. 再看 success / repair。
3. 最后看 p50 speedup。

如果某个 tactic 比 FP16 有更多成功但 regression 很高，它只能算 basin redistribution，不算稳健 deployment candidate。如果 `window_2_12`、`window_4_9` 或 `window_6_11` 在 held-out 上接近 `window_0_20` 的 regression，同时明显更快，就说明 BO 找到了更好的 speed-risk point。

## 预期产物

- Summary:

```text
toy_quantvla/results/phase42_n17_bo_heldout_validation_15case_v1_summary.json
```

- Report:

```text
docs/phase42_n17_bo_heldout_validation_report_zh.md
```
