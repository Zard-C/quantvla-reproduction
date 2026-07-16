# Follow-up tactic selection

- summary: `toy_quantvla/results/phase43_n17_hybrid_bo_probe_10case_v1_summary.json`
- base tactics: `fp16 speed_only window_0_20 window_2_12`
- max tactics: `7`
- min speedup preference: `1.20x`

## Selected

```text
fp16 speed_only window_0_20 window_2_12 blocks0_3_window_0_20 blocks0_3 blocks0_3_window_2_12
```

## Candidate ranking

| tactic | success | speedup | repairs | regressions | speed feasible |
| --- | ---: | ---: | ---: | ---: | --- |
| `blocks0_3_window_0_20` | 9/10 | 0.97x | 0 | 0 | no |
| `blocks0_3` | 9/10 | 1.27x | 1 | 1 | yes |
| `blocks0_3_window_2_12` | 9/10 | 1.09x | 1 | 1 | no |
| `blocks16_31_window_2_12` | 7/10 | 1.06x | 1 | 3 | no |
| `blocks8_15_window_2_12` | 7/10 | 1.05x | 1 | 3 | no |

Selection rule: keep anchors, then add hybrid candidates with fewer paired regressions first; within the same regression count prefer speed-feasible, higher-success, faster tactics.
