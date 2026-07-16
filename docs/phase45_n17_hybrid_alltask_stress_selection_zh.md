# Follow-up tactic selection

- summary: `toy_quantvla/results/phase44_n17_hybrid_heldout_15case_v1_summary.json`
- base tactics: `fp16 speed_only window_0_20 window_2_12`
- max tactics: `6`
- min speedup preference: `1.15x`

## Selected

```text
fp16 speed_only window_0_20 window_2_12 blocks0_3_window_2_12 blocks0_3
```

## Candidate ranking

| tactic | success | speedup | repairs | regressions | speed feasible |
| --- | ---: | ---: | ---: | ---: | --- |
| `blocks0_3_window_2_12` | 13/15 | 1.03x | 3 | 1 | no |
| `blocks0_3` | 12/15 | 1.24x | 3 | 2 | yes |
| `blocks0_3_window_0_20` | 9/15 | 1.04x | 1 | 3 | no |

Selection rule: keep anchors, then add hybrid candidates with fewer paired regressions first; within the same regression count prefer speed-feasible, higher-success, faster tactics.
