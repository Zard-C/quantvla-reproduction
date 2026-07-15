# Phase 39 A2: same-observation real drift sequence

这个文件把 same-observation `compiled - eager` drift 保存成逐 policy step 的 delta 序列，用于 replay-style scaling：

```text
action_t = action_ref_t + lambda * eta_t
```

- Variant: `speed_only`
- Online drift JSON: `toy_quantvla/results/phase39_stageA2_sameobs_realdrift_speedonly_seed20260613_v1_sameobs_online_drift.json`
- JSON artifact: `toy_quantvla/results/phase39_stageA2_sameobs_realdrift_speedonly_seed20260613_v1_sameobs_real_drift_sequences.json`

| case | direction | window | status | steps | mean L2 | p50 L2 | p95 L2 | max L2 | sequence |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 4:9 | real_speed_only_sameobs_sequence | full | complete | 224 | 0.0001557 | 0.001735 | 0.003651 | 0.005351 | toy_quantvla/results/phase39_stageA2_sameobs_realdrift_speedonly_seed20260613_v1_sameobs_real_drift_sequences/case_t4_i9_real_speed_only_sameobs_sequence_win_full.json |
| 4:9 | real_speed_only_sameobs_sequence | early | complete | 75 | 0.0002734 | 0.001623 | 0.003343 | 0.004684 | toy_quantvla/results/phase39_stageA2_sameobs_realdrift_speedonly_seed20260613_v1_sameobs_real_drift_sequences/case_t4_i9_real_speed_only_sameobs_sequence_win_early.json |
| 6:8 | real_speed_only_sameobs_sequence | full | complete | 649 | 0.0001585 | 0.001409 | 0.003068 | 0.07739 | toy_quantvla/results/phase39_stageA2_sameobs_realdrift_speedonly_seed20260613_v1_sameobs_real_drift_sequences/case_t6_i8_real_speed_only_sameobs_sequence_win_full.json |
| 6:8 | real_speed_only_sameobs_sequence | early | complete | 200 | 0.0001552 | 0.001458 | 0.003123 | 0.006682 | toy_quantvla/results/phase39_stageA2_sameobs_realdrift_speedonly_seed20260613_v1_sameobs_real_drift_sequences/case_t6_i8_real_speed_only_sameobs_sequence_win_early.json |
