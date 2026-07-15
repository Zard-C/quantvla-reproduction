# Phase 39 A2: same-observation real drift sequence

这个文件把 same-observation `compiled - eager` drift 保存成逐 policy step 的 delta 序列，用于 replay-style scaling：

```text
action_t = action_ref_t + lambda * eta_t
```

- Variant: `combo_blocks0_3_window0_120`
- Online drift JSON: `toy_quantvla/results/phase39_stageA2_sameobs_realdrift_combo_blocks0_3_window0_120_seed20260613_v1_sameobs_online_drift.json`
- JSON artifact: `toy_quantvla/results/phase39_stageA2_sameobs_realdrift_combo_blocks0_3_window0_120_seed20260613_v1_sameobs_real_drift_sequences.json`

| case | direction | window | status | steps | mean L2 | p50 L2 | p95 L2 | max L2 | sequence |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 4:9 | real_combo_blocks0_3_window0_120_sameobs_sequence | full | complete | 224 | 0.0001813 | 0 | 0.002847 | 0.004592 | toy_quantvla/results/phase39_stageA2_sameobs_realdrift_combo_blocks0_3_window0_120_seed20260613_v1_sameobs_real_drift_sequences/case_t4_i9_real_combo_blocks0_3_window0_120_sameobs_sequence_win_full.json |
| 4:9 | real_combo_blocks0_3_window0_120_sameobs_sequence | early | complete | 75 | 0 | 0 | 0 | 0 | toy_quantvla/results/phase39_stageA2_sameobs_realdrift_combo_blocks0_3_window0_120_seed20260613_v1_sameobs_real_drift_sequences/case_t4_i9_real_combo_blocks0_3_window0_120_sameobs_sequence_win_early.json |
| 6:8 | real_combo_blocks0_3_window0_120_sameobs_sequence | full | complete | 649 | 0.0002145 | 0.001167 | 0.002568 | 0.007381 | toy_quantvla/results/phase39_stageA2_sameobs_realdrift_combo_blocks0_3_window0_120_seed20260613_v1_sameobs_real_drift_sequences/case_t6_i8_real_combo_blocks0_3_window0_120_sameobs_sequence_win_full.json |
| 6:8 | real_combo_blocks0_3_window0_120_sameobs_sequence | early | complete | 200 | 0.0003583 | 0 | 0.002153 | 0.005478 | toy_quantvla/results/phase39_stageA2_sameobs_realdrift_combo_blocks0_3_window0_120_seed20260613_v1_sameobs_real_drift_sequences/case_t6_i8_real_combo_blocks0_3_window0_120_sameobs_sequence_win_early.json |
