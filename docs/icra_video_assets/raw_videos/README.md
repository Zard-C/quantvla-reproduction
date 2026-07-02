# Raw ICRA Video Assets

This directory keeps local copies of small rollout mp4 files for inspection.
They are copied from the 5090 rerun / recovery bundles.

## Quantization Matrix

Path:

```text
docs/icra_video_assets/raw_videos/quant_matrix/
```

Contents:

- 12 videos from `icra_quant_video_matrix_20260702_233443`
- cases: `8:7`, `4:10`, `0:3`, `8:0`
- modes: `fp16`, `none`, `atm_ohb`
- summary: `docs/icra_video_assets/raw_videos/quant_matrix/manifest.tsv`

Important note: this is a controlled rerun, not a byte-identical recreation of
the older Phase5 contact sheets. Q3/Q4 match the regression story; Q1/Q2 should
be captioned according to the rerun outcomes.

## Compile / Tactic

Path:

```text
docs/icra_video_assets/raw_videos/compile_tactic/
```

Contents:

- 6 recovered videos for C1/C2 compile/tactic qualitative cases
- each case has FP16 baseline, speed-only compile, and protected tactic videos

These files are useful for manual review and video overlay generation. Avoid
embedding raw local paths or account/server names in anonymized submission
materials.
