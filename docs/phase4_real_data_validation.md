# QuantVLA Phase 4 Real-Data Validation Plan

Boundary: this phase uses real LeRobot LIBERO validation observations for offline teacher/student action-drift validation. It is not yet a LIBERO simulator success-rate benchmark.

## Why

Phase 3 used synthetic observations. That was enough to test GR00T loading, module selection, fake W4A8, smoothing, ATM, and OHB, but it cannot validate the activation distribution seen by the LIBERO checkpoint. Phase 4 replaces synthetic observations with real dataset samples and keeps the comparison controlled by matching GR00T denoising seeds.

## Dataset

For the downloaded checkpoint `/root/autodl-tmp/models/gr00t-n1.5-libero-long-posttrain`, the matching GR00T README points to:

```bash
IPEC-COMMUNITY/libero_10_no_noops_lerobot
```

The GR00T LIBERO `modality.json` must be present at:

```bash
<dataset>/meta/modality.json
```

The two video keys are:

- `observation.images.image`
- `observation.images.wrist_image`

## Minimal Download

If HF auth/proxy is available on the 5090, download only a few episodes first:

```bash
cd /root/autodl-tmp/quantvla-reproduction
/root/autodl-tmp/envs/gr00t-py312-cu128/bin/python toy_quantvla/download_lerobot_subset.py \
  --repo-id IPEC-COMMUNITY/libero_10_no_noops_lerobot \
  --local-dir /root/autodl-tmp/datasets/libero_10_subset \
  --start-episode 0 \
  --num-episodes 4 \
  --modality-json /root/autodl-tmp/Isaac-GR00T-n1.5/examples/Libero/modality.json
```

If the mirror works, add:

```bash
--hf-endpoint https://hf-mirror.com
```

## Validation Command

```bash
cd /root/autodl-tmp/quantvla-reproduction
/root/autodl-tmp/envs/gr00t-py312-cu128/bin/python toy_quantvla/phase4_real_data_validation.py \
  --isaac-root /root/autodl-tmp/Isaac-GR00T-n1.5 \
  --compat-stubs toy_quantvla/compat_stubs \
  --model-path /root/autodl-tmp/models/gr00t-n1.5-libero-long-posttrain \
  --dataset-path /root/autodl-tmp/datasets/libero_10_subset \
  --video-backend torchcodec \
  --denoising-steps 1 \
  --num-observations 8 \
  --sample-stride 50 \
  --configs llm_only,dit_mlp_only,llm_dit_mlp \
  --activation-scale-modes absmax \
  --smoothing-alpha none \
  --atm-ohb-modes none,ohb,atm,atm_ohb
```

## Metrics

The main metric is teacher/student action-chunk drift:

- NMSE
- relative RMSE
- cosine
- max absolute difference
- per-action-key metrics in the JSON

The script also reports teacher-vs-demonstration action drift as a diagnostic, but that is not a success-rate metric.

## Next Gate

If real-data offline drift is acceptable, run the same validation with more samples and denoising steps 8. Only then move to LIBERO simulator rollouts for success rate.
