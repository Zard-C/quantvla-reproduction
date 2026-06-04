# QuantVLA Phase 4 Real-Data Validation Plan

Boundary: this phase uses real LeRobot LIBERO validation observations for offline teacher/student action-drift validation. It is not yet a LIBERO simulator success-rate benchmark.

## Why

Phase 3 used synthetic observations. That was enough to test GR00T loading, module selection, fake W4A8, smoothing, ATM, and OHB, but it cannot validate the activation distribution seen by the LIBERO checkpoint. Phase 4 replaces synthetic observations with real dataset samples and keeps the comparison controlled by matching GR00T denoising seeds.

## Dataset

For the downloaded checkpoint `/root/autodl-tmp/models/gr00t-n1.5-libero-long-posttrain`, the visible Hugging Face dataset is:

```bash
IPEC-COMMUNITY/libero_10_no_noops_1.0.0_lerobot
```

The GR00T README omits the `1.0.0` component for LIBERO-10; that older name returned 404 on 2026-06-04.

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
source /etc/network_turbo
/root/autodl-tmp/envs/gr00t-py312-cu128/bin/python toy_quantvla/download_lerobot_subset.py \
  --repo-id IPEC-COMMUNITY/libero_10_no_noops_1.0.0_lerobot \
  --local-dir /root/autodl-tmp/datasets/libero_10_subset \
  --start-episode 0 \
  --num-episodes 4 \
  --modality-json /root/autodl-tmp/Isaac-GR00T-n1.5/examples/Libero/modality.json
```

Use the official Hugging Face endpoint with `/etc/network_turbo`. Do not send a Hugging Face token to a third-party mirror.

## Validation Command

```bash
cd /root/autodl-tmp/quantvla-reproduction
/root/autodl-tmp/envs/gr00t-py312-cu128/bin/python toy_quantvla/phase4_real_data_validation.py \
  --isaac-root /root/autodl-tmp/Isaac-GR00T-n1.5 \
  --compat-stubs toy_quantvla/compat_stubs \
  --model-path /root/autodl-tmp/models/gr00t-n1.5-libero-long-posttrain \
  --dataset-path /root/autodl-tmp/datasets/libero_10_subset \
  --video-backend torchvision_av \
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

For this 5090 environment, `torchcodec` was not installed and `decord`/`opencv` could not decode the downloaded AV1 videos. `torchvision_av` successfully decoded the validation subset.

## Next Gate

The denoising-steps 8, 8-sample follow-up is recorded in:

```bash
docs/phase4_real_data_validation_d8_n8.md
toy_quantvla/results/phase4_real_data_validation_d8_n8.json
```

For the intended `llm_dit_mlp` selected QuantVLA scope, `atm_ohb` improves real-data action drift versus `none`:

| mode | NMSE mean | rel RMSE mean | cosine mean |
|---|---:|---:|---:|
| none | 0.00508872 | 0.0640329 | 0.997771 |
| atm_ohb | 0.00301077 | 0.04486 | 0.998726 |

This clears the offline Phase 4 gate for a small LIBERO simulator smoke rollout. Do not treat this as benchmark evidence. First run a FP16 official-server baseline to validate the simulator environment, then add the quantized student path.

Current blocker: `/root/autodl-tmp/envs/gr00t-py312-cu128` does not import `libero`, so the simulator environment is not ready in that venv. Phase 5 should use a LIBERO-capable Python environment with real `pytorch3d` instead of the Phase 3/4 import stub.
