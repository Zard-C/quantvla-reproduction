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

For the cross-episode held-out follow-up, the local subset was expanded to 64 parquet episodes with the same script by setting `--num-episodes 64`.

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

The denoising-steps 8 follow-ups are recorded in:

```bash
docs/phase4_real_data_validation_d8_n8.md
toy_quantvla/results/phase4_real_data_validation_d8_n8.json
docs/phase4_real_data_validation_d8_n8_identity.md
toy_quantvla/results/phase4_real_data_validation_d8_n8_identity.json
docs/phase4_real_data_validation_d8_cal16_eval32.md
toy_quantvla/results/phase4_real_data_validation_d8_cal16_eval32.json
docs/phase4_real_data_validation_d8_cal16_eval128_random.md
toy_quantvla/results/phase4_real_data_validation_d8_cal16_eval128_random.json
docs/phase4_real_data_validation_d8_cal16_eval128_random_regressions.md
toy_quantvla/results/phase4_real_data_validation_d8_cal16_eval128_random_regressions.json
```

For the intended `llm_dit_mlp` selected QuantVLA scope, `atm_ohb` improves real-data action drift versus `none` on the original 8-observation d8 run:

| mode | NMSE mean | rel RMSE mean | cosine mean |
|---|---:|---:|---:|
| none | 0.00508872 | 0.0640329 | 0.997771 |
| identity | 0.00508872 | 0.0640329 | 0.997771 |
| atm_ohb | 0.00301077 | 0.04486 | 0.998726 |

The `identity` row installs the same custom DiT attention processor with `alpha = 1` and `beta = 1`. It exactly matches `none` in the printed metrics, so the `atm_ohb` gain is not explained by processor replacement drift.

A 16-calibration / 32-evaluation split inside the locally downloaded episode keeps the same direction for the main configuration:

| mode | NMSE mean | rel RMSE mean | cosine mean |
|---|---:|---:|---:|
| none | 0.00294977 | 0.0500502 | 0.998968 |
| identity | 0.00294977 | 0.0500502 | 0.998968 |
| atm_ohb | 0.00199345 | 0.0407359 | 0.999044 |

A stronger random held-out run uses 16 calibration observations sampled from episodes 0-15 and 128 evaluation observations sampled from episodes 16-63 with fixed seed `260204`. This run adds separate `atm` and `ohb` modes:

| mode | NMSE mean | NMSE change vs none | rel RMSE mean | rel RMSE change vs none | cosine mean |
|---|---:|---:|---:|---:|---:|
| none | 0.0178962 | 0.0% | 0.0981458 | 0.0% | 0.992492 |
| identity | 0.0178962 | 0.0% | 0.0981458 | 0.0% | 0.992492 |
| atm | 0.0159471 | -10.9% | 0.0904183 | -7.9% | 0.993101 |
| ohb | 0.0160919 | -10.1% | 0.0853958 | -13.0% | 0.992769 |
| atm_ohb | 0.0153168 | -14.4% | 0.0838784 | -14.5% | 0.993048 |

The random held-out run keeps the main direction while making the claim less dependent on contiguous frames. `atm` contributes more of the NMSE gain than `ohb`; `ohb` contributes more of the relative-RMSE gain; the combined `atm_ohb` mode is best on both mean NMSE and mean relative RMSE.

The regression pass compares each mode against `none` on the 128 held-out observations:

| mode | worse by NMSE | mean delta NMSE | mean delta rel RMSE | mean delta max abs |
|---|---:|---:|---:|---:|
| identity | 0/128 | 0 | 0 | 0 |
| atm | 37/128 | -0.00194913 | -0.00772748 | -0.0140478 |
| ohb | 24/128 | -0.00180429 | -0.01275 | -0.0119415 |
| atm_ohb | 34/128 | -0.00257947 | -0.0142674 | -0.019289 |

For the top seven `atm_ohb` regressions, the largest per-key NMSE deltas are distributed across `action.x`, `action.yaw`, `action.gripper`, `action.roll`, and `action.z`. Several large NMSE regressions occur on low-RMS rotation or gripper components, so the max-absolute-difference view is needed alongside NMSE to separate normalized error from physically larger deviations.

Important caveats:

- The earlier 16/32 split is frame-held-out inside local episode 0. The random 128 run is the current held-out offline gate because it samples evaluation observations from episodes 16-63 after calibrating on episodes 0-15.
- `dit_mlp_only + atm_ohb` still regresses on the original d8 n8 run. The worst observation is dataset index `30` (`NMSE 0.0770986`, `max_abs_diff 0.971985`). Since `identity` equals `none`, this looks like ATM/OHB over-compensation for that state rather than custom processor replacement error.
- These are offline teacher/student action-drift probes. They do not establish LIBERO simulator success rate, packed-int-kernel speedup, latency, memory, or throughput claims.

This clears the offline Phase 4 gate for a small LIBERO simulator smoke rollout, with `llm_dit_mlp + atm_ohb` as the main quantized student configuration. First run a FP16 official-server baseline to validate the simulator environment, then add the quantized student path.

Phase 5 environment note: `/root/autodl-tmp/envs/gr00t-libero-py310` has been prepared with Python 3.10, `torch 2.8.0+cu128`, `flash-attn 2.8.3`, real `pytorch3d.transforms`, `libero 0.1.1`, and `robosuite 1.4.0`. The older `/root/autodl-tmp/envs/gr00t-py312-cu128` environment remains suitable for Phase 3/4 offline probes but not for simulator rollout.
