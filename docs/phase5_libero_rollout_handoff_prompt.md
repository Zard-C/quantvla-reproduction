# QuantVLA Phase 5 LIBERO Rollout Handoff Prompt

You are working in `/root/autodl-tmp/quantvla-reproduction`.

Do not commit model weights, datasets, generated checkpoints, rollout videos, or large tensor dumps.

Read first:

```bash
sed -n '1,260p' docs/phase4_real_data_validation.md
sed -n '1,220p' docs/phase4_real_data_validation_d8_n8.md
sed -n '1,220p' docs/phase4_real_data_validation_d8_n8_identity.md
sed -n '1,220p' docs/phase4_real_data_validation_d8_cal16_eval32.md
sed -n '1,220p' docs/phase4_real_data_validation_d8_cal16_eval128_random.md
sed -n '1,260p' docs/phase4_real_data_validation_d8_cal16_eval128_random_regressions.md
sed -n '1,220p' /root/autodl-tmp/Isaac-GR00T-n1.5/examples/Libero/README.md
sed -n '1,320p' /root/autodl-tmp/Isaac-GR00T-n1.5/examples/Libero/eval/run_libero_eval.py
```

## Status

Phase 4 cleared the offline real-data action-drift gate for a small simulator smoke rollout. The strongest current evidence is the denoising-steps 8 random held-out run: 16 calibration observations from episodes 0-15 and 128 evaluation observations from episodes 16-63, sampled with fixed seed `260204`.

| run | config | mode | NMSE mean | rel RMSE mean | cosine mean |
|---|---|---|---:|---:|---:|
| d8 n8 | `llm_dit_mlp` | none | 0.00508872 | 0.0640329 | 0.997771 |
| d8 n8 | `llm_dit_mlp` | identity | 0.00508872 | 0.0640329 | 0.997771 |
| d8 n8 | `llm_dit_mlp` | `atm_ohb` | 0.00301077 | 0.04486 | 0.998726 |
| d8 cal16/eval32 | `llm_dit_mlp` | none | 0.00294977 | 0.0500502 | 0.998968 |
| d8 cal16/eval32 | `llm_dit_mlp` | identity | 0.00294977 | 0.0500502 | 0.998968 |
| d8 cal16/eval32 | `llm_dit_mlp` | `atm_ohb` | 0.00199345 | 0.0407359 | 0.999044 |
| d8 cal16/eval128 random | `llm_dit_mlp` | none | 0.0178962 | 0.0981458 | 0.992492 |
| d8 cal16/eval128 random | `llm_dit_mlp` | identity | 0.0178962 | 0.0981458 | 0.992492 |
| d8 cal16/eval128 random | `llm_dit_mlp` | `atm` | 0.0159471 | 0.0904183 | 0.993101 |
| d8 cal16/eval128 random | `llm_dit_mlp` | `ohb` | 0.0160919 | 0.0853958 | 0.992769 |
| d8 cal16/eval128 random | `llm_dit_mlp` | `atm_ohb` | 0.0153168 | 0.0838784 | 0.993048 |

The `identity` control installs the same custom DiT attention processor with `alpha = 1` and `beta = 1`. It matches `none` in every recorded run, so the observed ATM/OHB movement is not processor replacement drift. On the random held-out run, `atm` and `ohb` each improve the mean drift separately; `atm_ohb` is best on mean NMSE and mean relative RMSE.

Important caveat: `dit_mlp_only + atm_ohb` regressed on the d8 n8 run (`NMSE mean 0.00982088`, `max abs diff 0.971985`), with dataset index `30` as the outlier. Treat ATM/OHB as a configuration-specific balancing path for `llm_dit_mlp`, not a universally safe per-module improvement. In the random 128 held-out run, `atm_ohb` still worsens NMSE on 34/128 observations even though its mean delta is favorable; the top-seven regression analysis is in `docs/phase4_real_data_validation_d8_cal16_eval128_random_regressions.md`.

## Environment Gate

Use the Phase 5 environment prepared at:

```bash
/root/autodl-tmp/envs/gr00t-libero-py310
```

Readiness check passed with Python 3.10, `torch 2.8.0+cu128`, CUDA available on RTX 5090, `flash-attn 2.8.3`, real `pytorch3d.transforms`, `libero 0.1.1`, and `robosuite 1.4.0`. The Phase 3/4 Python 3.12 environment remains useful for offline probes but should not be used for simulator rollout.

## Objective

Run a very small LIBERO simulator success-rate smoke test before any full benchmark:

1. Validate the official FP16 GR00T server and LIBERO eval loop.
2. Record exact environment versions, command lines, task suite, trials, logs, and video paths.
3. Only after FP16 works, wire in the quantized student path for `llm_dit_mlp + atm_ohb`.
4. Keep offline action drift, simulator smoke success, and full benchmark success rate separate in the report.

## Official Baseline Commands

The local checkpoint is:

```bash
/root/autodl-tmp/models/gr00t-n1.5-libero-long-posttrain
```

The matching local GR00T source is:

```bash
/root/autodl-tmp/Isaac-GR00T-n1.5
```

Start the official FP16 inference server:

```bash
cd /root/autodl-tmp/Isaac-GR00T-n1.5
python scripts/inference_service.py \
  --model_path /root/autodl-tmp/models/gr00t-n1.5-libero-long-posttrain \
  --server \
  --data_config examples.Libero.custom_data_config:LiberoDataConfig \
  --denoising-steps 8 \
  --port 5555 \
  --embodiment-tag new_embodiment
```

In a second shell, run a tiny `libero_10` smoke evaluation:

```bash
cd /root/autodl-tmp/Isaac-GR00T-n1.5/examples/Libero/eval
python run_libero_eval.py \
  --task-suite-name libero_10 \
  --num-trials-per-task 1 \
  --headless \
  --port 5555
```

Notes:

- `run_libero_eval.py` logs to `/tmp/logs/libero_eval_<task_suite>.log`.
- Rollout MP4s are written under `./rollouts/<date>/` relative to `examples/Libero/eval`.
- The default script evaluates every task in the suite. If a faster one-task smoke is needed, patch a local copy or add a scoped CLI option rather than editing generated logs.

## Quantized Student Target

After the FP16 server baseline is verified, implement the student server path by reusing the Phase 4 mechanics:

- `patch_modules(policy.model, config_groups("llm_dit_mlp"), weight_bits=4, activation_bits=8, activation_scale="absmax")`
- collect teacher/student attention stats on calibration observations, initially reusing the Phase 4 random-run calibration source from episodes 0-15
- `compute_atm_ohb_scales(..., log_clamp=0.3)`
- `install_attention_processors(policy.model, mode="apply", scales=select_scales(scales, "atm_ohb"))`

Do not modify the official GR00T files in place unless the change is intentionally documented. Prefer a small wrapper script under `toy_quantvla/` that starts a `RobotInferenceServer` with the patched policy.

## Required Report

Create a Phase 5 report under `docs/` with:

- environment paths and package versions
- exact server and eval commands
- whether FP16 official-server rollout succeeded
- task suite, task count, trials per task, total episodes, successes
- log path and rollout video path summary
- quantized student configuration, calibration sample source, and whether ATM/OHB was active
- any simulator exceptions, GUI/headless issues, or robosuite/LIBERO version mismatches

Full benchmark claims require matched FP16 and quantized runs with the same task suite, same trial count, same initial states, and documented seeds or deterministic reset behavior.
