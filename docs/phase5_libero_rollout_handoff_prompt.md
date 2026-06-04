# QuantVLA Phase 5 LIBERO Rollout Handoff Prompt

You are working in `/root/autodl-tmp/quantvla-reproduction`.

Do not commit model weights, datasets, generated checkpoints, rollout videos, or large tensor dumps.

Read first:

```bash
sed -n '1,260p' docs/phase4_real_data_validation.md
sed -n '1,220p' docs/phase4_real_data_validation_d8_n8.md
sed -n '1,220p' /root/autodl-tmp/Isaac-GR00T-n1.5/examples/Libero/README.md
sed -n '1,320p' /root/autodl-tmp/Isaac-GR00T-n1.5/examples/Libero/eval/run_libero_eval.py
```

## Status

Phase 4 cleared the offline real-data action-drift gate for a small simulator smoke rollout. The strongest current evidence is the denoising-steps 8, 8-observation run:

| config | mode | NMSE mean | rel RMSE mean | cosine mean |
|---|---|---:|---:|---:|
| `llm_dit_mlp` | none | 0.00508872 | 0.0640329 | 0.997771 |
| `llm_dit_mlp` | `atm_ohb` | 0.00301077 | 0.04486 | 0.998726 |

Important caveat: `dit_mlp_only + atm_ohb` regressed on this same run (`NMSE mean 0.00982088`, `max abs diff 0.971985`). Treat ATM/OHB as a configuration-specific balancing path, not a universally safe per-module improvement.

## Environment Gate

Current blocker:

```text
/root/autodl-tmp/envs/gr00t-py312-cu128/bin/python -c "import libero, robosuite"
ModuleNotFoundError: No module named 'libero'
```

Before running Phase 5, create or select a LIBERO-capable environment. The Phase 3/4 Python 3.12 setup uses a local `pytorch3d.transforms` import stub, which is acceptable for offline LIBERO smoke paths because the checkpoint metadata has `rotation_type: null`. A simulator rollout should use the official LIBERO/robosuite stack and real `pytorch3d`, preferably Python 3.10 as recommended by GR00T.

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
  --task_suite_name libero_10 \
  --num_trials_per_task 1 \
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
- collect teacher/student attention stats on calibration observations
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
