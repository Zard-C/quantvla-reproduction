# QuantVLA Phase 5 Quantized ATM/OHB LIBERO Rollout

Boundary: this is the matched simulator rollout for the fake-quantized `llm_dit_mlp + atm_ohb` student. It uses the same official LIBERO eval client as the accepted FP16 baseline. It is not a packed-int-kernel latency or memory benchmark.

## Status

Current state at snapshot `2026-06-04T11:23:00Z`: the matched quantized rollout is running on the RTX 5090 machine.

- Quantized server PID: `407691`
- Eval PID: `414689`
- Quantized server log: `/tmp/logs/quantvla_student_server_atm_ohb.log`
- Quantized server prepare JSON: `toy_quantvla/results/phase5_quantized_server_prepare_atm_ohb.json`
- Eval stdout log: `/tmp/logs/libero_eval_libero_10_quant_atm_ohb.stdout`
- Eval official log: `/tmp/logs/libero_eval_libero_10.log`
- FP16 baseline log backup: `/tmp/logs/libero_eval_libero_10_fp16_5trial_20260604_172710.log`
- Rollout directory: `/root/autodl-tmp/Isaac-GR00T-n1.5/examples/Libero/eval/rollouts/2026_06_04`
- Quantized rollout prefix observed so far: `2026_06_04-19_20_51`

Early sanity check:

- Episodes completed at snapshot: `1`
- Successes at snapshot: `1`
- First episode result: success on task `put both the alphabet soup and the tomato sauce in the basket`

Do not interrupt the run unless explicitly deciding to abort the quantized comparison.

## Quantized Server

Command:

```bash
cd /root/autodl-tmp/quantvla-reproduction
/root/autodl-tmp/envs/gr00t-libero-py310/bin/python toy_quantvla/quantized_inference_service.py \
  --config llm_dit_mlp \
  --mode atm_ohb \
  --output-json toy_quantvla/results/phase5_quantized_server_prepare_atm_ohb.json
```

The server loads the official GR00T `Gr00tPolicy`, collects teacher attention stats on the Phase 4 calibration observations, patches W4A8 fake quantization into `llm_dit_mlp`, collects student attention stats, installs ATM/OHB attention processors, then starts `RobotInferenceServer` on port `5555`.

Prepare-only result:

- Model: `/root/autodl-tmp/models/gr00t-n1.5-libero-long-posttrain`
- Dataset: `/root/autodl-tmp/datasets/libero_10_subset`
- Data config: `examples.Libero.custom_data_config:LiberoDataConfig`
- Denoising steps: `8`
- Config: `llm_dit_mlp`
- Mode: `atm_ohb`
- Weight bits: `4`
- Activation bits: `8`
- Activation scale: `absmax`
- Quantized modules: `116`
- Quantized groups: `dit_mlp_selected`, `llm_selected`
- ATM/OHB modules: `16`
- Prepare seconds: about `20.9`

Calibration dataset indices:

```text
115, 462, 632, 1063, 1273, 1482, 1823, 2034, 2406, 2536, 3053, 3198, 3492, 3824, 3980, 4299
```

Scale summary:

| scale | mean | min | max |
|---|---:|---:|---:|
| alpha | 1.21141 | 1.00084 | 1.34986 |
| beta | 1.17794 | 1.01553 | 1.34986 |
| alpha raw | 1.22870 | 1.00084 | 1.39882 |
| beta raw | 1.18788 | 1.01553 | 1.41846 |

## Eval

Command:

```bash
cd /root/autodl-tmp/Isaac-GR00T-n1.5/examples/Libero/eval
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
/root/autodl-tmp/envs/gr00t-libero-py310/bin/python run_libero_eval.py \
  --task-suite-name libero_10 \
  --num-trials-per-task 5 \
  --headless \
  --port 5555
```

This matches the accepted FP16 baseline protocol:

- Task suite: `libero_10`
- Trials per task: `5`
- Total episodes: `50`
- Denoising steps: `8`
- Headless EGL
- Official task order

## Progress Commands

```bash
pgrep -af "quantized_inference_service.py"
pgrep -af "run_libero_eval.py"
```

```bash
/root/autodl-tmp/envs/gr00t-libero-py310/bin/python -c "from pathlib import Path; import re; text=Path('/tmp/logs/libero_eval_libero_10.log').read_text(); print('episodes', len(re.findall(r'^Success: ', text, flags=re.M))); print('successes', len(re.findall(r'^Success: True$', text, flags=re.M))); print('failures', len(re.findall(r'^Success: False$', text, flags=re.M))); print('terminated_exceptions', text.count('Caught exception: executing action in terminated episode')); print('\\n'.join(text.splitlines()[-60:]))"
```

## Cleanup After Completion

After the 50-episode eval completes:

1. Parse per-task success rates from `/tmp/logs/libero_eval_libero_10.log`.
2. Save or summarize the final result in this document.
3. Stop the quantized server if still running.
4. Verify port `5555` returns `111`.
5. Commit and push the final report.

Do not commit rollout videos, model weights, datasets, generated checkpoints, or large tensor dumps.
