# QuantVLA Phase 5 Quantized ATM/OHB LIBERO Rollout

Boundary: this is the matched simulator rollout for the fake-quantized `llm_dit_mlp + atm_ohb` student. It uses the same official LIBERO eval client as the accepted FP16 baseline. It is not a packed-int-kernel latency or memory benchmark.

## Status

Final state at snapshot `2026-06-04T12:10:00Z`: the matched quantized rollout completed all `50` episodes on the RTX 5090 machine.

- Quantized server PID: `407691`
- Eval PID: `414689`
- Quantized server log: `/tmp/logs/quantvla_student_server_atm_ohb.log`
- Quantized server prepare JSON: `toy_quantvla/results/phase5_quantized_server_prepare_atm_ohb.json`
- Eval stdout log: `/tmp/logs/libero_eval_libero_10_quant_atm_ohb.stdout`
- Eval official log: `/tmp/logs/libero_eval_libero_10.log`
- FP16 baseline log backup: `/tmp/logs/libero_eval_libero_10_fp16_5trial_20260604_172710.log`
- Rollout directory: `/root/autodl-tmp/Isaac-GR00T-n1.5/examples/Libero/eval/rollouts/2026_06_04`
- Quantized rollout prefix: `2026_06_04-19_20_51`

Final result:

- Episodes completed: `50/50`
- Successes: `38/50`
- Success rate: `76.0%`
- Failures: `12`
- `executing action in terminated episode` exceptions: `12`
- FP16 baseline: `38/50` (`76.0%`)
- Aggregate delta vs FP16: `0`

The aggregate success rate matches the accepted FP16 baseline exactly, but the per-task distribution differs. The quantized run improves task ids `1` and `2`, regresses task ids `6`, `7`, and `8`, and matches the remaining tasks.

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

Per-task result:

| task id | quant successes / trials | quant rate | FP16 successes / trials | delta | task |
|---:|---:|---:|---:|---:|---|
| 0 | 5/5 | 100% | 5/5 | 0 | put both the alphabet soup and the tomato sauce in the basket |
| 1 | 4/5 | 80% | 3/5 | +1 | put both the cream cheese box and the butter in the basket |
| 2 | 5/5 | 100% | 3/5 | +2 | turn on the stove and put the moka pot on it |
| 3 | 5/5 | 100% | 5/5 | 0 | put the black bowl in the bottom drawer of the cabinet and close it |
| 4 | 4/5 | 80% | 4/5 | 0 | put the white mug on the left plate and put the yellow and white mug on the right plate |
| 5 | 5/5 | 100% | 5/5 | 0 | pick up the book and place it in the back compartment of the caddy |
| 6 | 2/5 | 40% | 3/5 | -1 | put the white mug on the plate and put the chocolate pudding to the right of the plate |
| 7 | 3/5 | 60% | 4/5 | -1 | put both the alphabet soup and the cream cheese box in the basket |
| 8 | 0/5 | 0% | 1/5 | -1 | put both moka pots on the stove |
| 9 | 5/5 | 100% | 5/5 | 0 | put the yellow and white mug in the microwave and close it |

## Interpretation

The `llm_dit_mlp + atm_ohb` student preserves the FP16 simulator success rate on this matched `libero_10` 5-trial run. It does not establish a packed-kernel acceleration result, but it does clear the key method-level simulator gate: selective W4A8 fake quantization plus ATM/OHB did not reduce aggregate success relative to the accepted FP16 baseline.

The distribution shift is important:

- The student improves task id `2` by `+2/5`, suggesting the quantized and balanced policy can change rollout behavior in a beneficial direction for some initial states.
- The student loses the already-fragile task id `8`, going from FP16 `1/5` to quantized `0/5`.
- Aggregate equality hides task-specific movement, so future reports should keep per-task success rates and not only the total.

## Progress Commands

```bash
pgrep -af "quantized_inference_service.py"
pgrep -af "run_libero_eval.py"
```

```bash
/root/autodl-tmp/envs/gr00t-libero-py310/bin/python -c "from pathlib import Path; import re; text=Path('/tmp/logs/libero_eval_libero_10.log').read_text(); print('episodes', len(re.findall(r'^Success: ', text, flags=re.M))); print('successes', len(re.findall(r'^Success: True$', text, flags=re.M))); print('failures', len(re.findall(r'^Success: False$', text, flags=re.M))); print('terminated_exceptions', text.count('Caught exception: executing action in terminated episode')); print('\\n'.join(text.splitlines()[-60:]))"
```

## Cleanup After Completion

Cleanup was completed after the 50-episode eval:

- `run_libero_eval.py` exited after all `50` episodes.
- The quantized server was stopped with `SIGTERM`.
- Port `5555` returned `111`.
- GPU memory returned to about `2 MiB`.

Do not commit rollout videos, model weights, datasets, generated checkpoints, or large tensor dumps.
