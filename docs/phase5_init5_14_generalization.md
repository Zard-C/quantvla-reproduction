# QuantVLA Phase 5 Init 5-14 Generalization

Boundary: this experiment extends the accepted Phase 5 simulator comparison to disjoint LIBERO initial states. It does not repeat init indices `0..4`; it evaluates init indices `5..14` for every `libero_10` task.

## Protocol

- Task suite: `libero_10`
- Init indices: `5, 6, 7, 8, 9, 10, 11, 12, 13, 14`
- Episodes per policy: `10 tasks * 10 init states = 100`
- FP16 and quantized student must use the same init indices.
- Denoising steps: `8`
- Headless EGL simulator setup.
- Eval wrapper: `toy_quantvla/libero_eval_init_range.py`

This is a generalization-style extension over different initial states, not repeated trials on identical init states.

## FP16 Result

Final state at snapshot `2026-06-04T16:49:00Z`: the FP16 init `5..14` run completed all `100` episodes on the RTX 5090 machine.

- Episodes completed: `100/100`
- Successes: `70/100`
- Success rate: `70.0%`
- Failures: `30`
- `executing action in terminated episode` exceptions: `30`
- Server log: `/tmp/logs/fp16_server_init5_14.log`
- Eval log: `/tmp/logs/libero_eval_libero_10_fp16_init5_14.log`
- Eval stdout: `/tmp/logs/libero_eval_libero_10_fp16_init5_14.stdout`
- Rollout prefix: `2026_06_04-22_18_42`

The FP16 server was stopped after completion. Port `5555` returned `111`, confirming it was released before launching the quantized run.

Per-task result:

| task id | successes / inits | success rate | task |
|---:|---:|---:|---|
| 0 | 8/10 | 80% | put both the alphabet soup and the tomato sauce in the basket |
| 1 | 8/10 | 80% | put both the cream cheese box and the butter in the basket |
| 2 | 9/10 | 90% | turn on the stove and put the moka pot on it |
| 3 | 10/10 | 100% | put the black bowl in the bottom drawer of the cabinet and close it |
| 4 | 4/10 | 40% | put the white mug on the left plate and put the yellow and white mug on the right plate |
| 5 | 10/10 | 100% | pick up the book and place it in the back compartment of the caddy |
| 6 | 3/10 | 30% | put the white mug on the plate and put the chocolate pudding to the right of the plate |
| 7 | 7/10 | 70% | put both the alphabet soup and the cream cheese box in the basket |
| 8 | 2/10 | 20% | put both moka pots on the stove |
| 9 | 9/10 | 90% | put the yellow and white mug in the microwave and close it |

Per-init result:

| init index | successes / tasks | success rate |
|---:|---:|---:|
| 5 | 5/10 | 50% |
| 6 | 8/10 | 80% |
| 7 | 7/10 | 70% |
| 8 | 7/10 | 70% |
| 9 | 7/10 | 70% |
| 10 | 7/10 | 70% |
| 11 | 8/10 | 80% |
| 12 | 6/10 | 60% |
| 13 | 9/10 | 90% |
| 14 | 6/10 | 60% |

Weak spots in this disjoint-init FP16 run are task id `8` (`2/10`), task id `6` (`3/10`), and task id `4` (`4/10`). This differs from the accepted init `0..4` baseline where task id `4` was `4/5`; the init `5..14` range exposes a harder slice for that task.

Failure list:

| task id | init index | task |
|---:|---:|---|
| 0 | 5 | put both the alphabet soup and the tomato sauce in the basket |
| 0 | 14 | put both the alphabet soup and the tomato sauce in the basket |
| 1 | 5 | put both the cream cheese box and the butter in the basket |
| 1 | 14 | put both the cream cheese box and the butter in the basket |
| 2 | 5 | turn on the stove and put the moka pot on it |
| 4 | 6 | put the white mug on the left plate and put the yellow and white mug on the right plate |
| 4 | 7 | put the white mug on the left plate and put the yellow and white mug on the right plate |
| 4 | 8 | put the white mug on the left plate and put the yellow and white mug on the right plate |
| 4 | 9 | put the white mug on the left plate and put the yellow and white mug on the right plate |
| 4 | 10 | put the white mug on the left plate and put the yellow and white mug on the right plate |
| 4 | 11 | put the white mug on the left plate and put the yellow and white mug on the right plate |
| 6 | 5 | put the white mug on the plate and put the chocolate pudding to the right of the plate |
| 6 | 7 | put the white mug on the plate and put the chocolate pudding to the right of the plate |
| 6 | 8 | put the white mug on the plate and put the chocolate pudding to the right of the plate |
| 6 | 9 | put the white mug on the plate and put the chocolate pudding to the right of the plate |
| 6 | 10 | put the white mug on the plate and put the chocolate pudding to the right of the plate |
| 6 | 12 | put the white mug on the plate and put the chocolate pudding to the right of the plate |
| 6 | 14 | put the white mug on the plate and put the chocolate pudding to the right of the plate |
| 7 | 11 | put both the alphabet soup and the cream cheese box in the basket |
| 7 | 12 | put both the alphabet soup and the cream cheese box in the basket |
| 7 | 14 | put both the alphabet soup and the cream cheese box in the basket |
| 8 | 5 | put both moka pots on the stove |
| 8 | 6 | put both moka pots on the stove |
| 8 | 7 | put both moka pots on the stove |
| 8 | 8 | put both moka pots on the stove |
| 8 | 9 | put both moka pots on the stove |
| 8 | 10 | put both moka pots on the stove |
| 8 | 12 | put both moka pots on the stove |
| 8 | 13 | put both moka pots on the stove |
| 9 | 12 | put the yellow and white mug in the microwave and close it |

Server command:

```bash
cd /root/autodl-tmp/Isaac-GR00T-n1.5
/root/autodl-tmp/envs/gr00t-libero-py310/bin/python scripts/inference_service.py \
  --model_path /root/autodl-tmp/models/gr00t-n1.5-libero-long-posttrain \
  --server \
  --data_config examples.Libero.custom_data_config:LiberoDataConfig \
  --denoising-steps 8 \
  --port 5555 \
  --embodiment-tag new_embodiment
```

Eval command:

```bash
cd /root/autodl-tmp/quantvla-reproduction
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
/root/autodl-tmp/envs/gr00t-libero-py310/bin/python toy_quantvla/libero_eval_init_range.py \
  --task-suite-name libero_10 \
  --init-start 5 \
  --num-inits 10 \
  --headless \
  --port 5555 \
  --log-file /tmp/logs/libero_eval_libero_10_fp16_init5_14.log
```

Startup sanity check:

```text
Task suite: libero_10
Init indices: [5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
```

## Quantized Result

Final state at snapshot `2026-06-04T20:45:53Z`: the quantized `llm_dit_mlp + atm_ohb` init `5..14` run completed all `100` episodes on the RTX 5090 machine.

- Policy: `llm_dit_mlp + atm_ohb`
- Fake quantization: W4A8, activation scale `absmax`
- Quantized modules: `116`
- Quantized groups: `dit_mlp_selected`, `llm_selected`
- ATM/OHB calibrated modules: `16`
- Calibration observations: `16`
- Episodes completed: `100/100`
- Successes: `76/100`
- Success rate: `76.0%`
- Failures: `24`
- `executing action in terminated episode` exceptions: `24`
- Delta vs FP16 init `5..14`: `+6/100`
- Server log: `/tmp/logs/quantvla_student_server_atm_ohb_init5_14.log`
- Eval log: `/tmp/logs/libero_eval_libero_10_quant_atm_ohb_init5_14.log`
- Eval stdout: `/tmp/logs/libero_eval_libero_10_quant_atm_ohb_init5_14.stdout`
- Prepare JSON: `toy_quantvla/results/phase5_quantized_server_prepare_atm_ohb_init5_14.json`
- Rollout prefix: `2026_06_05-00_52_16`

The quantized server was stopped after completion. Port `5555` returned `111`, confirming it was released.

Per-task comparison:

| task id | FP16 | quant | delta | task |
|---:|---:|---:|---:|---|
| 0 | 8/10 | 8/10 | +0 | put both the alphabet soup and the tomato sauce in the basket |
| 1 | 8/10 | 10/10 | +2 | put both the cream cheese box and the butter in the basket |
| 2 | 9/10 | 8/10 | -1 | turn on the stove and put the moka pot on it |
| 3 | 10/10 | 10/10 | +0 | put the black bowl in the bottom drawer of the cabinet and close it |
| 4 | 4/10 | 8/10 | +4 | put the white mug on the left plate and put the yellow and white mug on the right plate |
| 5 | 10/10 | 10/10 | +0 | pick up the book and place it in the back compartment of the caddy |
| 6 | 3/10 | 5/10 | +2 | put the white mug on the plate and put the chocolate pudding to the right of the plate |
| 7 | 7/10 | 7/10 | +0 | put both the alphabet soup and the cream cheese box in the basket |
| 8 | 2/10 | 3/10 | +1 | put both moka pots on the stove |
| 9 | 9/10 | 7/10 | -2 | put the yellow and white mug in the microwave and close it |

Per-init comparison:

| init index | FP16 | quant | delta |
|---:|---:|---:|---:|
| 5 | 5/10 | 8/10 | +3 |
| 6 | 8/10 | 8/10 | +0 |
| 7 | 7/10 | 6/10 | -1 |
| 8 | 7/10 | 6/10 | -1 |
| 9 | 7/10 | 9/10 | +2 |
| 10 | 7/10 | 7/10 | +0 |
| 11 | 8/10 | 9/10 | +1 |
| 12 | 6/10 | 7/10 | +1 |
| 13 | 9/10 | 8/10 | -1 |
| 14 | 6/10 | 8/10 | +2 |

Paired outcome changes over the exact same `100` task-init pairs:

| transition | count |
|---|---:|
| FP16 success, quant success | 62 |
| FP16 failure, quant failure | 16 |
| FP16 failure, quant success | 14 |
| FP16 success, quant failure | 8 |

The net `+6` aggregate gain comes from `14` FP16 failures repaired by quantization minus `8` FP16 successes lost by quantization. The biggest task-level improvement is task id `4` (`+4`), followed by task ids `1` and `6` (`+2` each). Task ids `2` and `9` regress, with task id `9` dropping from `9/10` to `7/10`.

Quantized failure list:

| task id | init index | task |
|---:|---:|---|
| 0 | 8 | put both the alphabet soup and the tomato sauce in the basket |
| 0 | 10 | put both the alphabet soup and the tomato sauce in the basket |
| 2 | 5 | turn on the stove and put the moka pot on it |
| 2 | 7 | turn on the stove and put the moka pot on it |
| 4 | 7 | put the white mug on the left plate and put the yellow and white mug on the right plate |
| 4 | 8 | put the white mug on the left plate and put the yellow and white mug on the right plate |
| 6 | 6 | put the white mug on the plate and put the chocolate pudding to the right of the plate |
| 6 | 7 | put the white mug on the plate and put the chocolate pudding to the right of the plate |
| 6 | 10 | put the white mug on the plate and put the chocolate pudding to the right of the plate |
| 6 | 12 | put the white mug on the plate and put the chocolate pudding to the right of the plate |
| 6 | 14 | put the white mug on the plate and put the chocolate pudding to the right of the plate |
| 7 | 8 | put both the alphabet soup and the cream cheese box in the basket |
| 7 | 11 | put both the alphabet soup and the cream cheese box in the basket |
| 7 | 12 | put both the alphabet soup and the cream cheese box in the basket |
| 8 | 5 | put both moka pots on the stove |
| 8 | 6 | put both moka pots on the stove |
| 8 | 7 | put both moka pots on the stove |
| 8 | 8 | put both moka pots on the stove |
| 8 | 10 | put both moka pots on the stove |
| 8 | 12 | put both moka pots on the stove |
| 8 | 13 | put both moka pots on the stove |
| 9 | 9 | put the yellow and white mug in the microwave and close it |
| 9 | 13 | put the yellow and white mug in the microwave and close it |
| 9 | 14 | put the yellow and white mug in the microwave and close it |

Quantized server command:

```bash
cd /root/autodl-tmp/quantvla-reproduction
/root/autodl-tmp/envs/gr00t-libero-py310/bin/python toy_quantvla/quantized_inference_service.py \
  --config llm_dit_mlp \
  --mode atm_ohb \
  --output-json toy_quantvla/results/phase5_quantized_server_prepare_atm_ohb_init5_14.json
```

Quantized eval command:

```bash
cd /root/autodl-tmp/quantvla-reproduction
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
/root/autodl-tmp/envs/gr00t-libero-py310/bin/python toy_quantvla/libero_eval_init_range.py \
  --task-suite-name libero_10 \
  --init-start 5 \
  --num-inits 10 \
  --headless \
  --port 5555 \
  --log-file /tmp/logs/libero_eval_libero_10_quant_atm_ohb_init5_14.log
```

## Interpretation

This disjoint-init experiment supports the method-level Phase 5 gate: `llm_dit_mlp + atm_ohb` fake W4A8 quantization is not merely preserving the accepted init `0..4` result; on init `5..14` it improves aggregate LIBERO-10 success from `70/100` to `76/100`.

The result should still be read as behavior-level simulator evidence, not as a packed-int deployment result. The current implementation uses fake quantized linear layers and attention compensation to test policy behavior under the paper's quantization idea. It does not yet demonstrate real int4/int8 kernel speed, memory savings, or end-to-end deployment efficiency.

The gain is not uniform. Quantization substantially helps the harder FP16 slice for task id `4` and improves task ids `1`, `6`, and `8`, but it regresses task ids `2` and `9`. This pattern is consistent with the offline Phase 4 finding that ATM/OHB reduces average held-out error while still causing individual regressions on some observations.

## Parse Commands

```bash
/root/autodl-tmp/envs/gr00t-libero-py310/bin/python -c "from pathlib import Path; import re; text=Path('/tmp/logs/libero_eval_libero_10_fp16_init5_14.log').read_text(); print('episodes', len(re.findall(r'^Success: ', text, flags=re.M))); print('successes', len(re.findall(r'^Success: True$', text, flags=re.M))); print('failures', len(re.findall(r'^Success: False$', text, flags=re.M))); print('terminated_exceptions', text.count('Caught exception: executing action in terminated episode')); print('\\n'.join(text.splitlines()[-80:]))"
```

```bash
/root/autodl-tmp/envs/gr00t-libero-py310/bin/python -c "from pathlib import Path; import re; text=Path('/tmp/logs/libero_eval_libero_10_quant_atm_ohb_init5_14.log').read_text(); print('episodes', len(re.findall(r'^Success: ', text, flags=re.M))); print('successes', len(re.findall(r'^Success: True$', text, flags=re.M))); print('failures', len(re.findall(r'^Success: False$', text, flags=re.M))); print('terminated_exceptions', text.count('Caught exception: executing action in terminated episode')); print('\\n'.join(text.splitlines()[-80:]))"
```

## Next

1. Treat Phase 5 as passed for behavior-level fake quantization on both the original init `0..4` protocol and the disjoint init `5..14` extension.
2. Investigate task-level regressions, especially task ids `2` and `9`, before claiming monotonic improvement.
3. Decide whether the next reproduction target is real packed quantized inference, larger calibration sweeps, or additional LIBERO suites.

Do not commit rollout videos, model weights, datasets, generated checkpoints, or large tensor dumps.
