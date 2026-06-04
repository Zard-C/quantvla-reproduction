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

## Parse Command

```bash
/root/autodl-tmp/envs/gr00t-libero-py310/bin/python -c "from pathlib import Path; import re; text=Path('/tmp/logs/libero_eval_libero_10_fp16_init5_14.log').read_text(); print('episodes', len(re.findall(r'^Success: ', text, flags=re.M))); print('successes', len(re.findall(r'^Success: True$', text, flags=re.M))); print('failures', len(re.findall(r'^Success: False$', text, flags=re.M))); print('terminated_exceptions', text.count('Caught exception: executing action in terminated episode')); print('\\n'.join(text.splitlines()[-80:]))"
```

## Next

1. Run `llm_dit_mlp + atm_ohb` on the same init `5..14` protocol.
2. Compare FP16 and quantized per-task and per-init deltas.
3. Preserve the FP16 weak-task context when interpreting aggregate success, especially task ids `4`, `6`, and `8`.

Do not commit rollout videos, model weights, datasets, generated checkpoints, or large tensor dumps.
