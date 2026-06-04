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

## FP16 Run

Current state at snapshot `2026-06-04T14:18:00Z`: the FP16 init `5..14` run is active on the RTX 5090 machine.

- FP16 server PID: `455603`
- FP16 eval PID: `455912`
- Server log: `/tmp/logs/fp16_server_init5_14.log`
- Eval log: `/tmp/logs/libero_eval_libero_10_fp16_init5_14.log`
- Eval stdout: `/tmp/logs/libero_eval_libero_10_fp16_init5_14.stdout`

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

## Progress Command

```bash
/root/autodl-tmp/envs/gr00t-libero-py310/bin/python -c "from pathlib import Path; import re; text=Path('/tmp/logs/libero_eval_libero_10_fp16_init5_14.log').read_text(); print('episodes', len(re.findall(r'^Success: ', text, flags=re.M))); print('successes', len(re.findall(r'^Success: True$', text, flags=re.M))); print('failures', len(re.findall(r'^Success: False$', text, flags=re.M))); print('terminated_exceptions', text.count('Caught exception: executing action in terminated episode')); print('\\n'.join(text.splitlines()[-80:]))"
```

## Next

1. Wait for the FP16 init `5..14` run to finish.
2. Stop the FP16 server and verify port `5555` is released.
3. Run `llm_dit_mlp + atm_ohb` on the same init `5..14` protocol.
4. Compare FP16 and quantized per-task deltas.

Do not commit rollout videos, model weights, datasets, generated checkpoints, or large tensor dumps.
