# Phase 5 Resume Handoff

Snapshot after completion: the official FP16 `libero_10` 5-trial baseline finished successfully as a baseline run.

## Final Status

- Eval process: finished; no `run_libero_eval.py` process remains.
- Final result: `38/50`
- Final success rate: `76.0%`
- Official README reference for the Long checkpoint: `38/50 (76%)`
- Terminated-episode exceptions: `12`, matching the `12` failed episodes.
- Eval log: `/tmp/logs/libero_eval_libero_10.log`
- Rollout directory: `/root/autodl-tmp/Isaac-GR00T-n1.5/examples/Libero/eval/rollouts/2026_06_04`
- Rollout prefix: `2026_06_04-17_27_10`

The FP16 inference server may still be running on port `5555` after eval completion. Stop it before launching a quantized student server.

## Quick Verification

```bash
pgrep -af "scripts/inference_service.py"
pgrep -af "run_libero_eval.py"
/root/autodl-tmp/envs/gr00t-libero-py310/bin/python -c "import socket; s=socket.socket(); print(s.connect_ex(('127.0.0.1',5555))); s.close()"
```

Expected after cleanup:

- no `run_libero_eval.py`
- no `scripts/inference_service.py`
- socket check returns `111`

## Parse Command

```bash
/root/autodl-tmp/envs/gr00t-libero-py310/bin/python -c "from pathlib import Path; import re; text=Path('/tmp/logs/libero_eval_libero_10.log').read_text(); print('episodes', len(re.findall(r'^Success: ', text, flags=re.M))); print('successes', len(re.findall(r'^Success: True$', text, flags=re.M))); print('failures', len(re.findall(r'^Success: False$', text, flags=re.M))); print('terminated_exceptions', text.count('Caught exception: executing action in terminated episode'))"
```

Expected output:

```text
episodes 50
successes 38
failures 12
terminated_exceptions 12
```

## Per-Task Result

| task id | successes / trials | success rate | task |
|---:|---:|---:|---|
| 0 | 5/5 | 100% | put both the alphabet soup and the tomato sauce in the basket |
| 1 | 3/5 | 60% | put both the cream cheese box and the butter in the basket |
| 2 | 3/5 | 60% | turn on the stove and put the moka pot on it |
| 3 | 5/5 | 100% | put the black bowl in the bottom drawer of the cabinet and close it |
| 4 | 4/5 | 80% | put the white mug on the left plate and put the yellow and white mug on the right plate |
| 5 | 5/5 | 100% | pick up the book and place it in the back compartment of the caddy |
| 6 | 3/5 | 60% | put the white mug on the plate and put the chocolate pudding to the right of the plate |
| 7 | 4/5 | 80% | put both the alphabet soup and the cream cheese box in the basket |
| 8 | 1/5 | 20% | put both moka pots on the stove |
| 9 | 5/5 | 100% | put the yellow and white mug in the microwave and close it |

## Next Step

The FP16 baseline is accepted. The next simulator run can compare quantized students against this baseline:

1. Primary: `llm_dit_mlp + atm_ohb`.
2. Secondary: `llm_dit_mlp + ohb`.
3. Use the same `libero_10`, `5` trials per task, denoising steps `8`, and headless EGL setup.
4. Keep per-task results. Task id `8` is already weak under FP16 (`1/5`), so aggregate drops should be interpreted with per-task context.

## What Not To Do

- Do not treat the earlier `6/10` one-trial smoke as the comparison baseline.
- Do not start a quantized student server while the old FP16 server still owns port `5555`.
- Do not delete `/tmp/logs/libero_eval_libero_10.log` or the rollout directory.
- Do not commit model weights, datasets, generated checkpoints, rollout videos, or large tensor dumps.
