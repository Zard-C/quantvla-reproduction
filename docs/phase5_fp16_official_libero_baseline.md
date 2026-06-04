# QuantVLA Phase 5 FP16 Official LIBERO Baseline

Boundary: this is the official FP16 GR00T server + LIBERO simulator baseline for the current reproduction. It does not include any quantized student path. The final 5-trial run matches the official README reference for the Long LIBERO checkpoint, so it is accepted as the FP16 simulator comparison baseline for the next quantized runs.

## Final 5-Trial Result

- Date: 2026-06-04 UTC
- Workspace: `/root/autodl-tmp/quantvla-reproduction`
- GR00T source: `/root/autodl-tmp/Isaac-GR00T-n1.5`
- Checkpoint: `/root/autodl-tmp/models/gr00t-n1.5-libero-long-posttrain`
- Environment: `/root/autodl-tmp/envs/gr00t-libero-py310`
- Task suite: `libero_10`
- Trials per task: `5`
- Total episodes: `50`
- Denoising steps: `8`
- Server mode: official FP16 `Gr00tPolicy` through `scripts/inference_service.py`
- Client/eval mode: official `examples/Libero/eval/run_libero_eval.py`
- Rendering: headless EGL with `MUJOCO_GL=egl` and `PYOPENGL_PLATFORM=egl`
- Final result: `38/50`
- Final success rate: `76.0%`
- Official README reference for this Long checkpoint: `38/50 (76%)`
- `executing action in terminated episode` exceptions: `12`
- Eval log: `/tmp/logs/libero_eval_libero_10.log`
- Rollout directory: `/root/autodl-tmp/Isaac-GR00T-n1.5/examples/Libero/eval/rollouts/2026_06_04`
- Rollout filename prefix: `2026_06_04-17_27_10`

Per-task result:

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

The main weak task is task id `8`, `put both moka pots on the stove`, at `1/5`. This task should be watched closely in quantized student rollouts because it already has low FP16 headroom.

## Commands

Official FP16 server:

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

Official LIBERO eval:

```bash
cd /root/autodl-tmp/Isaac-GR00T-n1.5/examples/Libero/eval
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
/root/autodl-tmp/envs/gr00t-libero-py310/bin/python run_libero_eval.py \
  --task-suite-name libero_10 \
  --num-trials-per-task 5 \
  --headless \
  --port 5555
```

The 5-trial run completed all 50 episodes. `run_libero_eval.py` exited normally. The FP16 server was left running after eval completion and should be stopped before launching a quantized student server on port `5555`.

## Environment Versions

- Python: `3.10.20`
- Torch: `2.8.0+cu128`
- CUDA device: `NVIDIA GeForce RTX 5090`
- `flash-attn`: `2.8.3`
- `libero`: `0.1.1`
- `robosuite`: `1.4.0`
- `bddl`: `1.0.1`

LIBERO config:

```yaml
assets: /root/autodl-tmp/envs/gr00t-libero-py310/lib/python3.10/site-packages/libero/libero/assets
bddl_files: /root/autodl-tmp/envs/gr00t-libero-py310/lib/python3.10/site-packages/libero/libero/bddl_files
benchmark_root: /root/autodl-tmp/envs/gr00t-libero-py310/lib/python3.10/site-packages/libero/libero
datasets: /root/autodl-tmp/datasets/libero
init_states: /root/autodl-tmp/envs/gr00t-libero-py310/lib/python3.10/site-packages/libero/libero/init_files
```

The first eval run downloaded missing LIBERO assets from Hugging Face to `/root/.cache/libero/assets` before simulator rollout. The asset download completed successfully.

## Alignment Checks

No obvious global configuration mismatch was found:

- Checkpoint is the local Long/LIBERO-10 path: `/root/autodl-tmp/models/gr00t-n1.5-libero-long-posttrain`.
- Official README lists Long as `youliangtan/gr00t-n1.5-libero-long-posttrain`, `LiberoDataConfig`, `denoising-steps 8`, and `38/50 (76%)`.
- The server used `examples.Libero.custom_data_config:LiberoDataConfig`, not the Goal-only `LiberoDataConfigMeanStd`.
- `Gr00tPolicy` sets both the model and modality transform to eval mode. `VideoColorJitter` is disabled in eval mode.
- Local `libero_10` has the standard 10 tasks in task order `[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]`, with 50 init states per task.
- Package `libero/libero/assets` is absent in this install, so LIBERO correctly downloaded and used the Hugging Face asset cache at `/root/.cache/libero/assets`.

Because the final `38/50` exactly matches the official reference, the earlier `6/10` one-init smoke failure pattern is best interpreted as fragile single-init variance rather than a checkpoint, data config, task ordering, or simulator setup mismatch.

## One-Trial Smoke Run

Before the accepted 5-trial baseline, a smaller official FP16 smoke run was executed with `1` trial per task:

- Episodes completed: `10/10`
- Successes: `6/10`
- Smoke success rate: `60.0%`
- Rollout filename prefix: `2026_06_04-16_13_42`
- Server cleanup: official server was stopped after eval; port `5555` returned `ECONNREFUSED` (`111`) on a local socket check.

| episode | success | task |
|---:|---|---|
| 1 | true | put both the alphabet soup and the tomato sauce in the basket |
| 2 | true | put both the cream cheese box and the butter in the basket |
| 3 | true | turn on the stove and put the moka pot on it |
| 4 | true | put the black bowl in the bottom drawer of the cabinet and close it |
| 5 | true | put the white mug on the left plate and put the yellow and white mug on the right plate |
| 6 | true | pick up the book and place it in the back compartment of the caddy |
| 7 | false | put the white mug on the plate and put the chocolate pudding to the right of the plate |
| 8 | false | put both the alphabet soup and the cream cheese box in the basket |
| 9 | false | put both moka pots on the stove |
| 10 | false | put the yellow and white mug in the microwave and close it |

The one-trial run is not accepted as a quantized-student comparison baseline because it is much smaller and landed below the official 5-trial reference.

## Targeted Failed-Task Rerun

To separate global baseline mismatch from init-state variance, the four failed task ids from the one-trial smoke run were rerun with init state `1` using the same official FP16 server and same eval mechanics. This was a targeted diagnostic, not a benchmark replacement.

- Command script: `/root/autodl-tmp/tmp/phase5_targeted_libero_eval.py`
- Log: `/tmp/logs/libero_eval_libero_10_targeted_failed_tasks_init1.log`
- Task ids: `6,7,8,9`
- Init index: `1`
- Final targeted result: `3/4`
- Server cleanup: official server was stopped after targeted rerun; port `5555` returned `ECONNREFUSED` (`111`) on a local socket check.

| task id | init | success | steps | exception | task |
|---:|---:|---|---:|---|---|
| 6 | 1 | true | 240 | none | put the white mug on the plate and put the chocolate pudding to the right of the plate |
| 7 | 1 | true | 326 | none | put both the alphabet soup and the cream cheese box in the basket |
| 8 | 1 | false | 1000 | executing action in terminated episode | put both moka pots on the stove |
| 9 | 1 | true | 253 | none | put the yellow and white mug in the microwave and close it |

This targeted rerun further supports the interpretation that one-init variance was the main issue in the `6/10` smoke result.

## Exceptions And Warnings

Failed episodes in both the one-trial and five-trial runs commonly logged:

```text
Caught exception: executing action in terminated episode
```

The final 5-trial run logged this exception `12` times, matching the `12` failed episodes. The eval process still completed all `50` episodes and reported the final success rate. This exception is likely a failure-at-horizon artifact rather than an inference-server crash: LIBERO/robosuite uses horizon `1000` with `ignore_done=False`, while the official loop can continue until `max_steps + num_steps_wait` for `libero_10`. A task that does not satisfy success by horizon can therefore hit `ValueError("executing action in terminated episode")` on a final extra step.

During interpreter teardown, robosuite/EGL may emit ignored cleanup warnings around `EGL_NOT_INITIALIZED` from `eglMakeCurrent`. These warnings happen after episode completion and did not abort the runs.

## Interpretation

The Phase 5 FP16 baseline is accepted. It validates the simulator environment, official FP16 GR00T server startup, ZMQ client/server path, LIBERO initial-state loading, headless rendering, logging, rollout video generation, and checkpoint/config alignment.

The next simulator comparison can use this as the FP16 reference:

1. Primary quantized candidate: `llm_dit_mlp + atm_ohb`.
2. Secondary quantized candidate: `llm_dit_mlp + ohb`.
3. Use the same `libero_10`, `5` trials per task, denoising steps `8`, headless EGL setup, and official task order.
4. Report per-task success deltas, not only aggregate success rate, because FP16 is already weak on task id `8`.
5. Keep simulator success rate separate from offline action-drift metrics and from packed-kernel speed or memory claims.
