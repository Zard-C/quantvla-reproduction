# Phase 11 DiT Cache Direction

## Goal

This phase shifts the acceleration direction from naked FP4 DiT MLP kernels to
Cache-DiT-style denoising reuse.

The motivation is the Phase 10 result: `dit_mlp_only` real FP4 can be promising
offline, but in closed-loop serving it executes many small `M=49` FP4 GEMMs and
the activation-pack/wrapper overhead dominates. Instead of only making every
DiT block cheaper, Phase 11 first asks whether some DiT block computation can be
skipped.

## Probe Result

Script:

```bash
toy_quantvla/phase11_dit_cache_probe.py
```

Run:

```bash
python toy_quantvla/phase11_dit_cache_probe.py \
  --num-observations 4 \
  --indices 115,215,315,415 \
  --denoising-steps 8 \
  --include-raw-rows \
  --output-json toy_quantvla/results/phase11_dit_cache_probe_obs4_seed20260610.json
```

Observed structure:

| Item | Value |
|---|---:|
| Real observations | 4 |
| DiT blocks | 16 |
| Denoising steps | 8 |
| Captured rows | 512 |

Average adjacent-step output drift grows over denoising time:

| Transition | Output rel RMSE | Output cosine | Residual rel RMSE |
|---|---:|---:|---:|
| 0 -> 1 | 0.05850 | 0.999015 | 0.06534 |
| 1 -> 2 | 0.08846 | 0.998825 | 0.06406 |
| 2 -> 3 | 0.10281 | 0.998336 | 0.06836 |
| 3 -> 4 | 0.11739 | 0.997525 | 0.07340 |
| 4 -> 5 | 0.13710 | 0.996320 | 0.08155 |
| 5 -> 6 | 0.15940 | 0.994229 | 0.09804 |
| 6 -> 7 | 0.20636 | 0.988869 | 0.13080 |

Most stable residual blocks:

| Block | Residual rel RMSE mean | Residual rel RMSE p90 | Output rel RMSE mean |
|---:|---:|---:|---:|
| 12 | 0.01502 | 0.01784 | 0.11255 |
| 15 | 0.02084 | 0.02813 | 0.08192 |
| 14 | 0.03961 | 0.05854 | 0.11015 |
| 0 | 0.05211 | 0.07765 | 0.13588 |

Interpretation: residual reuse looks more plausible than direct hidden-state
reuse. Blocks `12` and `15` are the first conservative candidates; block `14`
is a reasonable extension; block `13` is less stable than neighboring suffix
blocks.

## Static Residual Cache Smoke

Script:

```bash
toy_quantvla/phase11_dit_residual_cache_smoke.py
```

Mechanism:

```text
y_t ~= x_t + (y_prev_full - x_prev_full)
```

Selected blocks skip configured denoising steps and reuse the latest full-step
residual. This is an offline action-drift/timing smoke, not a closed-loop
success benchmark.

Results on observations `115,215,315,415`:

| Config | DiT block skip | Action rel RMSE mean/max | Cosine mean | Max abs diff max | Time ratio |
|---|---:|---:|---:|---:|---:|
| b15, steps 1/3/5 | 2.3% | 0.00205 / 0.00285 | 0.99999768 | 0.00366 | 1.032 |
| b12,b15, steps 1/3/5 | 4.7% | 0.00199 / 0.00279 | 0.99999779 | 0.00401 | 0.931 |
| b12,b14,b15, steps 1/3/5 | 7.0% | 0.00197 / 0.00291 | 0.99999795 | 0.00366 | 0.990 |
| b12-15, steps 1/3/5 | 9.4% | 0.00218 / 0.00288 | 0.99999752 | 0.00366 | 0.927 |
| b12,b15, steps 1-6 | 9.4% | 0.00235 / 0.00313 | 0.99999728 | 0.00412 | 0.852 |
| b12-15, steps 1-6 | 18.8% | 0.00529 / 0.00916 | 0.99998474 | 0.00885 | 1.217 |

Timing is noisy in this offline script, but action drift is informative. The
best first closed-loop candidate is:

```text
cache_blocks = 12,15
cache_steps = 1-6
```

It skips about `9.4%` of all DiT block calls while keeping action relative RMSE
around `0.00235`, well below the naked FP4 DiT MLP action drift observed in
Phase 10.

Avoid using `b12-15, steps 1-6` as the first rollout candidate. It doubles the
skip rate, but the action drift climbs and one observation reaches relative
RMSE `0.00916`.

## Next Step

Implement a residual-cache inference server variant, then run the deterministic
15-case matched closed-loop protocol:

```text
task4 init 6-10
task6 init 0-4
task8 init 6-10
```

Recommended first server config:

```text
FP16 + DiT residual cache
cache_blocks=12,15
cache_steps=1-6
```

If closed-loop success holds, the next candidates are:

```text
FP16 + DiT residual cache, cache_blocks=12-15, cache_steps=1,3,5
residual cache + up_proj FP4
residual cache + DiT MLP FP4 + OHB
```
