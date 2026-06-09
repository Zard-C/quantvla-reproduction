# Phase 12 FP4 Speed Route

Boundary: this note summarizes the current packed-FP4 engineering path after the Phase 11 residual-cache result and the Phase 12 `up_proj + OHB` latency probe. It is focused on inference speed, not on proving a new simulator success-rate result.

## Current State

The best behavior signal so far is still the packed FP4 `up_proj` path:

| policy | cases | successes | policy p50 | policy p90 | note |
|---|---:|---:|---:|---:|---|
| FP16 baseline | 15 | 7/15 | 0.0952s | 0.1725s | deterministic matched subset |
| packed FP4 `up_proj` | 15 | 9/15 | 0.0966s | 0.1806s | best current behavior/scope tradeoff |
| packed FP4 `dit_mlp_only` | 15 | 7/15 | 0.1477s | 0.2159s | no behavior gain, slower |
| FP16 DiT residual cache `b12,15 s1-6` | 15 | 5/15 | 0.1613s | 0.1669s | behavior and speed both negative |
| packed FP4 `up_proj + OHB` v3 | partial 2 episodes | 1/2 | 0.1552s server p50 | 0.1630s server p90 | stopped after latency was clearly negative |

The Phase 12 `up_proj + OHB` v3 run was intentionally stopped early. It is not a closed-loop success-rate benchmark. It is a latency probe showing that the current OHB application path is too expensive for speed-oriented deployment.

## Why Fake Quant And Packed FP4 Differ

The Phase 5 fake-quant result and the Phase 8/9 packed-FP4 result are not numerically equivalent:

| axis | fake W4A8 path | packed FP4 CUTLASS path |
|---|---|---|
| weight format | symmetric fake 4-bit dequantized back into torch linear | blockscaled E2M1 FP4 packed tensor |
| activation format | fake 8-bit activation quantization | activation packed to FP4 for CUTLASS GEMM |
| compute path | `F.linear` with fake-dequantized tensors | runtime activation pack + CUTLASS FP4 GEMM + Python wrapper |
| purpose | behavior-level perturbation study | deployment-oriented low-bit backend |
| expected speed | none | possible, but only after bridge overhead is removed |

This explains the apparent gap. Fake quant mostly tests whether the policy can tolerate quantization-like noise. Packed FP4 tests the actual low-bit format plus a new execution path. In LIBERO, small numerical differences can also be amplified by contact dynamics:

```text
small action drift -> changed contact sequence -> changed observation -> different closed-loop trajectory
```

So a small offline action error can become a large success/failure flip.

## Phase 12 OHB Probe

Implementation changes:

- `toy_quantvla/cutlass_fp4_inference_service.py`
  - Added `--atm-ohb-mode`, calibration indices, calibration seed, and `--ohb-skip-epsilon`.
  - The server can now calibrate teacher/student DiT attention stats around a packed FP4 patch.
- `toy_quantvla/phase3_atm_ohb_forward.py`
  - Avoids computing attention-logit std during apply mode.
  - Adds an OHB output wrapper that preserves the original attention processor fast path when only beta scaling is needed.
  - Skips installing processors for empty selected scales.
- `toy_quantvla/run_phase12_fp4_ohb_matched_set.sh`
  - Runs deterministic matched 15-case `packed FP4 up_proj + OHB`.

Prepare-only result for `packed FP4 up_proj + OHB`:

| metric | value |
|---|---:|
| patched modules | 12 LLM `up_proj` modules |
| prepare seconds | 173.9s |
| model load | 10.9s |
| teacher attention calibration | 2.8s |
| student attention calibration | 156.0s |
| prewarm total | 0.77s |
| OHB beta mean/min/max | 1.00034 / 0.99823 / 1.00309 |

The student calibration time is dominated by first-time CUTLASS shape compilation. This is a cold-start/build-time cost, not steady-state request latency, but it must be moved out of the user path for any deployment story.

The OHB scale is almost identity for packed FP4 `up_proj`. That means OHB is unlikely to change behavior much for this scope, while installing a DiT attention wrapper is expensive.

## Latency Finding

The v3 partial run produced:

| source | count | mean | p50 | p90 | p99 |
|---|---:|---:|---:|---:|---:|
| server `get_action` | 1566 | 0.1529s | 0.1552s | 0.1630s | 0.1727s |
| episode 1 client policy | 273 | 0.1604s | 0.1605s | 0.1680s | 0.1790s |
| episode 2 client policy | 991 | 0.1590s | 0.1599s | 0.1667s | 0.1762s |

Request traces showed:

```text
fp4_compile_count_delta = 0
fp4_module_calls_delta = 12 per request
fp4_local_compile_cache_hits_delta = 12 per request
```

So the slowdown was not caused by online CUTLASS compilation. It came from the extra OHB attention wrapper path and residual/output tensor operations. Even the lighter wrapper was still too expensive because it touches all DiT attention blocks across denoising steps.

## Speed-Oriented Conclusions

1. `up_proj` is the only packed FP4 scope currently worth carrying forward behaviorally.
   It reached 9/15 on the matched subset, but did not improve p50 latency over FP16.

2. OHB is not currently a speed-positive addition for packed `up_proj`.
   Its beta scales are nearly identity, and its online wrapper overhead is much larger than the expected numerical correction.

3. The current packed FP4 bridge is not a deployment implementation.
   It still loads FP16 weights first, packs selected weights at startup, packs activations every forward, and dispatches through Python module wrappers.

4. Cold compile and steady-state latency are separate problems.
   Persistent/precompiled CUTLASS kernels can fix startup pain, but not the current p50 gap if activation packing and wrapper overhead dominate.

5. Fake quant success should guide scope selection, not speed claims.
   Fake W4A8 showed that selective quantization can be behaviorally viable. It does not prove that the packed FP4 backend is already fast.

## Next Optimization Targets

Priority order:

1. Profile packed `up_proj none` at module level.
   Measure activation pack, GEMM, finalize, Python dispatch, and full `get_action` share. Do this before adding OHB or wider scopes.

2. Add an OHB skip threshold for near-identity scales.
   Use `--ohb-skip-epsilon 0.005` as a first test. For the current calibration this should skip all OHB processors and recover the `up_proj none` path.

3. Move cold compile out of evaluation.
   Precompile observed LLM sequence lengths and persist/share compiled artifacts where possible. The observed M values during calibration were around `555-566`.

4. Reduce activation packing overhead.
   The current bridge still packs activations at every forward. A production path should fuse or substantially optimize activation packing.

5. Avoid Python attention wrappers for speed runs.
   If OHB is needed, fuse beta into the attention/output path or skip it when beta is close to 1. Do not use a Python processor wrapper in speed benchmarks.

6. Only revisit DiT MLP packed FP4 after numerical recovery.
   `dit_mlp_only` is slower and behaviorally not better on the matched subset, so speed work should not start there.

## Recommended Next Run

Run a speed-only matched subset with:

```bash
TAG=phase12_speed_fp4_up_proj_none_profile \
RUN_FP16=0 \
RUN_FP4=1 \
DETERMINISTIC_POLICY_SEEDS=1 \
POLICY_SEED_BASE=20260609 \
bash toy_quantvla/run_phase9_up_proj_matched_set.sh
```

Then run the same scope through the Phase 12 runner with OHB effectively skipped:

```bash
TAG=phase12_speed_fp4_up_proj_ohb_skip_v1 \
OHB_SKIP_EPSILON=0.005 \
ATM_OHB_MODE=ohb \
DETERMINISTIC_POLICY_SEEDS=1 \
POLICY_SEED_BASE=20260609 \
bash toy_quantvla/run_phase12_fp4_ohb_matched_set.sh
```

The expected result is that skipped-OHB latency should return close to packed FP4 `up_proj none`. If it does, the remaining speed problem is the FP4 backend itself, not OHB.
