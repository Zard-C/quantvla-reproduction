# QuantVLA Phase 5 LLM/DiT MLP Ablation Init 0-14

Boundary: this experiment isolates the `llm_dit_mlp` fake W4A8 quantization path and single compensation modes on LIBERO-10 initial states `0..14`. It is a behavior-level simulator ablation, not a packed-int deployment benchmark.

## Protocol

- Task suite: `libero_10`
- Init indices: `0..14`
- Episodes per mode: `10 tasks * 15 init states = 150`
- Modes:
  - `llm_dit_mlp + none`
  - `llm_dit_mlp + atm`
  - `llm_dit_mlp + ohb`
- Denoising steps: `8`
- Fake quantization: W4A8, activation scale `absmax`
- Quantized modules: `116`
- Quantized groups: `dit_mlp_selected`, `llm_selected`
- Calibration observations: `16`
- Eval wrapper: `toy_quantvla/libero_eval_init_range.py`
- Remote machine: RTX 5090
- Run window:
  - `none`: `2026-06-05T12:02:20+08:00` to `2026-06-05T18:04:21+08:00`
  - `atm`: `2026-06-05T18:04:54+08:00` to `2026-06-06T01:20:33+08:00`
  - `ohb`: `2026-06-06T01:21:00+08:00` to `2026-06-06T08:30:25+08:00`

The orchestrator completed all three modes, stopped each server after evaluation, and confirmed port `5555` returned `111` between modes and at the end. Final GPU state was idle.

## Aggregate Result

| mode | successes | success rate | failures | terminated exceptions | delta vs none |
|---|---:|---:|---:|---:|---:|
| `none` | 113/150 | 75.3% | 37 | 37 | +0 |
| `atm` | 114/150 | 76.0% | 36 | 36 | +1 |
| `ohb` | 116/150 | 77.3% | 34 | 34 | +3 |

The strongest single compensation mode is `ohb`, but the net gains are small. Pure W4A8 fake quantization is already close to the compensated variants.

## Per-Task Result

| task id | none | atm | ohb | atm - none | ohb - none | task |
|---:|---:|---:|---:|---:|---:|---|
| 0 | 10/15 | 11/15 | 13/15 | +1 | +3 | put both the alphabet soup and the tomato sauce in the basket |
| 1 | 13/15 | 13/15 | 14/15 | +0 | +1 | put both the cream cheese box and the butter in the basket |
| 2 | 13/15 | 14/15 | 15/15 | +1 | +2 | turn on the stove and put the moka pot on it |
| 3 | 15/15 | 13/15 | 13/15 | -2 | -2 | put the black bowl in the bottom drawer of the cabinet and close it |
| 4 | 8/15 | 13/15 | 12/15 | +5 | +4 | put the white mug on the left plate and put the yellow and white mug on the right plate |
| 5 | 14/15 | 14/15 | 15/15 | +0 | +1 | pick up the book and place it in the back compartment of the caddy |
| 6 | 9/15 | 10/15 | 8/15 | +1 | -1 | put the white mug on the plate and put the chocolate pudding to the right of the plate |
| 7 | 8/15 | 8/15 | 8/15 | +0 | +0 | put both the alphabet soup and the cream cheese box in the basket |
| 8 | 9/15 | 4/15 | 6/15 | -5 | -3 | put both moka pots on the stove |
| 9 | 14/15 | 14/15 | 12/15 | +0 | -2 | put the yellow and white mug in the microwave and close it |

The compensation is strongly task-dependent. ATM sharply improves task id `4` but sharply hurts task id `8`. OHB is more balanced than ATM on task id `8`, but still regresses task ids `3`, `8`, and `9` relative to `none`.

## Per-Init Result

| init index | none | atm | ohb |
|---:|---:|---:|---:|
| 0 | 7/10 | 6/10 | 5/10 |
| 1 | 6/10 | 6/10 | 9/10 |
| 2 | 8/10 | 6/10 | 7/10 |
| 3 | 7/10 | 8/10 | 8/10 |
| 4 | 9/10 | 9/10 | 9/10 |
| 5 | 7/10 | 8/10 | 10/10 |
| 6 | 8/10 | 9/10 | 9/10 |
| 7 | 8/10 | 8/10 | 7/10 |
| 8 | 9/10 | 5/10 | 9/10 |
| 9 | 7/10 | 9/10 | 7/10 |
| 10 | 6/10 | 8/10 | 7/10 |
| 11 | 8/10 | 9/10 | 8/10 |
| 12 | 7/10 | 7/10 | 6/10 |
| 13 | 7/10 | 8/10 | 7/10 |
| 14 | 9/10 | 8/10 | 8/10 |

Init-level behavior is also not monotonic. For example, ATM drops init `8` from `9/10` to `5/10`, while OHB keeps init `8` at `9/10`.

## Paired Outcomes

Each comparison below is over the exact same `150` task-init pairs.

| comparison | repaired failures | new regressions | same success | same failure | net |
|---|---:|---:|---:|---:|---:|
| `atm` vs `none` | 14 | 13 | 100 | 23 | +1 |
| `ohb` vs `none` | 16 | 13 | 100 | 21 | +3 |
| `ohb` vs `atm` | 15 | 13 | 101 | 21 | +2 |

The single-mode compensations are not monotonic improvements. They redistribute which task-init pairs succeed.

## Relation To Existing Phase 5 Results

Existing simulator results:

| policy | init range | successes |
|---|---:|---:|
| FP16 | `0..4` | 38/50 |
| `llm_dit_mlp + atm_ohb` | `0..4` | 38/50 |
| FP16 | `5..14` | 70/100 |
| `llm_dit_mlp + atm_ohb` | `5..14` | 76/100 |

Combined:

| policy | init range | successes |
|---|---:|---:|
| FP16 | `0..14` | 108/150 |
| `llm_dit_mlp + atm_ohb` | `0..14` | 114/150 |
| `llm_dit_mlp + none` | `0..14` | 113/150 |
| `llm_dit_mlp + atm` | `0..14` | 114/150 |
| `llm_dit_mlp + ohb` | `0..14` | 116/150 |

This suggests that the largest practical finding is not that ATM/OHB rescue a broken quantized policy. Instead, selective W4A8 over LLM/DiT MLP layers is already behaviorally robust, and compensation modes provide small, task-dependent adjustments.

## Interpretation

The behavior-level reproduction supports selective fake W4A8 quantization for GR00T/LIBERO. The quantized policy variants remain in the same performance band as FP16 and in some init ranges exceed it.

The ablation does not support a simple "ATM always helps" or "OHB always helps" statement. ATM gives a net `+1/150` over `none`, mainly by improving task id `4`, while it regresses task id `8` by `-5`. OHB gives a larger net `+3/150` and is the best single compensation in this run, but it also regresses task ids `3`, `8`, and `9`.

Therefore, the strongest current claim is:

> `llm_dit_mlp` selective W4A8 fake quantization is behaviorally viable on LIBERO-10 init `0..14`; ATM/OHB act as task-dependent error redistribution mechanisms rather than guaranteed monotonic improvements. OHB is the best single compensation in this ablation, while ATM+OHB remains competitive with FP16 and passes the Phase 5 behavior gate.

This experiment still does not demonstrate packed integer kernel speed, memory savings, or deployment efficiency.

## Mechanistic Analysis

The main result should not be read as "ATM/OHB rescue a broken quantized policy." The `none` result is already `113/150`, which means the selected W4A8 path is behaviorally robust before attention compensation is applied.

The most important design choice is that `llm_dit_mlp` quantizes selected LLM linear layers and DiT feed-forward linears while leaving DiT attention projections floating point. This avoids the most fragile part of the action head. Attention logits pass through:

```text
softmax(QK^T / sqrt(d))
```

A small perturbation in Q/K scale can change attention entropy and routing. MLP quantization is more like an additive residual perturbation:

```text
y_student = y_teacher + epsilon
```

That error is partially absorbed by normalization, residual connections, the diffusion denoising process, and closed-loop replanning across observations. This explains why pure W4A8 can remain close to FP16.

The fact that quantized variants can exceed FP16 does not imply that quantization is intrinsically better. LIBERO success is a discontinuous closed-loop event. A small action perturbation can push a rollout into a different contact sequence:

```text
small action drift
-> changed object contact
-> changed next observation
-> different policy trajectory
-> success/failure flip
```

Therefore, the right analysis unit is not only output MSE or aggregate success. The paired flip counts are more informative: compensation modes repair some FP16/quantized failures and introduce new failures elsewhere.

ATM is a temperature-style correction for DiT attention. In this implementation it computes:

```text
alpha = std_teacher(attention_logits) / std_student(attention_logits)
query <- alpha * query
```

Scaling the query changes the attention logit distribution before softmax. This can restore teacher-like attention sharpness when the student is too soft or too sharp, but it also directly changes token routing. A single per-module calibration scale, estimated from 16 calibration observations, cannot be optimal for every LIBERO task and initial state. This explains the high-variance ATM behavior: it improves task id `4` by `+5`, but regresses task id `8` by `-5`.

OHB is an output-energy correction for the same attention blocks. It computes:

```text
beta = rms_teacher(attention_output) / rms_student(attention_output)
attention_output <- beta * attention_output
```

Unlike ATM, OHB does not directly alter the softmax routing. It changes how strongly the attention branch contributes before residual addition. This is a more conservative intervention, which is consistent with the ablation: OHB has the best single-mode aggregate score (`116/150`) and hurts task id `8` less than ATM (`6/15` vs `4/15`). It still regresses task ids `3`, `8`, and `9`, because residual energy changes are enough to shift closed-loop trajectories.

ATM and OHB should also not be expected to add linearly. ATM changes attention probabilities, which changes the output whose RMS OHB later rescales. The two operations are coupled within the same DiT attention block:

```text
ATM changes "where attention looks"
OHB changes "how loudly attention output enters the residual stream"
```

Applying both can overcorrect, partially cancel, or move the policy into a different trajectory basin. This explains why `atm_ohb` (`114/150`) does not exceed standalone OHB (`116/150`) in the current aggregate.

From a statistical standpoint, small aggregate gaps should be treated cautiously. The difference between `none`, `atm`, and `atm_ohb` is only `1/150`, while OHB is `3/150` above `none`. The stronger evidence is the structured task-level pattern:

- Selective W4A8 is robust: all quantized variants remain near or above FP16.
- ATM is high-variance: it strongly helps task id `4` and strongly hurts task id `8`.
- OHB is the best single compensation here, but still introduces regressions.
- Compensation changes the distribution of successes rather than monotonically improving all task-init pairs.

The current mechanistic conclusion is:

> Selective W4A8 over LLM/DiT MLP layers is viable because it avoids quantizing the most attention-sensitive DiT projections. ATM and OHB are calibration-based trajectory-redistribution mechanisms. OHB behaves like a more stable residual-energy correction, while ATM behaves like a higher-risk attention-temperature correction.

## Commands

Server command template:

```bash
cd /root/autodl-tmp/quantvla-reproduction
/root/autodl-tmp/envs/gr00t-libero-py310/bin/python toy_quantvla/quantized_inference_service.py \
  --config llm_dit_mlp \
  --mode MODE \
  --output-json toy_quantvla/results/phase5_ablation_llm_dit_mlp_MODE_init0_14_prepare.json
```

Eval command template:

```bash
cd /root/autodl-tmp/quantvla-reproduction
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
/root/autodl-tmp/envs/gr00t-libero-py310/bin/python toy_quantvla/libero_eval_init_range.py \
  --task-suite-name libero_10 \
  --init-start 0 \
  --num-inits 15 \
  --headless \
  --port 5555 \
  --log-file /tmp/logs/ablation_eval_libero_10_llm_dit_mlp_MODE_init0_14.log
```

Prepare JSON files:

- `toy_quantvla/results/phase5_ablation_llm_dit_mlp_none_init0_14_prepare.json`
- `toy_quantvla/results/phase5_ablation_llm_dit_mlp_atm_init0_14_prepare.json`
- `toy_quantvla/results/phase5_ablation_llm_dit_mlp_ohb_init0_14_prepare.json`

Do not commit rollout videos, model weights, datasets, generated checkpoints, or large tensor dumps.
