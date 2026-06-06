# Phase 5 Video Case Analysis

Boundary: this note analyzes already generated LIBERO rollout videos. It does
not introduce new inference runs.

## Source Material

Local contact sheets:

- `analysis_keyframes/batch/none_repair_task8_init4.jpg`
- `analysis_keyframes/batch/none_repair_task8_init10.jpg`
- `analysis_keyframes/batch/none_repair_task2_init5.jpg`
- `analysis_keyframes/batch/atmohb_repair_task4_init6.jpg`
- `analysis_keyframes/batch/atmohb_repair_task8_init9.jpg`
- `analysis_keyframes/batch/regress_none_task0_init1.jpg`
- `analysis_keyframes/batch/regress_atmohb_task9_init13.jpg`

Follow-up contact sheets:

- `analysis_keyframes/batch2/none_repair_task8_init6.jpg`
- `analysis_keyframes/batch2/none_repair_task8_init7.jpg`
- `analysis_keyframes/batch2/none_repair_task8_init8.jpg`
- `analysis_keyframes/batch2/none_repair_task8_init12.jpg`
- `analysis_keyframes/batch2/none_repair_task8_init13.jpg`
- `analysis_keyframes/batch2/atmohb_repair_task4_init9.jpg`
- `analysis_keyframes/batch2/atmohb_repair_task4_init10.jpg`
- `analysis_keyframes/batch2/atmohb_repair_task4_init11.jpg`

Regression contact sheets:

- `analysis_keyframes/regressions/none_regress_task0_init3.jpg`
- `analysis_keyframes/regressions/none_regress_task0_init7.jpg`
- `analysis_keyframes/regressions/none_regress_task7_init3.jpg`
- `analysis_keyframes/regressions/none_regress_task7_init10.jpg`
- `analysis_keyframes/regressions/atmohb_regress_task0_init8.jpg`
- `analysis_keyframes/regressions/atmohb_regress_task6_init1.jpg`
- `analysis_keyframes/regressions/atmohb_regress_task9_init9.jpg`
- `analysis_keyframes/regressions/both_quant_regress_task8_init0.jpg`

Each sheet has three policy rows and ten uniformly spaced frames over the
rollout. Green labels are successful rollouts; red labels are failed rollouts.

## Case Selection

The cases were selected from matched task/init outcome flips across FP16,
`llm_dit_mlp + none`, and `llm_dit_mlp + atm_ohb`.

| category | count | representative cases |
| --- | ---: | --- |
| `none` repairs both FP16 and ATM/OHB failures | 9 | task 8 init 4, task 8 init 10, task 2 init 5 |
| ATM/OHB repairs FP16 and `none` failures | 7 | task 4 init 6, task 8 init 9 |
| `none` regresses FP16 success | 16 | task 0 init 1 |
| ATM/OHB regresses FP16 success | 13 | task 9 init 13 |

## Observations

### None Repairs

Task 8 init 4 and init 10 show the same qualitative pattern. FP16 and ATM/OHB
run to the horizon (`991` frames) and fail. The `none` policy succeeds much
earlier (`770` frames for init 4, `452` frames for init 10). The visible
difference appears early: the successful `none` trajectory commits to a
different approach path and reaches the target region sooner, while the failed
rows spend the second half of the rollout in repeated approach or correction
motions.

Task 2 init 5 is an even stronger version of this pattern. FP16 and ATM/OHB
both run to horizon, while `none` succeeds in `262` frames. This looks less like
a small final-placement correction and more like a different early mode choice.

The follow-up task 8 cases reinforce the same pattern:

| task/init | FP16 | ATM/OHB | `none` | video-level observation |
| --- | --- | --- | --- | --- |
| task 8 init 6 | fail, `991` frames | fail, `991` frames | success, `554` frames | `none` reaches the target neighborhood earlier; failures remain in late correction |
| task 8 init 7 | fail, `991` frames | fail, `991` frames | success, `388` frames | `none` takes a short successful branch |
| task 8 init 8 | fail, `991` frames | fail, `991` frames | success, `395` frames | same early-branching pattern as init 7 |
| task 8 init 12 | fail, `991` frames | fail, `991` frames | success, `652` frames | slower success, but still avoids horizon failure |
| task 8 init 13 | fail, `991` frames | fail, `991` frames | success, `610` frames | medium-length repair with the same qualitative target-entry pattern |

Interpretation: raw quantization perturbation can act as a structured
closed-loop trajectory perturbation. On some initial states it nudges the policy
out of a failure basin and into a faster successful mode.

### ATM/OHB Repairs

Task 4 init 6 is the clearest ATM/OHB repair case. FP16 and `none` both run to
horizon, while ATM/OHB succeeds in `222` frames. The successful rollout is not
just longer or more exploratory; it is visibly more direct and terminates early.

Task 8 init 9 shows that task 8 is not simply helped by larger noise. Here
ATM/OHB succeeds in `453` frames, while FP16 and `none` both run to horizon. The
beneficial perturbation direction is init-dependent.

The follow-up task 4 cases show a consistent ATM/OHB stabilization pattern:

| task/init | FP16 | ATM/OHB | `none` | video-level observation |
| --- | --- | --- | --- | --- |
| task 4 init 9 | fail, `991` frames | success, `223` frames | fail, `991` frames | ATM/OHB completes the mug/plate interaction early |
| task 4 init 10 | fail, `991` frames | success, `242` frames | fail, `991` frames | same short successful branch |
| task 4 init 11 | fail, `991` frames | success, `255` frames | fail, `991` frames | same short successful branch |

Interpretation: ATM/OHB can suppress harmful quantization drift and keep action
chunks closer to a stable teacher-like path. This helps when raw quantization
pushes the rollout into a bad correction loop.

### Regressions

Task 0 init 1 shows `none` damaging an otherwise easy success. FP16 and ATM/OHB
both finish quickly (`349` and `246` frames), while `none` runs to horizon. The
failed rollout appears to disturb the early object/container relationship and
then enters long ineffective correction.

Task 9 init 13 shows an ATM/OHB regression. FP16 succeeds in `612` frames, while
ATM/OHB and `none` run to horizon. This suggests that even the stabilized
quantized policy can shift a successful FP16 trajectory across a task-specific
decision boundary.

Follow-up regression cases split into three visual failure modes:

| category | representative cases | video-level observation |
| --- | --- | --- |
| early object-container relation drift from `none` | task 0 init 3, task 0 init 7 | FP16/ATM-OHB finish quickly, while `none` disrupts the early object/container relationship and spends the rest of the rollout in ineffective correction |
| multi-object order or target-selection drift from `none` | task 7 init 3, task 7 init 10 | successful rows quickly move the relevant item into the container; `none` remains in repeated approach/correction and does not close the multi-object sequence |
| ATM/OHB stable but under-progressing path | task 0 init 8, task 6 init 1, task 9 init 9 | FP16 and/or `none` finish quickly, while ATM/OHB does not look wildly unstable but fails to complete the decisive push/place step |
| quantized early branch in the wrong direction | task 8 init 0 | FP16 succeeds, but both quantized rows run to horizon; this is the mirror image of the task 8 repair cases |

The regression cases are useful because they prevent the over-simple story that
raw quantization is always beneficial exploration or that ATM/OHB is always a
stabilizer. The same mechanisms can help or hurt depending on the initial
state: `none` can escape a failure basin or disturb an already good object
relationship; ATM/OHB can suppress harmful drift or keep the rollout on a path
that is too conservative to complete the task.

## Working Hypothesis

The simulator results are best described as quantization-induced trajectory
redistribution, not monotonic accuracy improvement.

At the offline level, ATM/OHB reduces average held-out error. At the closed-loop
simulator level, small action differences can change the rollout basin. Some
changes repair FP16 failures by escaping a bad attractor; others create drift
that prevents otherwise successful completion.

The key failure variable appears to be early trajectory branching rather than
only final placement precision. Successful repaired cases often terminate much
earlier than the corresponding failures, and the follow-up cases make this
pattern stable across multiple init states.

The regression follow-ups add a second constraint: not every failure is the same
kind of branch. For task 0 and task 7, raw `none` failures often look like early
object-relation or multi-object sequencing drift. For task 6 and task 9,
ATM/OHB-specific failures often look more like stable but insufficient progress
near a target. Task 8 has both directions: quantization repairs many FP16
failures, but task 8 init 0 shows that quantization can also push an FP16
success across the wrong side of the same decision boundary.

## Next Checks

1. For selected cases, export action traces if available: end-effector xyz,
   rotation, gripper state, and action chunk norm over time.
2. Compare frame-time of first object contact and first goal-region entry
   across policies.
3. If rerunning selected cases, prioritize trace capture for:
   task 8 init 7, task 8 init 9, task 4 init 10, task 0 init 3, task 6 init 1,
   and task 9 init 9.
