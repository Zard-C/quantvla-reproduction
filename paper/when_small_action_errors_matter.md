# When Small Action Errors Matter: Closed-Loop Analysis of Post-Training Quantization for VLA Policies

Working draft.

patrick.zhang  
patrick.zhang5233@gmail.com

## Abstract

Post-training quantization (PTQ) is usually presented as a compression procedure: a full-precision model is mapped into a low-precision representation, and the resulting model is evaluated by output drift, task accuracy, or deployment efficiency. For vision-language-action (VLA) policies, this framing is incomplete. A VLA model is not only a predictor; it is a closed-loop controller. We argue that PTQ for VLA policies should be evaluated as a closed-loop policy perturbation. Quantization changes the implemented policy function, which can change the state distribution induced by robot-environment interaction. Small action errors can therefore be amplified by contact dynamics, receding-horizon feedback, and thresholded success conditions.

We study this effect in a reproduction-oriented analysis of selective W4A8 fake quantization for a GR00T N1.5 policy on LIBERO-10. Offline teacher-student action drift shows that attention balancing methods can improve mean action error, but also worsen a non-trivial subset of held-out observations. In closed-loop simulation over 150 matched task-initialization pairs, selective W4A8 variants remain in the FP16 behavioral range under our evaluation protocol, but their gains are not monotonic: compensation methods repair some failed rollouts while introducing new failures elsewhere. The highest point-estimate single compensation mode, output head balancing (OHB), reaches 116/150 successes compared with 113/150 for uncompensated W4A8 and 108/150 for FP16, yet it still regresses several task slices. We do not claim statistically significant closed-loop superiority from these small aggregate gaps. Our study focuses on behavior under fake quantization rather than packed-kernel deployment efficiency. The results suggest that VLA quantization should be evaluated as policy perturbation, not merely as numerical approximation.

## 1. Introduction

Large vision-language-action models are increasingly used as generalist robot policies. They map visual observations and language instructions into action chunks, often through diffusion-style denoising or transformer action heads. As these models become larger, quantization is a natural route toward cheaper deployment. However, common quantization evaluations focus on local approximation quality: weight error, activation error, output MSE, or single-step action drift. These metrics are useful, but they do not fully capture what matters for embodied control.

In a closed-loop policy, an action error at time `t` changes the next observation. The next action is then produced on a state that may not have been visited by the original full-precision policy. This feedback loop creates a mismatch between local teacher-student error and final task success. The issue is especially sharp in manipulation, where small geometric deviations can change contact timing, grasp stability, object pose, or whether a state satisfies a binary success predicate.

This paper studies the following question:

> When a VLA policy is post-training quantized, how should we interpret small action errors in terms of closed-loop task success?

Our answer is that PTQ should be treated as a policy perturbation. It maps a full-precision policy into a constrained low-precision function class, thereby changing both the conditional action distribution and the induced state distribution. The relevant empirical object is not only local action drift on fixed observations, but also how the perturbed policy redistributes success and failure across matched closed-loop rollouts.

We make three observations:

1. Selective W4A8 quantization can be behaviorally robust even when it measurably changes action outputs.
2. Attention-balancing corrections can improve mean offline drift while still worsening individual observations and task slices.
3. Paired closed-loop rollout flips expose repair/regression structure that aggregate success rates hide.

Our contributions are:

1. We formulate PTQ for VLA policies as a closed-loop policy perturbation problem, where the main object is the induced state distribution rather than only local action approximation.
2. We provide a reproduction-oriented GR00T/LIBERO case study showing that selective W4A8 fake quantization can remain in the FP16 behavioral range while changing which task-initialization pairs succeed.
3. We show that offline mean drift and aggregate success rate can both obscure important structure, and advocate paired repair/regression analysis as a compact diagnostic for VLA quantization.

### Scope of Claims

This paper should be read as a behavior-level reproduction study and diagnostic analysis, not as a new quantization algorithm or a deployment benchmark. We make no latency, memory, or energy claim because our implementation uses fake quantization rather than packed low-precision kernels. We also do not treat small aggregate success-rate differences as evidence of policy dominance. The main claim is narrower: VLA quantization can preserve aggregate success while redistributing which closed-loop rollouts succeed, so paired rollout analysis is necessary to interpret the effect of small action perturbations.

## 2. Related Work

### Vision-Language-Action Policies

Recent robot policies increasingly combine large-scale visual, language, and action modeling. RT-1 introduced a scalable transformer policy trained on diverse real-robot data, while RT-2 framed robotic actions as tokens inside a vision-language-action model and showed transfer from web-scale vision-language training to control [RT-1; RT-2]. OpenVLA made this direction more accessible by releasing a 7B-parameter open-source VLA trained on large robot demonstration mixtures, and also noted practical interest in efficient fine-tuning and quantized serving [OpenVLA]. Diffusion Policy and related action-diffusion methods model robot behavior as conditional denoising, often with receding-horizon control, which makes local output perturbations especially relevant to closed-loop behavior [Diffusion Policy]. Our work focuses on the evaluation problem that appears when such policies are compressed after training.

### Post-Training Quantization

PTQ methods for large transformers often target memory and latency while preserving model quality. LLM.int8() handles transformer outliers through mixed-precision decomposition [LLM.int8]; GPTQ uses approximate second-order information for one-shot low-bit weight quantization [GPTQ]; SmoothQuant migrates quantization difficulty from activations to weights through an equivalent smoothing transform [SmoothQuant]; and AWQ uses activation statistics to protect salient weight channels [AWQ]. These methods demonstrate that transformer quantization can be accurate when calibration accounts for activation structure. However, most evaluations are open-loop language, vision-language, or recognition tasks. In VLA policies, the quantized model selects actions that alter future inputs, so the calibration and evaluation target must include closed-loop effects.

### Quantization For VLA Models

QuantVLA is, to our knowledge, the most direct prior work on PTQ for VLA systems. It proposes a scale-calibrated PTQ framework with selective quantization, attention temperature matching, and output head balancing, and reports memory and latency gains together with LIBERO success-rate improvements [QuantVLA]. Our study is not a deployment-efficiency reproduction of those claims. Instead, we use a behavior-level fake-quantization reproduction to analyze why VLA quantization can improve some rollouts while regressing others, and why paired closed-loop evaluation is necessary even when average offline drift improves.

### Sequential Prediction And Distribution Shift

The gap between offline error and closed-loop behavior is a classic issue in imitation learning and sequential prediction. DAgger formalizes the problem that future observations depend on previous actions, so policies trained or evaluated only under a fixed data distribution can suffer from compounding errors [DAgger]. VLA quantization creates a related but distinct problem: the policy is already trained, but PTQ perturbs its action outputs and therefore perturbs the state distribution it induces. This makes paired rollout analysis a natural complement to offline teacher-student action drift.

### Feedback Control Perspective

Classical feedback control emphasizes that closed-loop behavior is determined by the interaction between controller perturbations, plant dynamics, and feedback gain [Astrom and Murray]. We use this lens to interpret PTQ not as a static approximation problem, but as a bounded policy perturbation whose effect depends on local closed-loop sensitivity and task margins.

## 3. Quantization As Policy Perturbation

Let `pi_theta(a | s, l)` denote a full-precision VLA policy conditioned on state observation `s` and language instruction `l`. PTQ constructs a quantized policy:

```text
theta_q = Q(theta; C)
pi_q = pi_{theta_q}
```

where `Q` is a calibration-dependent projection into a low-precision parameterization and `C` is a calibration set. Even without gradient updates, this is not a neutral storage transform. It changes the implemented function.

Offline action drift measures a local difference such as:

```text
e(s) = || pi_q(s, l) - pi_theta(s, l) ||
```

on observations sampled from some dataset or teacher rollout distribution. Closed-loop performance depends on the state distributions induced by the two policies:

```text
d_{pi_theta}(s)  vs.  d_{pi_q}(s)
```

The difference is recursive. For a local transition model:

```text
s_{t+1} = f(s_t, a_t)
a_t     = pi(s_t)
```

a first-order error expansion gives:

```text
delta s_{t+1}
  approximately J_s delta s_t + J_a delta a_t
```

where `delta a_t = pi_q(s_t) - pi_theta(s_t)`. Even if `delta a_t` is small on the teacher state distribution, the accumulated `delta s_t` can move the quantized policy into states where the original local error estimate no longer applies.

This is why manipulation success is not a smooth function of single-step action MSE. A rollout can flip because of a small action perturbation if that perturbation changes:

- whether the gripper reaches the object before closing,
- whether a placed object crosses a success-region boundary,
- whether an object contact produces a different pose,
- whether a receding-horizon action chunk enters a different trajectory basin.

In this view, quantization is closer to post-training policy editing than to passive compression.

Figure 1 in the LaTeX/PDF version illustrates the central evaluation mismatch with a native TikZ diagram. Offline action drift compares two actions on the same observation, while closed-loop execution lets each policy induce its own future observation distribution.

## 4. Experimental Setup

### Model And Task

We use the GR00T N1.5 LIBERO long-horizon checkpoint and evaluate on LIBERO-10. The accepted FP16 baseline matches the official reference result of 38/50 on the standard 5-trial LIBERO-10 run.

The evaluated policy predicts action chunks with seven action components:

```text
x, y, z, roll, pitch, yaw, gripper
```

Unless noted otherwise, experiments use 8 denoising steps.

### Quantization Scope

We study the selective `llm_dit_mlp` W4A8 fake-quantization scope:

| group | modules |
|---|---:|
| selected LLM linear layers | 84 |
| selected DiT MLP linear layers | 32 |
| total quantized modules | 116 |

DiT attention projections are intentionally left in floating point. This design matters: attention projections feed a softmax routing mechanism, while MLP errors are more often residual perturbations. The selective scope therefore tests a method-level idea rather than naive whole-model quantization.

### Compensation Modes

Following QuantVLA terminology, we evaluate two scale-calibrated compensation mechanisms: attention temperature matching (ATM) and output head balancing (OHB). Our implementation should be read as a reproduction-oriented approximation of these mechanisms rather than a claim of exact code-level equivalence to the original implementation.

ATM rescales the DiT attention query to match teacher/student attention-logit standard deviation:

```text
alpha = std_teacher(attention_logits) / std_student(attention_logits)
query <- alpha * query
```

OHB rescales the attention output before residual addition to match teacher/student attention-output RMS:

```text
beta = rms_teacher(attention_output) / rms_student(attention_output)
attention_output <- beta * attention_output
```

The combined mode applies both corrections. In all cases the compensation acts on DiT attention processors; it does not mean DiT attention weights are quantized.

### Offline Drift Protocol

The offline validation uses real LIBERO observations:

| item | value |
|---|---:|
| calibration observations | 16 |
| held-out evaluation observations | 128 |
| quantized modules | 116 |
| activation scale | dynamic absmax |
| denoising steps | 8 |

Teacher and student calls use matched random seeds because action denoising begins from random Gaussian actions. We report NMSE, relative RMSE, cosine similarity, and max absolute difference.

### Closed-Loop Protocol

The main closed-loop analysis evaluates LIBERO-10 initial states `0..14`:

| item | value |
|---|---:|
| tasks | 10 |
| initial states per task | 15 |
| episodes per policy/mode | 150 |
| simulator | LIBERO headless EGL |
| hardware | RTX 5090 |

For paired comparisons, each mode is evaluated on the same task-initialization pairs. This enables us to count repaired failures and new regressions directly.

We report Wilson 95% confidence intervals for aggregate success rates as a descriptive uncertainty estimate, but use paired repair/regression counts as the primary behavioral diagnostic.

## 5. Results

### 5.1 Offline Mean Drift Improves, But Individual Regressions Remain

On the 128 held-out real observations, ATM and OHB reduce mean action drift relative to uncompensated W4A8.

| config | mode | modules | NMSE mean | rel RMSE mean | cosine mean | max abs diff |
|---|---|---:|---:|---:|---:|---:|
| `llm_dit_mlp` | none | 116 | 0.0178962 | 0.0981458 | 0.992492 | 0.989746 |
| `llm_dit_mlp` | identity | 116 | 0.0178962 | 0.0981458 | 0.992492 | 0.989746 |
| `llm_dit_mlp` | ATM | 116 | 0.0159471 | 0.0904183 | 0.993101 | 0.987793 |
| `llm_dit_mlp` | OHB | 116 | 0.0160919 | 0.0853958 | 0.992769 | 0.990967 |
| `llm_dit_mlp` | ATM+OHB | 116 | 0.0153168 | 0.0838784 | 0.993048 | 0.987061 |

The identity control exactly matches `none`, showing that the custom attention processor path itself does not introduce measurable drift. However, the mean improvement hides observation-level regressions.

| mode | observations worse than `none` by NMSE | mean delta NMSE | mean delta rel RMSE |
|---|---:|---:|---:|
| identity | 0/128 | 0 | 0 |
| ATM | 37/128 | -0.00194913 | -0.00772748 |
| OHB | 24/128 | -0.00180429 | -0.01275 |
| ATM+OHB | 34/128 | -0.00257947 | -0.0142674 |

Negative deltas mean improvement over `none`. Thus ATM+OHB has the best mean drift, while still worsening 34 of 128 held-out observations.

This is the first sign that mean offline drift is insufficient. The compensation moves the action distribution closer on average, but not uniformly.

![Figure 2: Offline mean drift versus observation-level regressions.](figures/fig2_offline_drift_regressions.svg)

Figure 2 summarizes this non-monotonicity visually: the same compensation mode can improve mean relative RMSE while making a subset of held-out observations worse by NMSE.

### 5.2 Selective W4A8 Remains in the FP16 Range

Over 150 LIBERO-10 task-initialization pairs, selective W4A8 remains in the same performance band as FP16. Some quantized variants have higher aggregate point estimates, but these gaps should not be read as evidence that quantization is inherently better than FP16.

| policy | successes | success rate (Wilson 95% CI) |
|---|---:|---:|
| FP16 | 108/150 | 72.0% [64.3, 78.6] |
| W4A8 `llm_dit_mlp` + none | 113/150 | 75.3% [67.9, 81.5] |
| W4A8 `llm_dit_mlp` + ATM | 114/150 | 76.0% [68.6, 82.1] |
| W4A8 `llm_dit_mlp` + OHB | 116/150 | 77.3% [70.0, 83.3] |
| W4A8 `llm_dit_mlp` + ATM+OHB | 114/150 | 76.0% [68.6, 82.1] |

The confidence intervals are wide and overlapping, so the table should not be read as a statistical ranking of policies. The more stable conclusion is that selective W4A8 remains in the same behavioral range as FP16 under this protocol, while changing the identity of successful and failed rollouts.

Rather, the quantized policies perturb the trajectory distribution. In finite closed-loop evaluation, that perturbation can push some failed FP16 trajectories into successful basins while pushing other successful trajectories into failure.

### 5.3 Aggregate Rates Hide Task-Level Redistribution

The following table shows that the effects are task-dependent.

| task id | FP16 | none | ATM | OHB | ATM+OHB |
|---:|---:|---:|---:|---:|---:|
| 0 | 13/15 | 10/15 | 11/15 | 13/15 | 13/15 |
| 1 | 11/15 | 13/15 | 13/15 | 14/15 | 14/15 |
| 2 | 12/15 | 13/15 | 14/15 | 15/15 | 13/15 |
| 3 | 15/15 | 15/15 | 13/15 | 13/15 | 15/15 |
| 4 | 8/15 | 8/15 | 13/15 | 12/15 | 12/15 |
| 5 | 15/15 | 14/15 | 14/15 | 15/15 | 15/15 |
| 6 | 6/15 | 9/15 | 10/15 | 8/15 | 7/15 |
| 7 | 11/15 | 8/15 | 8/15 | 8/15 | 10/15 |
| 8 | 3/15 | 9/15 | 4/15 | 6/15 | 3/15 |
| 9 | 14/15 | 14/15 | 14/15 | 12/15 | 12/15 |

ATM improves task 4 by 5 successes over `none`, but regresses task 8 by 5 successes. OHB is more balanced and gives the highest aggregate point estimate, but it still regresses tasks 3, 8, and 9 relative to `none`.

### 5.4 Paired Rollout Flips Reveal The Mechanism

Paired comparisons over identical task-initialization pairs show that the compensation modes are not monotonic improvements.

| comparison | repaired failures | new regressions | same success | same failure | net |
|---|---:|---:|---:|---:|---:|
| ATM vs none | 14 | 13 | 100 | 23 | +1 |
| OHB vs none | 16 | 13 | 100 | 21 | +3 |
| OHB vs ATM | 15 | 13 | 101 | 21 | +2 |

This table is more informative than the aggregate success rates. For example, OHB is not simply "3 episodes better" than `none`; it repairs 16 failures and introduces 13 new failures. Its net gain is small, but the underlying trajectory redistribution is substantial.

For the disjoint init `5..14` comparison between FP16 and ATM+OHB W4A8, we observe the same pattern:

| transition | count |
|---|---:|
| FP16 success, quant success | 62 |
| FP16 failure, quant failure | 16 |
| FP16 failure, quant success | 14 |
| FP16 success, quant failure | 8 |

The aggregate gain of 6/100 comes from 14 repaired FP16 failures minus 8 new quantized failures.

![Figure 3: Paired closed-loop rollout flips.](figures/fig3_paired_rollout_flips.svg)

Figure 3 is the main behavioral diagnostic. It separates net success-rate changes into repaired failures, new regressions, unchanged successes, and unchanged failures.

## 6. Analysis

### 6.1 Why Can A Quantized Policy Improve A Rollout?

The quantized policy is not guaranteed to be worse than FP16 in a finite simulator benchmark. FP16 is not an oracle; it is a learned policy with its own fragile regions. A small action perturbation can move a trajectory away from a failure mode. This is especially plausible when the FP16 policy is already weak on a task slice, such as LIBERO task 8 in our baseline.

Therefore, an aggregate improvement should not be read as "low precision is better." It means the perturbed policy enters different trajectory basins, some of which are more favorable under the benchmark's finite initial states and success predicates.

### 6.2 Why Does Offline Drift Fail To Predict Closed-Loop Behavior?

Offline drift is measured on a fixed observation set. Closed-loop rollouts produce observations adaptively. Once a quantized action changes the environment state, later observations may differ from the offline distribution. This creates two mismatches:

1. The local action error estimate may not apply to the new states.
2. The success function may be discontinuous around contact and placement boundaries.

This explains why ATM+OHB can have the best mean offline NMSE but not the best closed-loop aggregate success. It also explains why a mode can improve mean drift while still causing rollouts to regress.

### 6.3 A Control-Theoretic Interpretation

From a feedback-control perspective, quantization acts like a structured input disturbance injected into the learned controller:

```text
a_t^q = pi_theta(s_t^q, l) + eta(s_t^q, l)
```

where `eta` is the action perturbation induced by quantization and compensation. Linearizing around an FP16 trajectory gives:

```text
delta s_{t+1}
  approximately (A_t + B_t K_t) delta s_t + B_t eta_t
```

where `A_t = partial f / partial s`, `B_t = partial f / partial a`, and `K_t = partial pi_theta / partial s` along the trajectory. The closed-loop sensitivity is therefore governed not only by the magnitude of `eta_t`, but also by the local closed-loop gain induced by the environment dynamics and the policy feedback. A small action perturbation can be attenuated in low-gain regions and amplified near contact, grasping, or placement transitions.

This also makes success a margin problem. Let `h(s_T)` denote an implicit terminal success margin, with success when `h(s_T) > 0`. A rollout with large positive margin can tolerate sizeable action perturbations, while a low-margin rollout can flip if quantization moves `h(s_T)` across zero. This explains why paired outcomes contain both repairs and regressions: quantization can move a trajectory away from one failure basin while pushing another trajectory across a success boundary. The useful diagnostic is therefore not just average action error, but the alignment between action perturbations, closed-loop sensitivity, and task success margins.

### 6.4 Why Is Selective W4A8 Robust?

The `llm_dit_mlp` scope leaves DiT attention projections in floating point. This avoids quantizing the most routing-sensitive part of the action head. A perturbation to `QK^T` before softmax can change attention entropy and token routing:

```text
softmax(QK^T / sqrt(d))
```

By contrast, MLP quantization more often enters as an additive residual perturbation:

```text
y_student = y_teacher + epsilon
```

Layer normalization, residual connections, diffusion denoising, and receding-horizon replanning can absorb some of this error. This helps explain why uncompensated W4A8 already reaches 113/150 successes.

### 6.5 Why Are ATM And OHB Not Additive?

ATM changes where attention looks by changing the attention-logit scale. OHB changes how strongly the attention output enters the residual stream. These operations are coupled:

```text
ATM changes attention probabilities.
OHB rescales the output produced by those probabilities.
```

If ATM moves routing into a different pattern, OHB rescales a different output than the one calibrated under the uncompensated student. The combined correction can overcorrect, undercorrect, or move the policy into a different trajectory basin. This is consistent with ATM+OHB reaching 114/150, below standalone OHB at 116/150.

### 6.6 What Should Be Reported For VLA Quantization?

The minimum evidence package should include:

- offline action drift on held-out real observations,
- per-component action error, not only whole-action MSE,
- closed-loop success on matched task-initialization pairs,
- per-task and per-init success rates,
- paired repair/regression counts,
- explicit distinction between fake-quant behavior and packed-kernel deployment.

Aggregate success is still useful, but it should not be the only behavioral result.

## 7. Implications

### For Evaluation

VLA quantization should be evaluated as a closed-loop policy perturbation. Reporting only MSE or only aggregate success can be misleading. Paired rollout flips are a compact way to expose whether a method is a monotonic improvement or a redistribution of successes.

### For Calibration

Calibration should not only minimize average layerwise or action-level error. It should consider which errors matter for downstream control. A useful calibration set should cover fragile contact states, low-margin success boundaries, and task slices where the FP16 policy is already weak.

A control-aware calibration objective would weight errors by estimated closed-loop sensitivity, not only by their open-loop magnitude.

### For Fine-Tuning

Quantization-aware fine-tuning should be understood as policy adaptation under low-precision constraints. If the objective is only behavior cloning on static observations, it may still miss closed-loop state distribution shift. More policy-aware objectives may be needed, such as rollout-informed calibration, DAgger-style data aggregation, or task-slice reweighting.

### For Deployment Claims

Our results are behavior-level evidence for fake W4A8 quantization. They do not establish latency, memory, or energy improvements. A complete deployment claim requires packed low-precision kernels and end-to-end inference profiling.

## 8. Limitations

This study has several limitations:

- The quantized implementation uses fake quantization, not a final packed int4/int8 deployment backend.
- Results are from one GR00T N1.5 checkpoint and LIBERO-10, not multiple VLA families.
- The main closed-loop benchmark has 150 task-initialization pairs; larger runs would provide tighter statistical confidence, and the current aggregate success intervals overlap.
- We do not run real-robot experiments.
- ATM/OHB are implemented as reproduction-oriented calibration mechanisms; exact implementation details may differ from the original paper.
- We evaluate success-rate flips, but do not yet provide trajectory-level geometric diagnostics for each flip.

These limitations do not weaken the central methodological point: local action approximation and closed-loop policy behavior are distinct evaluation targets.

## 9. Conclusion

Post-training quantization of VLA policies should be viewed as policy perturbation. Even small action errors can matter because they enter a feedback loop, alter the closed-loop state distribution, and interact with task-dependent success margins. In our GR00T/LIBERO reproduction, selective W4A8 quantization remains behaviorally robust and can shift aggregate point estimates, but those shifts come from task-dependent repair/regression redistribution rather than uniform dominance. Attention compensation methods reduce mean offline drift, yet their closed-loop effects are non-monotonic.

The main lesson is practical: VLA quantization papers should report paired closed-loop rollouts, not only offline drift or aggregate success. Small action errors matter when they change the trajectory basin.

## Appendix A. Detailed Experimental Tables

### A.1 Standard 5-Trial FP16 And ATM+OHB W4A8

| task id | FP16 | W4A8 ATM+OHB | delta |
|---:|---:|---:|---:|
| 0 | 5/5 | 5/5 | 0 |
| 1 | 3/5 | 4/5 | +1 |
| 2 | 3/5 | 5/5 | +2 |
| 3 | 5/5 | 5/5 | 0 |
| 4 | 4/5 | 4/5 | 0 |
| 5 | 5/5 | 5/5 | 0 |
| 6 | 3/5 | 2/5 | -1 |
| 7 | 4/5 | 3/5 | -1 |
| 8 | 1/5 | 0/5 | -1 |
| 9 | 5/5 | 5/5 | 0 |
| total | 38/50 | 38/50 | 0 |

### A.2 Disjoint Init 5..14 FP16 And ATM+OHB W4A8

| task id | FP16 | W4A8 ATM+OHB | delta |
|---:|---:|---:|---:|
| 0 | 8/10 | 8/10 | 0 |
| 1 | 8/10 | 10/10 | +2 |
| 2 | 9/10 | 8/10 | -1 |
| 3 | 10/10 | 10/10 | 0 |
| 4 | 4/10 | 8/10 | +4 |
| 5 | 10/10 | 10/10 | 0 |
| 6 | 3/10 | 5/10 | +2 |
| 7 | 7/10 | 7/10 | 0 |
| 8 | 2/10 | 3/10 | +1 |
| 9 | 9/10 | 7/10 | -2 |
| total | 70/100 | 76/100 | +6 |

### A.3 Ablation: W4A8 None, ATM, OHB

| task id | none | ATM | OHB | ATM - none | OHB - none |
|---:|---:|---:|---:|---:|---:|
| 0 | 10/15 | 11/15 | 13/15 | +1 | +3 |
| 1 | 13/15 | 13/15 | 14/15 | 0 | +1 |
| 2 | 13/15 | 14/15 | 15/15 | +1 | +2 |
| 3 | 15/15 | 13/15 | 13/15 | -2 | -2 |
| 4 | 8/15 | 13/15 | 12/15 | +5 | +4 |
| 5 | 14/15 | 14/15 | 15/15 | 0 | +1 |
| 6 | 9/15 | 10/15 | 8/15 | +1 | -1 |
| 7 | 8/15 | 8/15 | 8/15 | 0 | 0 |
| 8 | 9/15 | 4/15 | 6/15 | -5 | -3 |
| 9 | 14/15 | 14/15 | 12/15 | 0 | -2 |
| total | 113/150 | 114/150 | 116/150 | +1 | +3 |

![Figure 4: Task-wise success redistribution across policy variants.](figures/fig4_task_redistribution.svg)

Figure 4 provides the detailed task-wise view behind the aggregate closed-loop results.

## Appendix B. Figure Source Notes

Figure 1 is a native LaTeX/TikZ diagram in `paper/main.tex`, based on the policy-perturbation framing in Section 3.

Figure 2 uses the random 128 held-out offline validation and regression analysis from:

- `docs/phase4_real_data_validation_d8_cal16_eval128_random.md`
- `docs/phase4_real_data_validation_d8_cal16_eval128_random_regressions.md`

Figure 3 uses the paired ablation counts from `docs/phase5_llm_dit_mlp_ablation_init0_14.md`.

Figure 4 uses the combined per-task success table from Section 5.3 and Appendix A.

All SVGs are generated by `paper/generate_figures.py`.

## References

- [QuantVLA] Jingxuan Zhang et al. "QuantVLA: Scale-Calibrated Post-Training Quantization for Vision-Language-Action Models." arXiv:2602.20309, 2026. https://arxiv.org/abs/2602.20309
- [GR00T N1] NVIDIA et al. "GR00T N1: An Open Foundation Model for Generalist Humanoid Robots." arXiv:2503.14734, 2025. https://arxiv.org/abs/2503.14734
- [LIBERO] Bo Liu et al. "LIBERO: Benchmarking Knowledge Transfer for Lifelong Robot Learning." arXiv:2306.03310, 2023. https://arxiv.org/abs/2306.03310
- [RT-1] Anthony Brohan et al. "RT-1: Robotics Transformer for Real-World Control at Scale." arXiv:2212.06817, 2022. https://arxiv.org/abs/2212.06817
- [RT-2] Anthony Brohan et al. "RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control." arXiv:2307.15818, 2023. https://arxiv.org/abs/2307.15818
- [OpenVLA] Moo Jin Kim et al. "OpenVLA: An Open-Source Vision-Language-Action Model." arXiv:2406.09246, 2024. https://arxiv.org/abs/2406.09246
- [Diffusion Policy] Cheng Chi et al. "Diffusion Policy: Visuomotor Policy Learning via Action Diffusion." arXiv:2303.04137, 2023. https://arxiv.org/abs/2303.04137
- [LLM.int8] Tim Dettmers et al. "LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale." arXiv:2208.07339, 2022. https://arxiv.org/abs/2208.07339
- [GPTQ] Elias Frantar et al. "GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers." arXiv:2210.17323, 2022. https://arxiv.org/abs/2210.17323
- [SmoothQuant] Guangxuan Xiao et al. "SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models." arXiv:2211.10438, 2022. https://arxiv.org/abs/2211.10438
- [AWQ] Ji Lin et al. "AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration." arXiv:2306.00978, 2023. https://arxiv.org/abs/2306.00978
- [DAgger] Stephane Ross, Geoffrey J. Gordon, and J. Andrew Bagnell. "A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning." arXiv:1011.0686, 2010. https://arxiv.org/abs/1011.0686
