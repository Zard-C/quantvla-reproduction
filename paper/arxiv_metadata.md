# arXiv Metadata Draft

## Title

When Small Action Errors Matter: Closed-Loop Analysis of Post-Training Quantization for VLA Policies

## Authors

patrick.zhang

## Suggested Categories

- Primary: `cs.RO`
- Cross-list candidates: `cs.LG`, `cs.CV`

## Suggested Comments

22 pages including appendix. Reproduction-oriented study of post-training quantization for VLA policies.

## Suggested License

Choose one during upload:

- `CC BY 4.0` if you want broad reuse with attribution.
- arXiv default license if you prefer a more conservative option.

## Abstract

Post-training quantization (PTQ) is usually presented as a compression procedure: a full-precision model is mapped into a low-precision representation, and the resulting model is evaluated by output drift, task accuracy, or deployment efficiency. For vision-language-action (VLA) policies, this framing is incomplete. A VLA model is not only a predictor; it is a closed-loop controller. We argue that PTQ for VLA policies should be evaluated as a closed-loop policy perturbation. Quantization changes the implemented policy function, which can change the state distribution induced by robot-environment interaction. Small action errors can therefore be amplified by contact dynamics, receding-horizon feedback, and thresholded success conditions.

We study this effect in a reproduction-oriented analysis of selective W4A8 fake quantization for a GR00T N1.5 policy on LIBERO-10. Offline teacher-student action drift shows that attention balancing methods can improve mean action error, but also worsen a non-trivial subset of held-out observations. In closed-loop simulation over 150 matched task-initialization pairs, selective W4A8 variants remain in the FP16 behavioral range under our evaluation protocol, but their gains are not monotonic: compensation methods repair some failed rollouts while introducing new failures elsewhere. The highest point-estimate single compensation mode, output head balancing (OHB), reaches 116/150 successes compared with 113/150 for uncompensated W4A8 and 108/150 for FP16, yet it still regresses several task slices. We further show through controlled perturbation and acceleration-boundary experiments that closed-loop sensitivity is anisotropic across action dimensions, rollout phases, and model-layer boundaries. A small proxy-guided mixed-precision probe supports the resulting design rule: protecting closed-loop-sensitive action-head blocks can recover some speed-only regressions while preserving part of the latency gain, but it cannot eliminate trajectory redistribution. We do not claim statistically significant closed-loop superiority from these small aggregate gaps. Our study focuses on behavior under fake quantization and prototype acceleration boundaries rather than final packed-kernel deployment efficiency. The results suggest that VLA quantization should be evaluated as policy perturbation, not merely as numerical approximation.
