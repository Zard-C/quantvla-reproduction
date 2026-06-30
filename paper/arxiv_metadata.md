# arXiv Metadata Draft

## Title

Inference Acceleration as Closed-Loop Perturbation: Sensitivity-Guided Speedups for VLA Policies

## Authors

patrick.zhang

## Suggested Categories

- Primary: `cs.RO`
- Cross-list candidates: `cs.LG`, `cs.CV`

## Suggested Comments

Behavior-level study of inference acceleration for VLA policies, including fake quantization, graph compilation, sensitivity probes, and matched closed-loop rollouts.

## Suggested License

Choose one during upload:

- `CC BY 4.0` if you want broad reuse with attribution.
- arXiv default license if you prefer a more conservative option.

## Abstract

Inference acceleration for vision-language-action (VLA) policies is often treated as a systems problem: quantize weights, compile graphs, fuse kernels, or replay static execution paths, then report drift, success, latency, and memory. This framing is incomplete for robot policies. A VLA model is a closed-loop controller, so an implementation perturbation changes not only the current action but also the future state distribution induced by interaction with the environment. Small numerical differences can therefore be filtered, amplified, repaired, or inverted by feedback dynamics, contact, receding-horizon replanning, and thresholded task margins.

We formulate post-training VLA acceleration as a closed-loop policy perturbation problem and use this lens to derive five claims: acceleration error is filtered by closed-loop sensitivity; rollout flips are margin-crossing events; open-loop drift is insufficient without state-distribution control; sensitivity is anisotropic across action channels, rollout phases, and model boundaries; and acceleration should be viewed as exploration of a constrained policy-implementation design space rather than imitation of a single FP16 point. We test these claims on a GR00T N1.5 policy in LIBERO-10 through selective W4A8 fake quantization, torch.compile action-head acceleration, eager-island protection, controlled action perturbations, first-divergence trace analysis, and matched rollout repair/regression accounting. A finer duration search identifies a narrow early eager window, policy steps 0--120, that restores FP16-level success on a 33-case set while preserving speed-only latency. We do not claim final packed-kernel deployment efficiency or universal superiority of any single backend. The contribution is a modeling and experimental guide for VLA acceleration: identify closed-loop-sensitive dimensions, phases, and modules, then protect the smallest regions that improve paired repair/regression while preserving speed.
