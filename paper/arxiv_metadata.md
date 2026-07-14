# arXiv Metadata Draft

## Title

Inference Acceleration as Closed-Loop Perturbation: Closed-Loop Sensitivity-Guided Tactic Search for VLA Policies

## Authors

patrick.zhang

## Suggested Categories

- Primary: `cs.RO`
- Cross-list candidates: `cs.LG`, `cs.CV`

## Suggested Comments

Method and behavior-level study of closed-loop sensitivity-guided tactic search for VLA inference acceleration.

## Suggested License

Choose one during upload:

- `CC BY 4.0` if you want broad reuse with attribution.
- arXiv default license if you prefer a more conservative option.

## Abstract

Inference acceleration for vision-language-action (VLA) policies is usually framed as a local systems problem: quantize weights, compile graphs, fuse kernels, or replay static execution paths, then check numerical drift and serving latency. This framing misses the central difficulty for robot policies: the accelerated model is a controller inside a feedback loop, so small implementation perturbations can alter future observations, contact timing, receding-horizon replanning, and success margins.

We formulate post-training VLA acceleration as a closed-loop policy perturbation problem and propose Closed-Loop Sensitivity-Guided Tactic Search (CLSG-TS), a rollout-budget-aware procedure for selecting acceleration tactics. The method prunes candidate tactics with cheap filters, evaluates survivors on matched fragile rollouts, estimates closed-loop risk through paired repairs/regressions and first-divergence diagnostics, and selects behavior-first, speed-constrained, or routed deployment points on held-out speed--risk frontiers. On GR00T policies in LIBERO-10, speed-only action-head compilation gives a 2.23x median server-latency speedup on a 33-case probe but reduces success from 19/33 to 16/33. CLSG-TS recovers behavior with early-block and duration fallbacks, while held-out folds and a GR00T N1.7 check show that no single local winner is universal. The contribution is a method for VLA acceleration under closed-loop risk, not a claim of universal backend superiority.
