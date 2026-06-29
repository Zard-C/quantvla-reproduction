# QuantVLA Reproduction Study

This repository contains a reproduction-oriented study of inference acceleration for vision-language-action (VLA) robot policies, centered on GR00T N1.5 and LIBERO-10 closed-loop rollouts. The project started from post-training quantization and expanded to graph compilation, eager islands, mixed precision, and other implementation-level acceleration boundaries.

## Paper

**When Inference Acceleration Changes Behavior: Closed-Loop Analysis for VLA Policies**
patrick.zhang

This paper argues that inference acceleration for VLA policies should be evaluated as a **closed-loop policy perturbation**, not only as static numerical approximation or systems throughput. Quantization, graph compilation, eager-island placement, and kernel replacement can all introduce small action-level perturbations that are filtered by feedback, contact dynamics, receding-horizon control, and thresholded success margins.

Links:

- [PDF](paper/main.pdf)
- [Current Chinese status note](docs/current_status_zh.md)
- [GitHub Release: paper-v1](https://github.com/Zard-C/quantvla-reproduction/releases/tag/paper-v1)
- [arXiv-ready source bundle](paper/dist/when_small_action_errors_matter_arxiv_v1.tar.gz)
- [Paper source](paper/main.tex)
- [Readable Markdown draft](paper/when_small_action_errors_matter.md)

Zenodo DOI will be added after the GitHub repository is enabled in Zenodo and the release is archived.

## Main Findings

- Selective W4A8 fake quantization over LLM and DiT MLP layers remains in the FP16 behavioral range on LIBERO-10 under the evaluated protocol, but its gains are not monotonic.
- Offline mean action drift can improve while individual held-out observations regress.
- Aggregate closed-loop success rates hide paired repair/regression structure.
- Attention temperature matching (ATM), output head balancing (OHB), and compile/eager-island boundaries redistribute successes rather than providing uniform dominance.
- Closed-loop sensitivity is anisotropic across action dimensions, rollout durations, and model-layer boundaries.
- From a control perspective, acceleration perturbations behave like structured input disturbances whose impact depends on closed-loop sensitivity and task success margins.

## Repository Map

- `paper/`: paper source, generated PDF, figures, release notes, and arXiv source bundle.
- `docs/`: staged experiment notes, rollout analyses, ablation summaries, and implementation reports.
- `toy_quantvla/`: scripts and utilities used for toy studies, GR00T probing, quantized inference services, and LIBERO evaluations.
- `analysis_keyframes/`: selected rollout keyframes used for qualitative analysis.

## Build The Paper

```bash
cd paper
make
```

The build expects `tectonic`, `rsvg-convert`, and Python 3. See [paper/README.md](paper/README.md) for details.

## Scope

This repository is a research reproduction and analysis workspace. The paper focuses on closed-loop behavior under fake quantization and prototype acceleration boundaries. It does not claim final packed-kernel deployment speedups, memory savings, or statistically significant policy superiority.
