# QuantVLA Reproduction Study

This repository contains a reproduction-oriented study of post-training quantization for vision-language-action (VLA) robot policies, centered on GR00T N1.5 and LIBERO-10 closed-loop rollouts.

## Paper

**When Small Action Errors Matter: Closed-Loop Analysis of Post-Training Quantization for VLA Policies**  
patrick.zhang

This paper argues that post-training quantization for VLA policies should be evaluated as a **closed-loop policy perturbation**, not only as static numerical approximation. Small action errors can be amplified by feedback, contact dynamics, receding-horizon control, and thresholded success margins.

Links:

- [PDF](paper/main.pdf)
- [GitHub Release: paper-v1](https://github.com/Zard-C/quantvla-reproduction/releases/tag/paper-v1)
- [arXiv-ready source bundle](paper/dist/when_small_action_errors_matter_arxiv_v1.tar.gz)
- [Paper source](paper/main.tex)
- [Readable Markdown draft](paper/when_small_action_errors_matter.md)

Zenodo DOI will be added after the GitHub repository is enabled in Zenodo and the release is archived.

## Main Findings

- Selective W4A8 fake quantization over LLM and DiT MLP layers remains in the FP16 behavioral range on LIBERO-10 under the evaluated protocol.
- Offline mean action drift can improve while individual held-out observations regress.
- Aggregate closed-loop success rates hide paired repair/regression structure.
- Attention temperature matching (ATM) and output head balancing (OHB) redistribute successes rather than providing monotonic improvements.
- From a control perspective, quantization behaves like a structured input disturbance whose impact depends on closed-loop sensitivity and task success margins.

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

This repository is a research reproduction and analysis workspace. The paper focuses on behavior under fake quantization and does not claim packed-kernel deployment speedups, memory savings, or statistically significant policy superiority.
