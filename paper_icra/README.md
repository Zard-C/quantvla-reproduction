# ICRA Submission Draft

This directory contains the anonymous ICRA-style version of the paper:

> Inference Acceleration as Closed-Loop Perturbation: Sensitivity-Guided Speedups for VLA Policies

The full technical report and arXiv-facing source remain in `../paper/`. This draft is intentionally shorter and should converge toward an 8-page robotics conference paper plus references and optional video.

The current LaTeX file uses a two-column `article` fallback because the local TeX environment does not include `ieeeconf.cls`. Before formal submission, replace the fallback class with the official ICRA/PaperCept template released for the target year.

## Build

```bash
cd paper_icra
make
```

If `tectonic` needs network access on the local machine, run it through the local proxy setup:

```bash
zsh -ic 'proxy_on; make'
```

## Current Intent

- Anonymous conference draft, no author identity or artifact links in the main text.
- Main line: VLA inference acceleration is a closed-loop policy perturbation problem.
- Core evidence: W4A8 redistribution, action/time/layer sensitivity probes, the 33-case `0--120` duration-proxy discovery, N1.5 multi-fold tactic validation, and the small N1.7 held-out routing check.
- Current method object: Closed-Loop Sensitivity-Guided Tactic Search, including behavior-first, speed-constrained, and routed-tactic deployment choices.
- Long tables, keyframe contact sheets, code links, and artifact notes should stay in the arXiv report or supplementary material unless space allows.
