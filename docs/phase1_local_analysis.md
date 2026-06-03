# QuantVLA Phase 1 Local Analysis

Date: 2026-06-03

Objective: reproduce the method described in the QuantVLA paper, not blindly reproduce the official repository.

Primary sources:

- Paper: https://arxiv.org/abs/2602.20309
- Paper HTML: https://arxiv.org/html/2602.20309v4
- Project page: https://quantvla.github.io/
- Official repo reference commit: `9e435e56aac4e743526557e5d9ba6e4fddd6ccca`
- Official repo: https://github.com/AIoT-MLSys-Lab/QuantVLA

## 中文执行摘要

第一阶段结论：这个复现应该以论文方法为主线，而不是以官方仓库能否跑通为主线。QuantVLA 真正需要复现的核心不是一套脚本，而是三个机制是否成立：

1. 选择性量化：LLM 的线性层量化，DiT action head 只量化 MLP，DiT attention 的 `Q/K/V/O` 保持浮点。
2. ATM：用 teacher/student 的 attention logits 标准差比值，为每个 DiT attention head 学一个 `alpha`，把量化后的 logits 温度拉回 teacher。
3. OHB：用 teacher/student 的 post-output RMS 比值，为每个 DiT attention layer 学一个 `beta`，把 residual 注入能量拉回 teacher。

论文里有两个必须警惕的地方：

- `alpha = Std_T / Std_Q` 后，数学上应该让 `L_Q_calibrated = alpha * L_Q`。论文文字/公式有歧义，但官方代码也是乘 `alpha`。
- `beta = RMS_T / RMS_Q` 后，数学上应该让 `Z_Q_calibrated = beta * Z_Q`。论文里出现了类似除法的写法，和定义不一致；官方代码也是乘 `beta`。

所以第二阶段 toy 实验必须先验证 `alpha/beta` 的乘法方向，不能直接相信论文符号。只有 toy 证明方向和统计效果正确后，再上 GR00T/LIBERO。

## 0. Working Position

We treat the paper as the source of the method. The official code is a secondary artifact used to recover missing engineering details and identify possible traps.

The reproduction question is:

Can we independently implement and validate QuantVLA's core PTQ method, including selective layer quantization, Attention Temperature Matching, and Output Head Balancing, and then determine whether the reported VLA behavior is supported by experiments?

This means:

- We do not need to preserve the official repo's structure.
- We do not accept confusing comments or hard-coded paths as ground truth.
- We should prefer a clean implementation that is mathematically auditable.
- The official repo is useful for model names, env vars, calibration defaults, layer naming, and possible implementation intent.

## 1. Paper Claim Map

### Main Claim

QuantVLA is a training-free PTQ framework for VLA models with a DiT-style action head. It quantizes selected linear layers using W4A8, keeps the fragile DiT attention projections in floating point, and adds two scale calibration mechanisms:

- ATM: per-head attention temperature matching.
- OHB: per-layer output head energy balancing.

The paper claims this preserves or improves LIBERO success rate while reducing memory on the quantized LLM+DiT components.

### Reported Main Results

From the paper tables:

| Model | Setting | Spatial | Object | Goal | Long | Avg | Memory LLM+DiT |
|---|---:|---:|---:|---:|---:|---:|---:|
| pi0.5 | FP16 | 98.5 | 99.0 | 97.5 | 93.5 | 97.1 | 4.27 GB |
| pi0.5 + DuQuant LLM+DiT | W4A8 | 86.0 | 97.5 | 71.5 | 50.0 | 76.3 | 1.17 GB |
| pi0.5 + QuantVLA LLM only | W4A8 | 98.5 | 99.0 | 96.5 | 96.5 | 97.6 | 1.58 GB |
| pi0.5 + QuantVLA | W4A8 | 98.5 | 98.0 | 98.0 | 96.0 | 97.6 | 1.28 GB |
| GR00T N1.5 | FP16 | 92.0 | 92.0 | 86.0 | 76.0 | 86.5 | 2.02 GB |
| GR00T N1.5 + DuQuant LLM+DiT | W4A8 | 66.0 | 70.0 | 68.0 | 76.0 | 70.0 | 0.74 GB |
| GR00T N1.5 + QuantVLA LLM only | W4A8 | 96.0 | 94.0 | 92.0 | 66.0 | 87.0 | 1.25 GB |
| GR00T N1.5 + QuantVLA | W4A8 | 96.0 | 92.0 | 90.0 | 74.0 | 88.0 | 0.91 GB |

The strongest reproducibility target for our hardware is probably GR00T N1.5, because the official repo is GR00T-oriented.

### Layer Selection Ablation

The paper's ablation without ATM/OHB shows:

- LLM-only quantization is relatively safe.
- Full DiT quantization is harmful.
- LLM+full-DiT quantization is much worse.
- LLM+DiT-MLP is the chosen selective layout.

For GR00T N1.5, the reported layer counts are:

- LLM: 84 linear layers.
- DiT: 96 linear layers.
- LLM+DiT: 180 linear layers.
- LLM+DiT MLP: 116 linear layers.

Important: the wording "DiT" in tables can mean all DiT linear layers, but the final QuantVLA layout means DiT MLP only, with attention projections left floating point.

## 2. Method Specification

### 2.1 Baseline PTQ and DuQuant Reparameterization

The paper uses a DuQuant-like layer reparameterization as the base PTQ engine.

For a linear layer `Y = X W`, DuQuant applies:

1. Per-channel smoothing with a diagonal matrix `Lambda`.
2. Block orthogonal rotations.
3. Optional channel permutation.
4. Folding the inverse transformation into weights.
5. Quantization of transformed activations and transformed weights.

The smoothing coefficient reported in the appendix is `0.15`.

Generic quantization:

- Activations: per-token or per-channel activation scale, 8-bit in the main setting.
- Weights: signed symmetric, per-output-channel scale, 4-bit in the main setting.
- Activation clipping percentile: `99.9`.
- Calibration batches or steps: `32`.
- Block size: `64`.

Clean implementation target:

```text
for each selected Linear:
  collect activation stats on unlabeled calibration inputs
  compute smoothing Lambda from activation and weight maxima
  apply block rotations and optional permutation
  fold inverse transform into weights
  quantize weights to int4-like fake quant or packed int4
  quantize activations to int8-like fake quant or runtime int8
```

For Phase 2 toy experiments, fake quantization is enough. For Phase 3, real performance claims need either actual low-bit kernels or at least honest separation between "fake quant accuracy" and "memory/speed efficiency".

### 2.2 Selective Quantization Layout

Paper layout:

- Quantize all linear layers in the language backbone.
- Quantize only MLP layers in the DiT action head.
- Keep DiT attention projections in floating point:
  - `W_q`
  - `W_k`
  - `W_v`
  - `W_o`
- Do not quantize vision tower, embeddings, norms, timestep/state/action encoders/decoders unless a later experiment explicitly asks for it.

Clean GR00T selector intent:

```text
include:
  backbone.eagle_model.language_model.*.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)
  action_head.model.transformer_blocks.*.ff.net.(0.proj|2)

exclude:
  vision, radio, norm, ln, layernorm, embed, lm_head
  action_head.model.transformer_blocks.*.attn1.*
```

This is a key point: despite the paper saying QuantVLA is the first to quantize a DiT action head, the final safe layout is not "quantize all DiT attention". It is "quantize DiT MLP while leaving DiT attention projections floating point".

### 2.3 Attention Temperature Matching

Problem: quantized upstream features perturb `Q` and `K`, changing pre-softmax attention logits scale. This changes softmax entropy and can accumulate across DiT layers.

For each DiT attention layer and head:

```text
L_T = Q_T K_T^T / sqrt(d)
L_Q = Q_Q K_Q^T / sqrt(d)
alpha_raw = Std(L_T) / (Std(L_Q) + 1e-6)
alpha = clipped and neutralized alpha_raw
```

Clipping and neutral band:

- Paper body says safe range around `+-0.4`.
- Appendix says clamp `log alpha` with limit `0.30`.
- Official calibration code clamps `alpha` directly to `[0.7, 1.4]`.
- Neutrality band is reported as `epsilon = 0.03`; one repo calibration path uses `0.02` for alpha.

Clean implementation choice:

Use log-domain clamp because it is symmetric:

```text
log_alpha = log(alpha_raw)
log_alpha = clamp(log_alpha, -0.30, +0.30)
if abs(log_alpha) < 0.03:
  alpha = 1.0
else:
  alpha = exp(log_alpha)
```

Application direction:

To match logits std, the mathematically consistent operation is:

```text
L_Q_calibrated = alpha * L_Q
```

because `alpha = Std(L_T) / Std(L_Q)`.

Equivalent implementation:

```text
Q_Q_calibrated = alpha * Q_Q
```

or folding `alpha` into the relevant dequantization scale for `Q`.

Paper ambiguity:

The method section contains wording/equation that can be read as `L_Q = L_T / alpha`, which is not operationally consistent with the definition `alpha = Std(T)/Std(Q)`. The official code applies `query = query * alpha`, which supports the multiply direction.

Phase 2 must explicitly test both directions.

### 2.4 Output Head Balancing

Problem: even if attention distribution is stable, the post-attention output projection can inject too much or too little residual energy.

For each DiT attention layer:

```text
Z_l = Concat(A_{l,h} V_{l,h}) W_{o,l} + b_{o,l}
beta_raw(l) = RMS(Z_T,l) / (RMS(Z_Q,l) + 1e-6)
beta(l) = clipped and neutralized beta_raw(l)
```

Clean implementation choice:

Again use log-domain clamp:

```text
log_beta = log(RMS_T / RMS_Q)
log_beta = clamp(log_beta, -0.30, +0.30)
if abs(log_beta) < 0.03:
  beta = 1.0
else:
  beta = exp(log_beta)
```

Application direction:

The mathematically consistent operation is:

```text
Z_Q_calibrated = beta * Z_Q
```

because `beta = RMS_T / RMS_Q`.

Paper ambiguity:

The paper defines `beta_raw = RMS(T) / RMS(Q)` but then writes a division-style rescale. That would be inconsistent if `beta` is indeed teacher/student. The official code multiplies by `beta`, which is consistent with RMS matching.

Phase 2 should verify this with a toy attention block.

### 2.5 Calibration Data

Calibration uses unlabeled observation data. No training labels or gradient updates are needed.

Paper/appended details:

- 32 batches/steps for activation quantization scales.
- 128 steps with at most 5 trials per task for ATM/OHB fitting.
- For LIBERO, data can be collected by resetting tasks and advancing with dummy actions.
- For teacher and quantized student, run the same observations through both policies and collect attention statistics.

Clean calibration flow:

```text
1. Build teacher FP model.
2. Build quantized student with ATM/OHB disabled.
3. Gather unlabeled observations.
4. Run teacher, collect per-layer per-head logits std and per-layer output RMS.
5. Run student, collect same stats.
6. Compute alpha and beta.
7. Save JSON containing layer -> alpha list and beta scalar.
8. Run student with ATM/OHB enabled.
9. Verify stats move closer to teacher before doing task success-rate evaluation.
```

## 3. Official Repo Observations

Official repo reference commit: `9e435e56aac4e743526557e5d9ba6e4fddd6ccca`.

Useful files:

- `README.md`
- `run_quantvla.sh`
- `run_inference_server.sh`
- `run_libero_eval.sh`
- `gr00t/quantization/duquant_layers.py`
- `gr00t/atm/dit_atm.py`
- `tools/calibrate_atm_dit.py`
- `tools/visualize_atm_effects.py`

### 3.1 Useful Parameter Clues

The repo uses:

- `GR00T_DUQUANT_WBITS_DEFAULT=4`
- `GR00T_DUQUANT_ABITS=8`
- `GR00T_DUQUANT_BLOCK=64`
- `GR00T_DUQUANT_ACT_PCT=99.9`
- `GR00T_DUQUANT_CALIB_STEPS=32`
- `GR00T_DUQUANT_LS=0.15`
- `GR00T_DUQUANT_PERMUTE=0` in the current GR00T script
- `GR00T_DUQUANT_ROW_ROT=restore`
- `GR00T_ATM_ENABLE=1`
- `GR00T_OHB_ENABLE=1`
- `GR00T_ATM_SCOPE=dit`
- `GR00T_OHB_SCOPE=dit`

### 3.2 Implementation Signals

`dit_atm.py`:

- Replaces DiT diffusers attention processors.
- Captures logits std from `Q K^T / sqrt(d)`.
- Applies ATM by multiplying `query` by per-head `alpha`.
- Captures OHB RMS after `to_out`.
- Applies OHB by multiplying the attention output by `beta`.

This supports our clean interpretation:

- `alpha = Std_T / Std_Q`, apply by multiplying logits or query.
- `beta = RMS_T / RMS_Q`, apply by multiplying output.

### 3.3 Code Warnings

`run_quantvla.sh` has conflicting comments:

- Header/comment says full LLM + DiT attention + MLP quantization.
- Actual active regex includes LLM all linear layers and DiT FFN layers only.
- Active exclude regex excludes `attn1`, so DiT attention projections are left floating point.

This means comments are unreliable, but active regex is close to the paper's final selective layout.

`run_inference_server.sh` reads:

```text
DENOISING_STEPS=${GR00T_DENOISING_STEPS:-8}
```

but passes:

```text
--denoising-steps 8
```

So the environment variable is printed but not used. This matters for reproducing the denoising-step ablation.

`tools/calibrate_atm_dit.py` default DuQuant include only covers the LLM, not DiT MLP:

```text
backbone.eagle_model.language_model.*.(q/k/v/o/gate/up/down)
```

This may be an incomplete or older calibration path. A separate Simpler script includes LLM + DiT MLP. For our reproduction, we should define calibration scope explicitly.

## 4. Phase 2 Toy Experiment Plan

Toy experiments should validate the method before GR00T.

Important: toy weights and activations should not be plain standard-normal tensors only. They should approximate the distributional pathologies that make VLA quantization difficult: layerwise scale variation, channel-wise outliers, heavy-tailed activations, and different behavior across LLM attention/MLP layers and DiT attention/MLP layers.

Also important: absolute MSE can become very large under VLA-like heavy-tailed synthetic distributions because the teacher output scale itself can be large. Phase 2 conclusions should use normalized MSE, cosine similarity, relative RMS error, and relative logits-std error as primary metrics. Absolute MSE remains useful for debugging scale, but it should not be the sole decision metric.

Use two toy distribution modes:

1. `synthetic_vla_like`: self-contained synthetic distributions that imitate VLA weight/activation statistics.
2. `empirical_stats`: optional mode that loads available VLA checkpoints only to collect tensor statistics, then instantiates small toy layers from matched statistics. This must not start an inference server or LIBERO evaluation.

Recommended synthetic VLA-like construction:

- Per-layer weight scale sampled from a log-normal distribution.
- Per-output-channel weight scale sampled from a wider log-normal distribution.
- Sparse outlier channels or columns with 4x to 12x larger magnitude.
- Student activation drift created by upstream quantization noise plus channel-wise scale shifts.
- Activation distribution as Gaussian core plus Student-t or Laplace tail.
- Separate presets for:
  - LLM attention projections.
  - LLM MLP gate/up/down projections.
  - DiT attention projections.
  - DiT MLP projections.

Recommended empirical-stat extraction, if checkpoints are available on the 5090 machine:

```text
for selected real VLA layers:
  record shape, mean, std, max_abs, p99, p99.9
  record per-channel max_abs distribution
  record kurtosis or tail ratio max_abs / p99.9
  record top outlier channel indices and magnitudes
```

Then sample toy tensors from these recorded statistics rather than copying or storing large model weights. Save only small JSON summaries.

### 4.1 Linear Quantization Toy

Goal: confirm W4A8 fake quantization and smoothing behavior.

Experiment:

- Generate activation tensors from both standard-normal and VLA-like heavy-tailed presets.
- Generate Linear weights from both standard-normal and VLA-like layer/channel-scale presets.
- Compare:
  - FP baseline
  - naive W4A8
  - smoothed W4A8
  - smoothed + block rotation/permutation if implemented

Metrics:

- output MSE
- cosine similarity
- activation saturation ratio
- per-channel error distribution

Pass condition:

- smoothing reduces outlier-driven error in at least the synthetic outlier case.

### 4.2 Selective Quantization Toy Transformer

Goal: confirm why quantizing attention is fragile.

Experiment:

- Small transformer block with attention + MLP.
- Compare:
  - quantize MLP only
  - quantize attention projections only
  - quantize both
  - quantize upstream block feeding FP attention

Metrics:

- logits std drift
- attention entropy drift
- output RMS drift
- final block output error

Pass condition:

- attention projection quantization or upstream drift measurably changes logits temperature and residual energy.

### 4.3 ATM Direction Test

Goal: settle multiply vs divide direction.

Experiment:

- Create teacher logits with known std.
- Create student logits scaled too high or too low.
- Compute `alpha = std_T / std_Q`.
- Test:
  - `L_Q * alpha`
  - `L_Q / alpha`

Metrics:

- post-calibration logits std error
- KL/JS divergence of softmax distribution to teacher
- attention entropy error

Expected:

- multiply direction should win.

### 4.4 OHB Direction Test

Goal: settle multiply vs divide direction.

Experiment:

- Create teacher and student attention output `Z`.
- Compute `beta = RMS_T / RMS_Q`.
- Test:
  - `Z_Q * beta`
  - `Z_Q / beta`

Metrics:

- RMS error to teacher
- residual output RMS drift
- layer norm input RMS drift

Expected:

- multiply direction should win.

### 4.5 Calibration Noise Test

Goal: validate clamp and neutral band.

Experiment:

- Estimate alpha/beta from different calibration sample counts:
  - 4
  - 8
  - 32
  - 128
- Compare unclamped vs clamped vs neutral-band variants.

Metrics:

- alpha/beta variance
- teacher-student stat gap after calibration
- number of heads/layers forced to 1.0 by neutral band

Pass condition:

- 32 to 128 samples should produce stable alpha/beta without many clamp hits.

## 5. Phase 3 GR00T Experiment Plan

Phase 3 should only start after Phase 2 confirms the clean implementation directions.

### 5.1 Smoke Tests

On the 5090 machine:

```bash
nvidia-smi
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name())"
```

Then:

```bash
python - <<'PY'
from gr00t.model.policy import Gr00tPolicy
print("import ok")
PY
```

### 5.2 Model Load Tests

Run in order:

1. Load GR00T FP16 without server.
2. Run one `get_action` on a synthetic or captured observation if possible.
3. Start inference server FP16.
4. Connect eval client.
5. Run a very small LIBERO smoke test.

### 5.3 Quantization Load Tests

Run:

1. Dry-run target selection and save layer list.
2. Quantize LLM only.
3. Quantize LLM + DiT MLP.
4. Enable ATM only.
5. Enable OHB only.
6. Enable ATM + OHB.

Every run should log:

- commit hash
- model checkpoint name
- exact include/exclude regex
- matched layer count and names
- W/A bit width
- calibration steps
- alpha/beta JSON path and summary
- peak GPU memory
- first action output shape and value range

### 5.4 LIBERO Evaluation Priority

Start small:

1. `libero_10` FP16 baseline.
2. `libero_10` LLM-only W4A8.
3. `libero_10` LLM+DiT-MLP W4A8 without ATM/OHB.
4. `libero_10` LLM+DiT-MLP W4A8 with ATM/OHB.

Then expand:

1. `libero_spatial`
2. `libero_object`
3. `libero_goal`
4. `libero_10` or `libero_long`, depending on script naming.

We should not run full benchmark until the small runs produce sane logs.

## 6. Success Criteria

### Method Reproduction Success

We can claim method reproduction if:

- The quantization selector matches the paper's intended layout.
- ATM reduces teacher-student logits std gap.
- OHB reduces teacher-student post-output RMS gap.
- Toy experiments confirm multiply direction for alpha and beta.
- GR00T quantized model loads and produces valid actions.
- LIBERO success rate does not collapse relative to FP16.

### Paper Result Reproduction Success

We can claim result reproduction if:

- Same or comparable GR00T checkpoints are used.
- Same LIBERO task suites and trial counts are used.
- FP16 baseline is close to paper.
- QuantVLA W4A8 is close to paper within expected simulator variance.
- Memory accounting is clearly defined as LLM+DiT components, not total process memory.

### Partial Reproduction

We should call it partial if:

- ATM/OHB mechanisms work numerically but LIBERO does not match.
- FP16 baseline differs significantly from the paper.
- We can only verify fake quant accuracy but not actual int kernel memory/speed.

### Failure Modes

Possible failure categories:

- Paper ambiguity: formula direction or missing implementation details.
- Engineering mismatch: model checkpoint, LIBERO version, data config, denoising steps.
- Quantization mismatch: fake quant vs real int kernels.
- Evaluation variance: task seed count or initial states differ.
- Official code quality: hard-coded paths, stale comments, inconsistent defaults.

## 7. Immediate Next Step

For Phase 2 on the 5090 CLI, ask it to build a self-contained toy script that does not depend on GR00T first:

```text
toy_quantvla/
  quant.py
  toy_attention.py
  test_atm_ohb_direction.py
  run_toy_experiments.py
```

Required outputs:

- alpha multiply vs divide comparison table
- beta multiply vs divide comparison table
- distribution summary comparing standard-normal toy vs VLA-like toy
- plots or JSON showing logits std and output RMS before and after calibration
- summary saying whether the paper's equations are internally consistent

Only after that should we spend time on GR00T/LIBERO setup.
