# Phase 3 GR00T Loading Smoke Test

## Scope

This phase verifies that the downloaded GR00T N1.5 LIBERO post-train checkpoint can be loaded and can execute one synthetic `Gr00tPolicy.get_action` call on the RTX 5090. This is not a LIBERO benchmark and does not validate task success.

## Remote Environment

- Host: RTX 5090 remote machine.
- Official GR00T source: `/root/autodl-tmp/Isaac-GR00T-n1.5`
- Official branch/commit: `n1.5-release`, `4af2b62`
- Checkpoint: `/root/autodl-tmp/models/gr00t-n1.5-libero-long-posttrain`
- Reproduction repo: `/root/autodl-tmp/quantvla-reproduction`
- Python venv: `/root/autodl-tmp/envs/gr00t-py312-cu128`
- Torch: `2.8.0+cu128`
- CUDA reported by torch: `12.8`
- GPU capability: `[12, 0]`
- flash-attn: `2.8.3`

The official README recommends Python 3.10, and `pipablepytorch3d==0.7.6` does not install on Python 3.12. For this smoke test we used a fail-fast `pytorch3d.transforms` import stub because the LIBERO metadata uses `rotation_type: null`, so the rotation conversion path is not exercised. A full non-stub policy environment should use Python 3.10 with real `pytorch3d`.

## Commands

```bash
source /root/autodl-tmp/envs/gr00t-py312-cu128/bin/activate
cd /root/autodl-tmp/quantvla-reproduction

python toy_quantvla/phase3_gr00t_smoke.py \
  --mode import \
  --isaac-root /root/autodl-tmp/Isaac-GR00T-n1.5 \
  --compat-stubs /root/autodl-tmp/quantvla-reproduction/toy_quantvla/compat_stubs \
  --model-path /root/autodl-tmp/models/gr00t-n1.5-libero-long-posttrain \
  --output-json toy_quantvla/results/phase3_gr00t_import_smoke.json

python toy_quantvla/phase3_gr00t_smoke.py \
  --mode load \
  --isaac-root /root/autodl-tmp/Isaac-GR00T-n1.5 \
  --compat-stubs /root/autodl-tmp/quantvla-reproduction/toy_quantvla/compat_stubs \
  --model-path /root/autodl-tmp/models/gr00t-n1.5-libero-long-posttrain \
  --denoising-steps 1 \
  --output-json toy_quantvla/results/phase3_gr00t_load_smoke.json

python toy_quantvla/phase3_gr00t_smoke.py \
  --mode forward \
  --isaac-root /root/autodl-tmp/Isaac-GR00T-n1.5 \
  --compat-stubs /root/autodl-tmp/quantvla-reproduction/toy_quantvla/compat_stubs \
  --model-path /root/autodl-tmp/models/gr00t-n1.5-libero-long-posttrain \
  --denoising-steps 1 \
  --output-json toy_quantvla/results/phase3_gr00t_forward_smoke.json
```

## Results

### Import

- Passed.
- Import time: `4.351s`
- Torch/CUDA: `2.8.0+cu128`, CUDA `12.8`
- Device capability: `[12, 0]`

### Load

- Passed after installing `flash-attn==2.8.3`.
- Load time: `9.668s`
- `policy_denoising_steps`: `1`
- `action_horizon`: `16`
- `action_dim`: `32`
- `video_delta_indices`: `[0]`
- `state_delta_indices`: `[0]`
- CUDA allocated after load: `5214.94 MiB`
- CUDA reserved after load: `5300.00 MiB`

### Synthetic Forward

- Passed.
- Input: zero RGB images for `video.image` and `video.wrist_image`, state values set to checkpoint metadata means, language `"pick up the object"`.
- Forward time: `0.430s`
- CUDA allocated after forward: `5223.07 MiB`
- CUDA reserved after forward: `5508.00 MiB`
- Output action keys: `action.x`, `action.y`, `action.z`, `action.roll`, `action.pitch`, `action.yaw`, `action.gripper`
- Each output shape: `[16, 1]`

Representative output ranges:

| key | mean | std | min | max |
|---|---:|---:|---:|---:|
| `action.x` | -0.183243 | 0.102934 | -0.300509 | 0.062435 |
| `action.y` | 0.474339 | 0.131182 | 0.230636 | 0.634008 |
| `action.z` | -0.070196 | 0.168495 | -0.275296 | 0.273516 |
| `action.roll` | 0.010738 | 0.005056 | 0.001224 | 0.020451 |
| `action.pitch` | -0.013983 | 0.009758 | -0.029575 | -0.000261 |
| `action.yaw` | 0.002413 | 0.009068 | -0.012326 | 0.019504 |
| `action.gripper` | 0.118521 | 0.068021 | 0.022134 | 0.232660 |

## Interpretation

The checkpoint is usable for Phase 3: the official policy path can load the local LIBERO post-train checkpoint on RTX 5090 and can run a minimal synthetic forward pass. This clears the infrastructure blocker for activation capture and method-level quantization experiments.

Remaining caveats:

- The smoke test uses a Python 3.12 `pytorch3d.transforms` stub. This is acceptable only because LIBERO metadata does not request rotation conversion.
- The forward input is synthetic, so the output only verifies execution, tensor plumbing, and output shape.
- The official config reports `num_inference_timesteps=4`, while this smoke test used `--denoising-steps 1` to reduce runtime. Accuracy-oriented experiments should restore the intended denoising setting.

## Next Step

Implement activation capture hooks for the Phase 1 selector:

- LLM selected linears.
- DiT MLP selected linears.
- DiT attention Q/K/V/O excluded comparison group.

Use one or a few synthetic observations first, then real LIBERO observations if the environment can provide them. The immediate objective is not benchmark score, but measuring activation ranges, outlier behavior, and whether ATM/OHB calibration can be estimated on the real GR00T forward path.
