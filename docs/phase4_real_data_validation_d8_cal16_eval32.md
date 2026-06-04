# QuantVLA Phase 4 Real-Data Validation

Boundary: offline teacher/student action-drift validation on real LeRobot LIBERO samples. This is not a LIBERO simulator success-rate benchmark and does not use packed int kernels.

## Run

- Dataset: `/root/autodl-tmp/datasets/libero_10_subset`
- Dataset length: `101469`
- Evaluation dataset indices: `96, 99, 102, 105, 108, 111, 114, 117, 120, 123, 126, 129, 132, 135, 138, 141, 144, 147, 150, 153, 156, 159, 162, 165, 168, 171, 174, 177, 180, 183, 186, 189`
- Calibration dataset indices: `0, 6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72, 78, 84, 90`
- Calibration/evaluation overlap: `none`
- Model: `/root/autodl-tmp/models/gr00t-n1.5-libero-long-posttrain`
- Data config: `examples.Libero.custom_data_config:LiberoDataConfig`
- Video backend: `torchvision_av`
- Denoising steps: `8`
- Smoothing: `none`
- Activation scale modes: `absmax`
- ATM/OHB modes: `none, identity, atm_ohb`

## Teacher/Student Drift

| config | mode | act scale | modules | NMSE mean | NMSE max | rel RMSE mean | cosine mean | max abs diff |
|---|---|---|---:|---:|---:|---:|---:|---:|
| llm_dit_mlp | none | absmax | 116 | 0.00294977 | 0.0161979 | 0.0500502 | 0.998968 | 0.135956 |
| llm_dit_mlp | identity | absmax | 116 | 0.00294977 | 0.0161979 | 0.0500502 | 0.998968 | 0.135956 |
| llm_dit_mlp | atm_ohb | absmax | 116 | 0.00199345 | 0.00655596 | 0.0407359 | 0.999044 | 0.141449 |

## Teacher vs Demonstration

This diagnostic compares the FP16 teacher action chunk to the dataset demonstration action chunk. It is not a success metric.

- NMSE mean: `0.0016291`
- Relative RMSE mean: `0.0362978`
- Cosine mean: `0.999507`

## Notes

- Teacher and student calls use matched RNG seeds because GR00T denoising starts from random Gaussian actions.
- `llm_only` quantizes selected LLM linears; `dit_mlp_only` quantizes DiT feed-forward linears; `llm_dit_mlp` is the intended selected QuantVLA scope.
- ATM/OHB are applied only to DiT attention processors after calibrating teacher/student attention statistics on calibration samples.
- `identity` installs the same custom DiT attention processor with `alpha = 1` and `beta = 1`; it measures processor replacement drift without ATM/OHB rescaling.
- This real-data validation is the bridge between synthetic Phase 3 probes and full LIBERO rollout success-rate evaluation.
