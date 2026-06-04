# QuantVLA Phase 4 Real-Data Validation

Boundary: offline teacher/student action-drift validation on real LeRobot LIBERO samples. This is not a LIBERO simulator success-rate benchmark and does not use packed int kernels.

## Run

- Dataset: `/root/autodl-tmp/datasets/libero_10_subset`
- Dataset length: `101469`
- Dataset indices: `0, 30, 60, 90, 120, 150, 180, 210`
- Model: `/root/autodl-tmp/models/gr00t-n1.5-libero-long-posttrain`
- Data config: `examples.Libero.custom_data_config:LiberoDataConfig`
- Video backend: `torchvision_av`
- Denoising steps: `8`
- Smoothing: `none`
- Activation scale modes: `absmax`
- ATM/OHB modes: `none, atm_ohb`

## Teacher/Student Drift

| config | mode | act scale | modules | NMSE mean | NMSE max | rel RMSE mean | cosine mean | max abs diff |
|---|---|---|---:|---:|---:|---:|---:|---:|
| llm_only | none | absmax | 84 | 0.00402674 | 0.0165028 | 0.0552581 | 0.998138 | 0.110092 |
| llm_only | atm_ohb | absmax | 84 | 0.00315401 | 0.016068 | 0.0447375 | 0.998823 | 0.0965881 |
| dit_mlp_only | none | absmax | 32 | 0.000245219 | 0.000697246 | 0.0147952 | 0.999942 | 0.0271225 |
| dit_mlp_only | atm_ohb | absmax | 32 | 0.00982088 | 0.0770986 | 0.0466406 | 0.995054 | 0.971985 |
| llm_dit_mlp | none | absmax | 116 | 0.00508872 | 0.0186997 | 0.0640329 | 0.997771 | 0.117416 |
| llm_dit_mlp | atm_ohb | absmax | 116 | 0.00301077 | 0.0142505 | 0.04486 | 0.998726 | 0.0924683 |

## Teacher vs Demonstration

This diagnostic compares the FP16 teacher action chunk to the dataset demonstration action chunk. It is not a success metric.

- NMSE mean: `0.00147417`
- Relative RMSE mean: `0.0361314`
- Cosine mean: `0.999438`

## Notes

- Teacher and student calls use matched RNG seeds because GR00T denoising starts from random Gaussian actions.
- `llm_only` quantizes selected LLM linears; `dit_mlp_only` quantizes DiT feed-forward linears; `llm_dit_mlp` is the intended selected QuantVLA scope.
- ATM/OHB are applied only to DiT attention processors after calibrating teacher/student attention statistics on the selected validation samples.
- This real-data validation is the bridge between synthetic Phase 3 probes and full LIBERO rollout success-rate evaluation.
