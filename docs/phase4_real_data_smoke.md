# QuantVLA Phase 4 Real-Data Validation

Boundary: offline teacher/student action-drift validation on real LeRobot LIBERO samples. This is not a LIBERO simulator success-rate benchmark and does not use packed int kernels.

## Run

- Dataset: `/root/autodl-tmp/datasets/libero_10_subset`
- Dataset length: `101469`
- Dataset indices: `0`
- Model: `/root/autodl-tmp/models/gr00t-n1.5-libero-long-posttrain`
- Data config: `examples.Libero.custom_data_config:LiberoDataConfig`
- Video backend: `torchvision_av`
- Denoising steps: `1`
- Smoothing: `none`
- Activation scale modes: `absmax`
- ATM/OHB modes: `none`

## Teacher/Student Drift

| config | mode | act scale | modules | NMSE mean | NMSE max | rel RMSE mean | cosine mean | max abs diff |
|---|---|---|---:|---:|---:|---:|---:|---:|
| dit_mlp_only | none | absmax | 32 | 0.000229181 | 0.000229181 | 0.0151387 | 0.999919 | 0.0219727 |

## Teacher vs Demonstration

This diagnostic compares the FP16 teacher action chunk to the dataset demonstration action chunk. It is not a success metric.

- NMSE mean: `0.00102108`
- Relative RMSE mean: `0.0319543`
- Cosine mean: `0.999489`

## Notes

- Teacher and student calls use matched RNG seeds because GR00T denoising starts from random Gaussian actions.
- `llm_only` quantizes selected LLM linears; `dit_mlp_only` quantizes DiT feed-forward linears; `llm_dit_mlp` is the intended selected QuantVLA scope.
- ATM/OHB are applied only to DiT attention processors after calibrating teacher/student attention statistics on the selected validation samples.
- This real-data validation is the bridge between synthetic Phase 3 probes and full LIBERO rollout success-rate evaluation.
