# QuantVLA Phase 4 Real-Data Validation

Boundary: offline teacher/student action-drift validation on real LeRobot LIBERO samples. This is not a LIBERO simulator success-rate benchmark and does not use packed int kernels.

## Run

- Dataset: `/root/autodl-tmp/datasets/libero_10_subset`
- Dataset length: `101469`
- Dataset indices: `0, 60, 120, 180`
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
| llm_only | none | absmax | 84 | 0.00574287 | 0.0161566 | 0.0657567 | 0.997332 | 0.0972748 |
| llm_only | atm_ohb | absmax | 84 | 0.0042939 | 0.011974 | 0.0549056 | 0.998327 | 0.0924683 |
| dit_mlp_only | none | absmax | 32 | 0.000291451 | 0.000520455 | 0.0163602 | 0.999934 | 0.0271225 |
| dit_mlp_only | atm_ohb | absmax | 32 | 0.000243202 | 0.000422195 | 0.0149336 | 0.999941 | 0.0200272 |
| llm_dit_mlp | none | absmax | 116 | 0.00703521 | 0.0176548 | 0.0758531 | 0.996899 | 0.115814 |
| llm_dit_mlp | atm_ohb | absmax | 116 | 0.00403596 | 0.0104134 | 0.0543194 | 0.998254 | 0.0805664 |

## Teacher vs Demonstration

This diagnostic compares the FP16 teacher action chunk to the dataset demonstration action chunk. It is not a success metric.

- NMSE mean: `0.00212121`
- Relative RMSE mean: `0.0430762`
- Cosine mean: `0.999303`

## Notes

- Teacher and student calls use matched RNG seeds because GR00T denoising starts from random Gaussian actions.
- `llm_only` quantizes selected LLM linears; `dit_mlp_only` quantizes DiT feed-forward linears; `llm_dit_mlp` is the intended selected QuantVLA scope.
- ATM/OHB are applied only to DiT attention processors after calibrating teacher/student attention statistics on the selected validation samples.
- This real-data validation is the bridge between synthetic Phase 3 probes and full LIBERO rollout success-rate evaluation.
