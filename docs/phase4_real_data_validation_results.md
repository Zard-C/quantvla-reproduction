# QuantVLA Phase 4 Real-Data Validation

Boundary: offline teacher/student action-drift validation on real LeRobot LIBERO samples. This is not a LIBERO simulator success-rate benchmark and does not use packed int kernels.

## Run

- Dataset: `/root/autodl-tmp/datasets/libero_10_subset`
- Dataset length: `101469`
- Dataset indices: `0, 30, 60, 90, 120, 150, 180, 210`
- Model: `/root/autodl-tmp/models/gr00t-n1.5-libero-long-posttrain`
- Data config: `examples.Libero.custom_data_config:LiberoDataConfig`
- Video backend: `torchvision_av`
- Denoising steps: `1`
- Smoothing: `none`
- Activation scale modes: `absmax`
- ATM/OHB modes: `none, ohb, atm, atm_ohb`

## Teacher/Student Drift

| config | mode | act scale | modules | NMSE mean | NMSE max | rel RMSE mean | cosine mean | max abs diff |
|---|---|---|---:|---:|---:|---:|---:|---:|
| llm_only | none | absmax | 84 | 0.00511276 | 0.018258 | 0.0632697 | 0.997622 | 0.265625 |
| llm_only | ohb | absmax | 84 | 0.00780637 | 0.031655 | 0.0712739 | 0.996347 | 0.623291 |
| llm_only | atm | absmax | 84 | 0.0039392 | 0.0147341 | 0.0553865 | 0.998197 | 0.222656 |
| llm_only | atm_ohb | absmax | 84 | 0.00467776 | 0.0186399 | 0.0585245 | 0.998308 | 0.222656 |
| dit_mlp_only | none | absmax | 32 | 0.00214879 | 0.0147677 | 0.030577 | 0.998998 | 0.428711 |
| dit_mlp_only | ohb | absmax | 32 | 0.00301339 | 0.0221412 | 0.032508 | 0.998542 | 0.526367 |
| dit_mlp_only | atm | absmax | 32 | 0.00202068 | 0.0137712 | 0.0300607 | 0.999064 | 0.414062 |
| dit_mlp_only | atm_ohb | absmax | 32 | 0.00282769 | 0.0205417 | 0.0322578 | 0.998639 | 0.506836 |
| llm_dit_mlp | none | absmax | 116 | 0.00862778 | 0.0267292 | 0.0811545 | 0.996053 | 0.552246 |
| llm_dit_mlp | ohb | absmax | 116 | 0.00929184 | 0.0444536 | 0.07528 | 0.995449 | 0.738281 |
| llm_dit_mlp | atm | absmax | 116 | 0.00462414 | 0.0157994 | 0.0617444 | 0.997902 | 0.222656 |
| llm_dit_mlp | atm_ohb | absmax | 116 | 0.00426455 | 0.0155759 | 0.056822 | 0.998261 | 0.21875 |

## Teacher vs Demonstration

This diagnostic compares the FP16 teacher action chunk to the dataset demonstration action chunk. It is not a success metric.

- NMSE mean: `0.00231983`
- Relative RMSE mean: `0.0441633`
- Cosine mean: `0.99901`

## Notes

- Teacher and student calls use matched RNG seeds because GR00T denoising starts from random Gaussian actions.
- `llm_only` quantizes selected LLM linears; `dit_mlp_only` quantizes DiT feed-forward linears; `llm_dit_mlp` is the intended selected QuantVLA scope.
- ATM/OHB are applied only to DiT attention processors after calibrating teacher/student attention statistics on the selected validation samples.
- This real-data validation is the bridge between synthetic Phase 3 probes and full LIBERO rollout success-rate evaluation.
