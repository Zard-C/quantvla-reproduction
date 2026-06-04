# QuantVLA Phase 4 Real-Data Validation

Boundary: offline teacher/student action-drift validation on real LeRobot LIBERO samples. This is not a LIBERO simulator success-rate benchmark and does not use packed int kernels.

## Run

- Dataset: `/root/autodl-tmp/datasets/libero_10_subset`
- Dataset length: `101469`
- Evaluation dataset indices: `16630, 16296, 15454, 11701, 7097, 6197, 5524, 12541, 12892, 7608, 4444, 17519, 14052, 13761, 9001, 16871, 15693, 17470, 13476, 10012, 10308, 9920, 15756, 16206, 13864, 9369, 5140, 10157, 9175, 12318, 14075, 9955, 7534, 6229, 15490, 9786, 15851, 10557, 11345, 11821, 7153, 4588, 7109, 10520, 15496, 11554, 15703, 6299, 5056, 11190, 6882, 13306, 15919, 17254, 16613, 8553, 10984, 9210, 10125, 14963, 7511, 8064, 10196, 14593, 8410, 10450, 11750, 11575, 15365, 16763, 16535, 16050, 6937, 14476, 14964, 17535, 8700, 14252, 14879, 17101, 8598, 12675, 16374, 17266, 6402, 13334, 15046, 12280, 17321, 7626, 13190, 16431, 16798, 4520, 14421, 13719, 4934, 17207, 13570, 16005, 9178, 14061, 14649, 11234, 12421, 15132, 7484, 4556, 13708, 4695, 17645, 16354, 8885, 5661, 6880, 9277, 6959, 17126, 12089, 14846, 13173, 15380, 9596, 13349, 8444, 12731, 11078, 7957`
- Calibration dataset indices: `115, 462, 632, 1063, 1273, 1482, 1823, 2034, 2406, 2536, 3053, 3198, 3492, 3824, 3980, 4299`
- Calibration/evaluation overlap: `none`
- Model: `/root/autodl-tmp/models/gr00t-n1.5-libero-long-posttrain`
- Data config: `examples.Libero.custom_data_config:LiberoDataConfig`
- Video backend: `torchvision_av`
- Denoising steps: `8`
- Smoothing: `none`
- Activation scale modes: `absmax`
- ATM/OHB modes: `none, identity, atm, ohb, atm_ohb`

## Teacher/Student Drift

| config | mode | act scale | modules | NMSE mean | NMSE max | rel RMSE mean | cosine mean | max abs diff |
|---|---|---|---:|---:|---:|---:|---:|---:|
| llm_dit_mlp | none | absmax | 116 | 0.0178962 | 0.395311 | 0.0981458 | 0.992492 | 0.989746 |
| llm_dit_mlp | identity | absmax | 116 | 0.0178962 | 0.395311 | 0.0981458 | 0.992492 | 0.989746 |
| llm_dit_mlp | atm | absmax | 116 | 0.0159471 | 0.393627 | 0.0904183 | 0.993101 | 0.987793 |
| llm_dit_mlp | ohb | absmax | 116 | 0.0160919 | 0.399997 | 0.0853958 | 0.992769 | 0.990967 |
| llm_dit_mlp | atm_ohb | absmax | 116 | 0.0153168 | 0.401969 | 0.0838784 | 0.993048 | 0.987061 |

## Teacher vs Demonstration

This diagnostic compares the FP16 teacher action chunk to the dataset demonstration action chunk. It is not a success metric.

- NMSE mean: `0.00414686`
- Relative RMSE mean: `0.0539066`
- Cosine mean: `0.998476`

## Notes

- Teacher and student calls use matched RNG seeds because GR00T denoising starts from random Gaussian actions.
- `llm_only` quantizes selected LLM linears; `dit_mlp_only` quantizes DiT feed-forward linears; `llm_dit_mlp` is the intended selected QuantVLA scope.
- ATM/OHB are applied only to DiT attention processors after calibrating teacher/student attention statistics on calibration samples.
- `identity` installs the same custom DiT attention processor with `alpha = 1` and `beta = 1`; it measures processor replacement drift without ATM/OHB rescaling.
- This real-data validation is the bridge between synthetic Phase 3 probes and full LIBERO rollout success-rate evaluation.
