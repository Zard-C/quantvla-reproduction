# QuantVLA Toy Experiment Summary

## Environment

|command|python|torch|cuda|cuda_available|device|
|---|---|---|---|---|---|
|python toy_quantvla/run_toy_experiments.py|3.12.3|2.8.0+cu128|12.8|True|NVIDIA GeForce RTX 5090|

## Conclusions

- atm_multiply_direction_wins: True
- ohb_multiply_direction_wins: True
- vla_attention_quantization_more_fragile_than_mlp: True
- smoothing_gain_larger_under_vla_like: True
- phase3_ready: True

## W4A8 Linear Smoothing

|distribution|method|output_mse|cosine|activation_saturation_ratio|weight_scale_max|
|---|---|---|---|---|---|
|standard_normal|naive_w4a8|0.0135451|0.993241|0.00100708|0.0511874|
|standard_normal|smoothed_w4a8|0.0132462|0.993401|0.00100708|0.146681|
|vla_like_dit_mlp|naive_w4a8|33.8456|0.923863|0.00100708|2.87978|
|vla_like_dit_mlp|smoothed_w4a8|26.1289|0.945467|0.00100708|0.362542|

## Selective Quantization

|distribution|variant|final_output_mse|logits_std_abs_error|attention_js|post_o_rms_abs_error|
|---|---|---|---|---|---|
|standard_normal|mlp_only|0.014132|0|0|0|
|standard_normal|attention_only|0.00711276|0.0142579|0.00310641|0.00516179|
|standard_normal|attention_and_mlp|0.0211799|0.0142579|0.00310641|0.00516179|
|standard_normal|upstream_drift_fp_attention|0|0|0|0|
|vla_like|mlp_only|43900.2|0|0|0|
|vla_like|attention_only|108263|117.548|0.384149|20.0766|
|vla_like|attention_and_mlp|111861|117.548|0.384149|20.0766|
|vla_like|upstream_drift_fp_attention|270265|524.562|0.252193|39.6053|

## ATM Direction

|distribution|direction|alpha_mean|logits_std_mae|attention_js|entropy_abs_error|
|---|---|---|---|---|---|
|standard_normal|none|1|0.00419641|0.000337371|0.00269341|
|standard_normal|multiply|1|0.00419641|0.000337371|0.00269341|
|standard_normal|divide|1|0.00419641|0.000337371|0.00269341|
|vla_like|none|0.746869|14.7218|0.176171|0.0554185|
|vla_like|multiply|0.746869|2.81685|0.171566|0.00357416|
|vla_like|divide|0.746869|30.7068|0.180301|0.0923407|

## OHB Direction

|distribution|direction|beta|teacher_rms|student_rms|rms_abs_error|post_o_mse|
|---|---|---|---|---|---|---|
|standard_normal|none|1|0.311809|0.309279|0.00253016|0.000400989|
|standard_normal|multiply|1|0.311809|0.309279|0.00253016|0.000400989|
|standard_normal|divide|1|0.311809|0.309279|0.00253016|0.000400989|
|vla_like|none|1.06104|42.1143|39.6916|2.42263|1151.01|
|vla_like|multiply|1.06104|42.1143|42.1143|0|1215.03|
|vla_like|divide|1.06104|42.1143|37.4084|4.7059|1101.41|

## Calibration Noise

|distribution|samples|alpha_mean|alpha_std|alpha_neutral_heads|alpha_clamp_hits|beta|calibrated_logits_std_mae|calibrated_post_o_rms_abs_error|
|---|---|---|---|---|---|---|---|---|
|standard_normal|4|1|0|4|0|1|0.00806199|0.00324869|
|standard_normal|8|1|0|4|0|1|0.00806199|0.00324869|
|standard_normal|32|1|0|4|0|1|0.00806199|0.00324869|
|standard_normal|128|1|0|4|0|1|0.00806199|0.00324869|
|vla_like|4|0.910038|0.100777|1|1|1|8.68334|1.83746|
|vla_like|8|0.959712|0.0770445|1|0|1.07792|4.4451|2.56106|
|vla_like|32|1.00022|0.0542073|1|0|1.03533|1.16879|0.156807|
|vla_like|128|0.99593|0.0389124|2|0|1.03255|0.903226|3.8147e-06|

## Distribution Sensitivity

|distribution|metric_group|mlp_only_final_output_mse|attention_only_final_output_mse|both_final_output_mse|attention_vs_mlp_mse_ratio|attention_js_attention_only|none_logits_std_mae|multiply_logits_std_mae|divide_logits_std_mae|multiply_improvement|divide_delta|none_rms_abs_error|multiply_rms_abs_error|divide_rms_abs_error|naive_output_mse|smoothed_output_mse|smoothing_improvement|naive_saturation_ratio|smoothed_saturation_ratio|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|standard_normal|selective_quantization|0.014132|0.00711276|0.0211799|0.503308|0.00310641||||||||||||||
|standard_normal|atm_direction||||||0.00419641|0.00419641|0.00419641|0|0|||||||||
|standard_normal|ohb_direction|||||||||0|0|0.00253016|0.00253016|0.00253016||||||
|vla_like|selective_quantization|43900.2|108263|111861|2.46611|0.384149||||||||||||||
|vla_like|atm_direction||||||14.7218|2.81685|30.7068|11.905|15.9849|||||||||
|vla_like|ohb_direction|||||||||2.42263|2.28327|2.42263|0|4.7059||||||
|standard_normal|linear_smoothing||||||||||||||0.0135451|0.0132462|0.000298981|0.00100708|0.00100708|
|vla_like|linear_smoothing||||||||||||||33.8456|26.1289|7.7168|0.00100708|0.00100708|
