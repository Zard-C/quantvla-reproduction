# Phase 4 Random 128 Regression Analysis

Boundary: offline teacher/student action-drift analysis on the cross-episode random 128 held-out run. This is not a simulator benchmark.

## Run

- Source JSON: `toy_quantvla/results/phase4_real_data_validation_d8_cal16_eval128_random.json`
- Config: `llm_dit_mlp`
- Baseline mode: `none`
- Compared modes: `identity`, `atm`, `ohb`, `atm_ohb`
- Calibration observations: `16`
- Evaluation observations: `128`
- Sampling: calibration from episodes 0-15, evaluation from episodes 16-63 using fixed random seed `260204`

## Mode Summary

| mode | worse by NMSE | mean delta NMSE | mean delta rel RMSE | mean delta max abs |
|---|---:|---:|---:|---:|
| identity | 0/128 | 0 | 0 | 0 |
| atm | 37/128 | -0.00194913 | -0.00772748 | -0.0140478 |
| ohb | 24/128 | -0.00180429 | -0.01275 | -0.0119415 |
| atm_ohb | 34/128 | -0.00257947 | -0.0142674 | -0.019289 |

Negative deltas mean the mode improved over `none`. `identity` has zero delta, confirming the custom attention processor path itself is not introducing drift.

## Top 7 `atm_ohb` Regressions

These are the seven held-out observations where `atm_ohb` increased whole-action NMSE most versus `none`.

| rank | dataset index | episode | frame | delta NMSE | delta rel RMSE | delta max abs | top key by NMSE delta | key delta NMSE | top key by max abs delta | key delta max abs |
|---:|---:|---:|---:|---:|---:|---:|---|---:|---|---:|
| 1 | 16630 | 59 | 112 | 0.0691338 | 0.16833 | 0.622192 | action.x | 0.48753 | action.gripper | 0.622192 |
| 2 | 16431 | 58 | 166 | 0.0682148 | 0.105398 | -0.00732422 | action.yaw | 1.51655 | action.x | 0.0468868 |
| 3 | 16005 | 57 | 108 | 0.040938 | 0.0818349 | 0.0402832 | action.gripper | 2.37507 | action.z | 0.0402832 |
| 4 | 7484 | 26 | 203 | 0.0257996 | 0.0687457 | 0.0430298 | action.roll | 0.581604 | action.y | 0.0682068 |
| 5 | 12089 | 43 | 237 | 0.00958882 | 0.0540914 | 0.0256348 | action.gripper | 1.4615 | action.z | 0.0531006 |
| 6 | 11234 | 40 | 90 | 0.00804033 | 0.00984955 | -0.00390625 | action.x | 0.269186 | action.z | 0.0400543 |
| 7 | 16354 | 58 | 89 | 0.00665862 | 0.00527311 | -0.0010376 | action.z | 0.864924 | action.z | 0.0393677 |

## Per-Key Detail For Top 7

### 1. index 16630 / episode 59 / frame 112

| key | base NMSE | atm_ohb NMSE | delta NMSE | base max abs | atm_ohb max abs | delta max abs | teacher RMS |
|---|---:|---:|---:|---:|---:|---:|---:|
| action.x | 0.360234 | 0.847764 | 0.48753 | 0.0457764 | 0.0650024 | 0.0192261 | 0.0204778 |
| action.gripper | 0.0409355 | 0.318124 | 0.277189 | 0.347534 | 0.969727 | 0.622192 | 0.430056 |
| action.yaw | 0.0588679 | 0.120153 | 0.0612848 | 0.0253784 | 0.0344421 | 0.00906372 | 0.039639 |
| action.pitch | 0.0676189 | 0.122116 | 0.0544976 | 0.0114054 | 0.0152072 | 0.00380179 | 0.0245937 |
| action.z | 0.0187594 | 0.0311068 | 0.0123474 | 0.0628388 | 0.086478 | 0.0236392 | 0.229415 |
| action.y | 0.00489231 | 0.00871904 | 0.00382673 | 0.0769043 | 0.189514 | 0.11261 | 0.736356 |
| action.roll | 0.0129882 | 0.00431253 | -0.00867564 | 0.0161086 | 0.00523856 | -0.01087 | 0.0548414 |

### 2. index 16431 / episode 58 / frame 166

| key | base NMSE | atm_ohb NMSE | delta NMSE | base max abs | atm_ohb max abs | delta max abs | teacher RMS |
|---|---:|---:|---:|---:|---:|---:|---:|
| action.yaw | 0.75891 | 2.27546 | 1.51655 | 0.0119641 | 0.0232031 | 0.011239 | 0.00837307 |
| action.pitch | 0.157767 | 0.354951 | 0.197183 | 0.0235418 | 0.0317304 | 0.00818852 | 0.0293663 |
| action.gripper | 0.082786 | 0.161574 | 0.0787882 | 0.989746 | 0.982422 | -0.00732422 | 0.860271 |
| action.roll | 0.0643054 | 0.116237 | 0.0519314 | 0.015994 | 0.017631 | 0.00163704 | 0.0281068 |
| action.z | 0.0328361 | 0.0789861 | 0.04615 | 0.0823975 | 0.126343 | 0.0439453 | 0.246241 |
| action.x | 0.0414041 | 0.0644013 | 0.0229971 | 0.146747 | 0.193634 | 0.0468868 | 0.382925 |
| action.y | 1.64118 | 0.630625 | -1.01056 | 0.015564 | 0.0100708 | -0.00549316 | 0.00724437 |

### 3. index 16005 / episode 57 / frame 108

| key | base NMSE | atm_ohb NMSE | delta NMSE | base max abs | atm_ohb max abs | delta max abs | teacher RMS |
|---|---:|---:|---:|---:|---:|---:|---:|
| action.gripper | 1.38667 | 3.76175 | 2.37507 | 0.00976562 | 0.0131836 | 0.00341797 | 0.00456229 |
| action.x | 0.265798 | 0.467149 | 0.20135 | 0.050354 | 0.0663757 | 0.0160217 | 0.0494473 |
| action.y | 0.0672441 | 0.165173 | 0.0979287 | 0.0210571 | 0.0448608 | 0.0238037 | 0.0427203 |
| action.pitch | 0.0471434 | 0.101425 | 0.0542819 | 0.0198863 | 0.0285135 | 0.00862715 | 0.0370771 |
| action.z | 0.0356486 | 0.0728472 | 0.0371986 | 0.107117 | 0.1474 | 0.0402832 | 0.272189 |
| action.yaw | 0.0373131 | 0.0532066 | 0.0158935 | 0.030355 | 0.0370678 | 0.00671282 | 0.102115 |
| action.roll | 0.179575 | 0.153442 | -0.0261331 | 0.0133911 | 0.0147662 | 0.00137512 | 0.0192914 |

### 4. index 7484 / episode 26 / frame 203

| key | base NMSE | atm_ohb NMSE | delta NMSE | base max abs | atm_ohb max abs | delta max abs | teacher RMS |
|---|---:|---:|---:|---:|---:|---:|---:|
| action.roll | 0.570511 | 1.15211 | 0.581604 | 0.0325036 | 0.0384379 | 0.00593433 | 0.0151265 |
| action.z | 0.383203 | 0.88634 | 0.503137 | 0.310364 | 0.353394 | 0.0430298 | 0.172772 |
| action.y | 0.238257 | 0.57229 | 0.334033 | 0.205307 | 0.273514 | 0.0682068 | 0.167814 |
| action.x | 0.283326 | 0.419465 | 0.136139 | 0.168972 | 0.19558 | 0.0266075 | 0.15619 |
| action.pitch | 0.0769768 | 0.113439 | 0.0364624 | 0.0181316 | 0.0274899 | 0.00935826 | 0.0271775 |
| action.gripper | 2.02751e-05 | 2.15667e-05 | 1.29159e-06 | 0.00976562 | 0.00806427 | -0.00170135 | 0.993686 |
| action.yaw | 0.0125565 | 0.0108562 | -0.00170033 | 0.0229992 | 0.0154083 | -0.00759085 | 0.101302 |

### 5. index 12089 / episode 43 / frame 237

| key | base NMSE | atm_ohb NMSE | delta NMSE | base max abs | atm_ohb max abs | delta max abs | teacher RMS |
|---|---:|---:|---:|---:|---:|---:|---:|
| action.gripper | 0.690092 | 2.15159 | 1.4615 | 0.0102539 | 0.0167236 | 0.00646973 | 0.00600547 |
| action.yaw | 0.070303 | 0.0986696 | 0.0283666 | 0.0154055 | 0.0142102 | -0.00119528 | 0.0186829 |
| action.x | 0.00268474 | 0.0308749 | 0.0281902 | 0.025177 | 0.0640869 | 0.0389099 | 0.238877 |
| action.z | 0.00631762 | 0.0334185 | 0.0271009 | 0.0210571 | 0.0741577 | 0.0531006 | 0.157074 |
| action.y | 0.00334226 | 0.00463414 | 0.00129188 | 0.0485229 | 0.067749 | 0.0192261 | 0.431608 |
| action.roll | 0.0760277 | 0.072909 | -0.00311868 | 0.0130964 | 0.011001 | -0.00209543 | 0.022285 |
| action.pitch | 0.35683 | 0.213339 | -0.143491 | 0.00263202 | 0.00204712 | -0.0005849 | 0.00180867 |

### 6. index 11234 / episode 40 / frame 90

| key | base NMSE | atm_ohb NMSE | delta NMSE | base max abs | atm_ohb max abs | delta max abs | teacher RMS |
|---|---:|---:|---:|---:|---:|---:|---:|
| action.x | 0.99504 | 1.26423 | 0.269186 | 0.148315 | 0.179443 | 0.0311279 | 0.0663786 |
| action.z | 0.221681 | 0.291226 | 0.0695445 | 0.277519 | 0.317574 | 0.0400543 | 0.27247 |
| action.roll | 0.236402 | 0.276393 | 0.0399904 | 0.0207578 | 0.0211507 | 0.000392884 | 0.0168805 |
| action.pitch | 0.402003 | 0.432511 | 0.0305083 | 0.0681399 | 0.0783937 | 0.0102538 | 0.0544279 |
| action.gripper | 0.165013 | 0.163708 | -0.00130499 | 0.989746 | 0.98584 | -0.00390625 | 0.609357 |
| action.yaw | 0.0622167 | 0.0479539 | -0.0142628 | 0.219705 | 0.212816 | -0.00688842 | 0.313812 |
| action.y | 1.8043 | 0.794399 | -1.0099 | 0.0228882 | 0.0192261 | -0.00366211 | 0.0111231 |

### 7. index 16354 / episode 58 / frame 89

| key | base NMSE | atm_ohb NMSE | delta NMSE | base max abs | atm_ohb max abs | delta max abs | teacher RMS |
|---|---:|---:|---:|---:|---:|---:|---:|
| action.z | 1.39841 | 2.26333 | 0.864924 | 0.134583 | 0.17395 | 0.0393677 | 0.0557279 |
| action.y | 0.158433 | 0.257786 | 0.0993532 | 0.0410271 | 0.0427437 | 0.00171661 | 0.047538 |
| action.pitch | 0.484057 | 0.574148 | 0.0900914 | 0.0104549 | 0.0104915 | 3.65674e-05 | 0.00550515 |
| action.roll | 0.341548 | 0.35369 | 0.0121418 | 0.0235735 | 0.0233116 | -0.000261933 | 0.0162954 |
| action.gripper | 0.389268 | 0.388905 | -0.000362618 | 0.986633 | 0.985596 | -0.0010376 | 0.557162 |
| action.x | 0.318613 | 0.256973 | -0.0616404 | 0.12085 | 0.109863 | -0.0109863 | 0.0988634 |
| action.yaw | 0.742695 | 0.5565 | -0.186195 | 0.0233844 | 0.0201215 | -0.00326294 | 0.012829 |

## Interpretation

- `atm` and `ohb` each improve mean drift versus `none`; combined `atm_ohb` gives the best mean NMSE and relative RMSE in this run.
- Regressions are not dominated by the processor replacement path, because `identity` exactly matches `none`.
- The largest per-key NMSE regressions often appear on low-RMS rotation/gripper components; max-absolute regressions identify the physically larger deviations separately.
- This remains an offline action-drift result. It should gate a small simulator smoke test, not a full benchmark claim.
