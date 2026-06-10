# Phase 14 Gripper Margin Analysis

Raw gripper threshold assumption: `0.5`.

## Online Same-Observation Traces

| case | success | steps | bin mismatch count | first mismatch | eager switches | compiled switches | max raw gripper diff | min compiled margin |
|---|---:|---:|---:|---:|---|---|---:|---:|
| task4_init9 | True | 224 | 0 | None | `[47, 102, 175]` | `[47, 102, 175]` | 0.00317383 | 0.478553 |
| task6_init8 | True | 649 | 0 | None | `[45, 261, 262, 268, 269, 270, 335, 360, 417, 418, 420, 421, 425, 426, 431, 432, 436, 437, 444, 464, 467, 474, 476, 479, 480, 481, 483, 488, 516]` | `[45, 261, 262, 268, 269, 270, 335, 360, 417, 418, 420, 421, 425, 426, 431, 432, 436, 437, 444, 464, 467, 474, 476, 479, 480, 481, 483, 488, 516]` | 0.00488281 | 0.477863 |

## Closed-Loop Pairs

| case | baseline | compiled | common steps | bin mismatch count | first mismatch | baseline switches | compiled switches | max pre-pos L2 | max action diff |
|---|---:|---:|---:|---:|---:|---|---|---:|---:|
| task4_init9 | True | False | 224 | 26 | 46 | `[47, 102, 175]` | `[46, 97, 195]` | 0.172551 | 2 |
| task6_init8 | True | False | 649 | 142 | 244 | `[45, 261, 262, 268, 269, 270, 335, 360, 417, 418, 420, 421, 425, 426, 431, 432, 436, 437, 444, 464, 467, 474, 476, 479, 480, 481, 483, 488, 516]` | `[45, 244, 245, 268, 269, 270, 335, 360, 381, 382, 383, 400, 402, 405, 423, 523, 525, 531, 558, 591, 592, 594, 595, 596, 597, 600, 631]` | 0.168962 | 2 |

## First Threshold Hits

### task4_init9

- first gripper mismatch: `46`
- first pre-position L2 >= 1mm: `49 (pre_pos_l2=0.00111332)`
- first pre-position L2 >= 5mm: `57 (pre_pos_l2=0.00580318)`
- first action max diff >= 0.01: `8 (action_max_abs_diff=0.0210571)`
- first action max diff >= 0.05: `46 (action_max_abs_diff=2)`

### task6_init8

- first gripper mismatch: `244`
- first pre-position L2 >= 1mm: `63 (pre_pos_l2=0.00102978)`
- first pre-position L2 >= 5mm: `391 (pre_pos_l2=0.00538739)`
- first action max diff >= 0.01: `54 (action_max_abs_diff=0.0485229)`
- first action max diff >= 0.05: `137 (action_max_abs_diff=0.0567627)`
