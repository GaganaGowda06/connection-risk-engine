# Stress test of the frozen champion

Champion run 4c88d55bbbed49018e3ce05e8199ce5b (registry version 2) was scored without refitting, recalibrating or changing any threshold. Its files hash identically before and after the run. The own routes bhm-eus, ast-bhm were scored from 2026-10-16 to 2026-10-22. The new routes bhm-man, bhm-lds, bhm-bri, bhm-liv, bhm-glc were scored from 2026-10-01 to 2026-10-22. Confidence intervals are 95% percentile bootstrap intervals that resample whole service dates (1000 resamples, seed 20261015).

The bar in each cell is the better of the persistence forecast and the historical route-hour baseline on that cell. The historical baseline for a date is fitted only on rows from earlier dates, and until a new route has 10 rows of its own it uses the pooled rows of the other routes.

## Data

| route | services | snapshot rows | scored rows | cancelled rows | rows with under 7 prior days | complete days | partial or missing days |
|---|---|---|---|---|---|---|---|
| bhm-eus | 456 | 2736 | 2664 | 72 | 0 | 7 of 7 | none |
| ast-bhm | 601 | 3606 | 3438 | 168 | 0 | 7 of 7 | none |
| bhm-man | 606 | 3636 | 3486 | 150 | 1164 | 22 of 22 | none |
| bhm-lds | 317 | 1902 | 1776 | 78 | 666 | 22 of 22 | none |
| bhm-bri | 426 | 2556 | 2370 | 186 | 1116 | 22 of 22 | none |
| bhm-liv | 588 | 3528 | 3378 | 150 | 1116 | 22 of 22 | none |
| bhm-glc | 183 | 1098 | 1050 | 48 | 336 | 22 of 22 | none |

Leakage check on real rows from the new routes: 50 rows rebuilt after deleting every actual time after T, 0 feature mismatches (rows by route: {'bhm-bri': 10, 'bhm-glc': 10, 'bhm-lds': 10, 'bhm-liv': 10, 'bhm-man': 10}).

## Results at 15 minutes to arrival

| route | rows | MAE model | MAE persistence | MAE historical | bar | model minus bar | 80% coverage | width | P(late 15) Brier | ECE |
|---|---|---|---|---|---|---|---|---|---|---|
| bhm-eus | 444 (7 d) | 3.52 [2.69, 4.37] | 3.98 [2.86, 5.04] | 6.54 [5.23, 7.91] | persistence | -0.47 [-0.77, -0.08] | 0.840 [0.807, 0.881] | 11.5 [10.2, 12.8] | 0.075 [0.044, 0.103] | 0.044 [0.021, 0.084] |
| ast-bhm | 573 (7 d) | 1.79 [1.50, 2.11] | 2.07 [1.69, 2.52] | 2.30 [1.89, 2.71] | persistence | -0.28 [-0.43, -0.17] | 0.787 [0.753, 0.826] | 6.3 [5.8, 6.7] | 0.005 [0.002, 0.009] | 0.004 [0.001, 0.009] |
| bhm-man | 581 (22 d) | 2.85 [2.51, 3.23] | 2.97 [2.58, 3.39] | 8.27 [7.37, 9.14] | persistence | -0.12 [-0.25, 0.02] | 0.716 [0.660, 0.770] | 7.6 [7.0, 8.2] | 0.048 [0.036, 0.061] | 0.059 [0.046, 0.074] |
| bhm-lds | 296 (22 d) | 3.35 [3.06, 3.69] | 3.12 [2.69, 3.58] | 6.71 [5.51, 8.00] | persistence | 0.23 [-0.11, 0.54] | 0.726 [0.664, 0.788] | 11.3 [10.2, 12.4] | 0.033 [0.021, 0.046] | 0.018 [0.014, 0.042] |
| bhm-bri | 395 (16 d) | 4.28 [3.95, 4.64] | 3.31 [2.83, 3.85] | 8.00 [6.78, 9.36] | persistence | 0.97 [0.54, 1.37] | 0.719 [0.661, 0.774] | 12.4 [11.2, 13.8] | 0.038 [0.024, 0.054] | 0.040 [0.021, 0.061] |
| bhm-liv | 563 (22 d) | 3.40 [3.10, 3.73] | 3.25 [3.01, 3.51] | 4.88 [4.36, 5.58] | persistence | 0.15 [-0.02, 0.32] | 0.456 [0.376, 0.537] | 6.8 [6.4, 7.3] | 0.024 [0.010, 0.039] | 0.008 [0.004, 0.024] |
| bhm-glc | 175 (22 d) | 4.43 [3.80, 5.12] | 3.68 [3.06, 4.54] | 10.47 [8.74, 12.65] | persistence | 0.75 [0.25, 1.16] | 0.743 [0.687, 0.792] | 19.8 [18.0, 21.7] | 0.061 [0.035, 0.093] | 0.040 [0.022, 0.076] |

Paired MAE difference, model minus each baseline (negative favours the model):

| route | vs persistence | vs historical |
|---|---|---|
| bhm-eus | -0.47 [-0.77, -0.08] | -3.02 [-4.02, -2.25] |
| ast-bhm | -0.28 [-0.43, -0.17] | -0.51 [-0.68, -0.35] |
| bhm-man | -0.12 [-0.25, 0.02] | -5.41 [-6.02, -4.76] |
| bhm-lds | 0.23 [-0.11, 0.54] | -3.36 [-4.54, -2.31] |
| bhm-bri | 0.97 [0.54, 1.37] | -3.73 [-4.94, -2.61] |
| bhm-liv | 0.15 [-0.02, 0.32] | -1.48 [-2.04, -0.99] |
| bhm-glc | 0.75 [0.25, 1.16] | -6.04 [-7.81, -4.51] |

Coverage by observed lateness at T (80% nominal):

| route | not_started | 0-4 | 5-14 | 15+ |
|---|---|---|---|---|
| bhm-eus | n/a | 0.85 (n=291) | 0.86 (n=111) | 0.74 (n=42) |
| ast-bhm | 0.73 (n=11) | 0.78 (n=530) | 0.94 (n=32) | n/a |
| bhm-man | n/a | 0.74 (n=303) | 0.66 (n=180) | 0.77 (n=98) |
| bhm-lds | n/a | 0.72 (n=205) | 0.75 (n=55) | 0.75 (n=36) |
| bhm-bri | n/a | 0.69 (n=248) | 0.74 (n=86) | 0.82 (n=61) |
| bhm-liv | n/a | 0.42 (n=415) | 0.56 (n=132) | 0.56 (n=16) |
| bhm-glc | n/a | 0.67 (n=86) | 0.68 (n=38) | 0.90 (n=51) |

## Results at 60 minutes to arrival

| route | rows | MAE model | MAE persistence | MAE historical | bar | model minus bar | 80% coverage | width | P(late 15) Brier | ECE |
|---|---|---|---|---|---|---|---|---|---|---|
| bhm-eus | 444 (7 d) | 4.94 [3.99, 6.14] | 6.01 [4.90, 7.32] | 6.54 [5.23, 7.91] | persistence | -1.07 [-1.29, -0.70] | 0.833 [0.816, 0.853] | 15.8 [14.6, 17.4] | 0.102 [0.073, 0.131] | 0.053 [0.035, 0.090] |
| ast-bhm | 573 (7 d) | 2.27 [1.89, 2.67] | 2.43 [1.87, 3.00] | 2.30 [1.89, 2.71] | historical | -0.03 [-0.12, 0.02] | 0.791 [0.749, 0.831] | 7.3 [7.0, 7.9] | 0.005 [0.002, 0.009] | 0.009 [0.005, 0.012] |
| bhm-man | 581 (22 d) | 4.69 [4.15, 5.24] | 5.46 [4.80, 6.16] | 8.27 [7.37, 9.14] | persistence | -0.77 [-1.09, -0.47] | 0.862 [0.818, 0.903] | 15.9 [15.1, 16.7] | 0.095 [0.074, 0.116] | 0.077 [0.057, 0.104] |
| bhm-lds | 296 (22 d) | 4.78 [4.28, 5.30] | 4.61 [3.95, 5.35] | 6.71 [5.51, 8.00] | persistence | 0.17 [-0.18, 0.55] | 0.777 [0.732, 0.822] | 17.6 [16.3, 19.1] | 0.069 [0.051, 0.091] | 0.061 [0.040, 0.089] |
| bhm-bri | 395 (16 d) | 6.41 [5.81, 6.98] | 5.34 [4.47, 6.25] | 8.00 [6.78, 9.36] | persistence | 1.07 [0.34, 1.79] | 0.797 [0.759, 0.836] | 23.7 [22.5, 25.0] | 0.093 [0.062, 0.126] | 0.028 [0.029, 0.080] |
| bhm-liv | 563 (22 d) | 4.12 [3.78, 4.51] | 4.17 [3.87, 4.51] | 4.88 [4.36, 5.58] | persistence | -0.05 [-0.16, 0.04] | 0.604 [0.534, 0.675] | 11.3 [10.4, 12.3] | 0.029 [0.014, 0.047] | 0.022 [0.012, 0.038] |
| bhm-glc | 175 (22 d) | 6.18 [5.30, 7.22] | 4.53 [3.71, 5.55] | 10.47 [8.74, 12.65] | persistence | 1.65 [0.91, 2.29] | 0.720 [0.667, 0.771] | 21.4 [19.9, 23.1] | 0.077 [0.046, 0.110] | 0.064 [0.041, 0.114] |

Paired MAE difference, model minus each baseline (negative favours the model):

| route | vs persistence | vs historical |
|---|---|---|
| bhm-eus | -1.07 [-1.29, -0.70] | -1.59 [-2.49, -1.02] |
| ast-bhm | -0.16 [-0.35, 0.02] | -0.03 [-0.12, 0.02] |
| bhm-man | -0.77 [-1.09, -0.47] | -3.58 [-4.10, -3.07] |
| bhm-lds | 0.17 [-0.18, 0.55] | -1.93 [-2.98, -0.99] |
| bhm-bri | 1.07 [0.34, 1.79] | -1.59 [-2.57, -0.69] |
| bhm-liv | -0.05 [-0.16, 0.04] | -0.75 [-1.26, -0.32] |
| bhm-glc | 1.65 [0.91, 2.29] | -4.29 [-6.20, -2.70] |

Coverage by observed lateness at T (80% nominal):

| route | not_started | 0-4 | 5-14 | 15+ |
|---|---|---|---|---|
| bhm-eus | 1.00 (n=1) | 0.85 (n=327) | 0.77 (n=83) | 0.79 (n=33) |
| ast-bhm | 0.79 (n=570) | 0.67 (n=3) | n/a | n/a |
| bhm-man | n/a | 0.92 (n=399) | 0.75 (n=123) | 0.71 (n=59) |
| bhm-lds | n/a | 0.85 (n=202) | 0.66 (n=59) | 0.57 (n=35) |
| bhm-bri | n/a | 0.83 (n=292) | 0.77 (n=70) | 0.58 (n=33) |
| bhm-liv | 1.00 (n=9) | 0.58 (n=446) | 0.65 (n=98) | 0.80 (n=10) |
| bhm-glc | n/a | 0.68 (n=72) | 0.67 (n=46) | 0.81 (n=57) |

## Results at 120 minutes to arrival

| route | rows | MAE model | MAE persistence | MAE historical | bar | model minus bar | 80% coverage | width | P(late 15) Brier | ECE |
|---|---|---|---|---|---|---|---|---|---|---|
| bhm-eus | 444 (7 d) | 6.16 [4.90, 7.63] | 7.26 [5.96, 8.67] | 6.54 [5.23, 7.91] | historical | -0.37 [-0.67, -0.14] | 0.815 [0.779, 0.854] | 18.0 [17.1, 19.0] | 0.124 [0.091, 0.154] | 0.069 [0.040, 0.113] |
| ast-bhm | 573 (7 d) | 2.26 [1.84, 2.70] | 2.43 [1.87, 3.00] | 2.30 [1.89, 2.71] | historical | -0.04 [-0.11, 0.02] | 0.825 [0.784, 0.863] | 7.1 [6.9, 7.3] | 0.005 [0.002, 0.009] | 0.011 [0.007, 0.014] |
| bhm-man | 581 (22 d) | 6.29 [5.59, 7.07] | 6.95 [6.16, 7.86] | 8.27 [7.37, 9.14] | persistence | -0.67 [-0.88, -0.47] | 0.859 [0.823, 0.890] | 20.3 [19.5, 21.1] | 0.166 [0.140, 0.196] | 0.139 [0.105, 0.175] |
| bhm-lds | 296 (22 d) | 5.47 [4.65, 6.35] | 5.81 [4.90, 6.78] | 6.71 [5.51, 8.00] | persistence | -0.34 [-0.55, -0.15] | 0.872 [0.827, 0.913] | 20.7 [19.1, 22.3] | 0.089 [0.063, 0.116] | 0.049 [0.030, 0.092] |
| bhm-bri | 395 (16 d) | 6.28 [5.24, 7.33] | 6.45 [5.34, 7.61] | 8.00 [6.78, 9.36] | persistence | -0.17 [-0.37, 0.04] | 0.742 [0.702, 0.781] | 16.6 [15.6, 17.5] | 0.141 [0.105, 0.179] | 0.120 [0.084, 0.169] |
| bhm-liv | 563 (22 d) | 4.63 [4.24, 5.10] | 4.75 [4.28, 5.33] | 4.88 [4.36, 5.58] | persistence | -0.12 [-0.26, 0.03] | 0.615 [0.539, 0.690] | 16.8 [16.5, 17.0] | 0.032 [0.016, 0.051] | 0.023 [0.019, 0.041] |
| bhm-glc | 175 (22 d) | 6.28 [5.14, 7.80] | 5.89 [4.65, 7.52] | 10.47 [8.74, 12.65] | persistence | 0.39 [0.16, 0.57] | 0.766 [0.708, 0.819] | 25.6 [23.2, 28.4] | 0.126 [0.082, 0.177] | 0.102 [0.077, 0.171] |

Paired MAE difference, model minus each baseline (negative favours the model):

| route | vs persistence | vs historical |
|---|---|---|
| bhm-eus | -1.10 [-1.38, -0.70] | -0.37 [-0.67, -0.14] |
| ast-bhm | -0.17 [-0.30, -0.03] | -0.04 [-0.11, 0.02] |
| bhm-man | -0.67 [-0.88, -0.47] | -1.98 [-2.55, -1.43] |
| bhm-lds | -0.34 [-0.55, -0.15] | -1.23 [-2.17, -0.29] |
| bhm-bri | -0.17 [-0.37, 0.04] | -1.72 [-2.59, -0.89] |
| bhm-liv | -0.12 [-0.26, 0.03] | -0.24 [-0.56, 0.01] |
| bhm-glc | 0.39 [0.16, 0.57] | -4.19 [-5.61, -2.88] |

Coverage by observed lateness at T (80% nominal):

| route | not_started | 0-4 | 5-14 | 15+ |
|---|---|---|---|---|
| bhm-eus | 0.83 (n=151) | 0.80 (n=239) | 0.77 (n=35) | 0.95 (n=19) |
| ast-bhm | 0.83 (n=573) | n/a | n/a | n/a |
| bhm-man | 0.88 (n=110) | 0.86 (n=265) | 0.83 (n=166) | 0.90 (n=40) |
| bhm-lds | 0.90 (n=40) | 0.84 (n=178) | 0.92 (n=59) | 1.00 (n=19) |
| bhm-bri | 0.86 (n=29) | 0.72 (n=277) | 0.74 (n=72) | 0.88 (n=17) |
| bhm-liv | 0.61 (n=547) | 0.54 (n=13) | 1.00 (n=2) | 1.00 (n=1) |
| bhm-glc | n/a | 0.62 (n=72) | 0.80 (n=64) | 0.97 (n=39) |

## Gap-free week against the live replay

The live replay lost 88 of 168 hours to capture gaps, so it scored fewer services. Both score the frozen champion on 16 to 22 October.

| route | horizon | gap-free rows | gap-free MAE | gap-free coverage | live rows | live MAE | live coverage |
|---|---|---|---|---|---|---|---|
| bhm-eus | 15 | 444 (7 d) | 3.52 [2.69, 4.37] | 0.840 [0.807, 0.881] | 96 (7 d) | 3.91 [2.77, 5.49] | 0.823 [0.747, 0.900] |
| bhm-eus | 60 | 444 (7 d) | 4.94 [3.99, 6.14] | 0.833 [0.816, 0.853] | 61 (5 d) | 5.38 [4.14, 6.46] | 0.787 [0.583, 0.929] |
| bhm-eus | 120 | 444 (7 d) | 6.16 [4.90, 7.63] | 0.815 [0.779, 0.854] | 48 (3 d) | 6.18 | 0.771 |
| ast-bhm | 15 | 573 (7 d) | 1.79 [1.50, 2.11] | 0.787 [0.753, 0.826] | 138 (7 d) | 2.34 [1.67, 2.68] | 0.775 [0.723, 0.850] |
| ast-bhm | 60 | 573 (7 d) | 2.27 [1.89, 2.67] | 0.791 [0.749, 0.831] | 82 (4 d) | 2.71 | 0.829 |
| ast-bhm | 120 | 573 (7 d) | 2.26 [1.84, 2.70] | 0.825 [0.784, 0.863] | 71 (3 d) | 3.06 | 0.746 |

## Where it breaks

Rows below pool the horizons 15, 60, 120.

### By route

| route | rows | MAE model | MAE persistence | MAE historical | 80% coverage | width |
|---|---|---|---|---|---|---|
| bhm-eus | 1332 | 4.87 [3.88, 6.03] | 5.75 [4.64, 7.00] | 6.54 [5.23, 7.91] | 0.830 [0.805, 0.858] | 15.1 [14.2, 16.2] |
| ast-bhm | 1719 | 2.10 [1.74, 2.49] | 2.31 [1.82, 2.83] | 2.30 [1.89, 2.71] | 0.801 [0.765, 0.834] | 6.9 [6.6, 7.3] |
| bhm-man | 1743 | 4.61 [4.12, 5.16] | 5.13 [4.54, 5.77] | 8.27 [7.37, 9.14] | 0.812 [0.777, 0.849] | 14.6 [14.0, 15.2] |
| bhm-lds | 888 | 4.54 [4.06, 5.05] | 4.51 [3.90, 5.19] | 6.71 [5.51, 8.00] | 0.792 [0.750, 0.828] | 16.5 [15.5, 17.7] |
| bhm-bri | 1185 | 5.66 [5.07, 6.24] | 5.03 [4.26, 5.87] | 8.00 [6.78, 9.36] | 0.753 [0.716, 0.790] | 17.6 [16.6, 18.7] |
| bhm-liv | 1689 | 4.05 [3.75, 4.41] | 4.06 [3.79, 4.37] | 4.88 [4.36, 5.58] | 0.558 [0.486, 0.628] | 11.6 [11.2, 12.2] |
| bhm-glc | 525 | 5.63 [4.81, 6.62] | 4.70 [3.84, 5.89] | 10.47 [8.74, 12.65] | 0.743 [0.705, 0.779] | 22.3 [20.7, 24.0] |

Signed error is the champion's median minus the observed delay in minutes, so positive means it expected the train to be later than it was. The two miss columns are the shares of rows where the observed delay fell below or above the 80% interval.

| route | signed error | missed below | missed above |
|---|---|---|---|
| bhm-eus | -1.81 | 7.5% | 9.5% |
| ast-bhm | -0.88 | 9.8% | 10.1% |
| bhm-man | -2.16 | 2.9% | 15.9% |
| bhm-lds | +0.30 | 13.4% | 7.4% |
| bhm-bri | +0.17 | 14.9% | 9.8% |
| bhm-liv | +0.45 | 39.0% | 5.2% |
| bhm-glc | +1.87 | 20.6% | 5.1% |

### By operator

| operator | rows | MAE model | MAE persistence | MAE historical | 80% coverage | width |
|---|---|---|---|---|---|---|
| AW (unseen in training) | 33 | 7.76 [3.13, 15.86] | 6.82 [2.09, 15.12] | 11.00 [5.09, 20.14] | 0.515 [0.273, 0.728] | 12.2 [10.7, 13.7] |
| LM (seen in training) | 3960 | 3.16 [2.82, 3.65] | 3.32 [2.97, 3.78] | 3.82 [3.35, 4.47] | 0.709 [0.628, 0.752] | 9.7 [9.2, 10.5] |
| VT (seen in training) | 1212 | 5.62 [4.72, 6.63] | 5.99 [4.84, 7.13] | 8.46 [7.28, 9.92] | 0.771 [0.726, 0.806] | 18.7 [17.7, 20.4] |
| XC (unseen in training) | 3876 | 4.93 [4.64, 5.24] | 4.97 [4.54, 5.40] | 7.87 [7.14, 8.63] | 0.792 [0.767, 0.815] | 16.2 [15.4, 16.9] |

### By route and operator (30 or more rows)

| route and operator | rows | MAE model | MAE persistence | MAE historical | 80% coverage | width |
|---|---|---|---|---|---|---|
| ast-bhm/LM (seen operator) | 1719 | 2.10 [1.74, 2.49] | 2.31 [1.82, 2.83] | 2.30 [1.89, 2.71] | 0.801 [0.765, 0.834] | 6.9 [6.6, 7.3] |
| bhm-bri/XC (unseen operator) | 1185 | 5.66 [5.07, 6.24] | 5.03 [4.26, 5.87] | 8.00 [6.78, 9.36] | 0.753 [0.716, 0.790] | 17.6 [16.6, 18.7] |
| bhm-eus/LM (seen operator) | 600 | 3.86 [3.07, 4.62] | 4.27 [3.42, 5.00] | 5.47 [4.07, 6.71] | 0.845 [0.806, 0.875] | 12.7 [11.8, 13.9] |
| bhm-eus/VT (seen operator) | 732 | 5.71 [4.44, 7.37] | 6.97 [5.58, 8.73] | 7.42 [6.06, 9.13] | 0.817 [0.773, 0.862] | 17.0 [15.9, 18.5] |
| bhm-glc/VT (seen operator) | 414 | 5.55 [4.77, 6.49] | 4.46 [3.64, 5.57] | 10.51 [8.55, 13.12] | 0.722 [0.676, 0.770] | 22.5 [20.8, 24.4] |
| bhm-glc/XC (unseen operator) | 111 | 5.93 [4.50, 7.47] | 5.59 [4.07, 7.33] | 10.31 [7.88, 12.69] | 0.820 [0.761, 0.882] | 21.5 [19.1, 24.0] |
| bhm-lds/XC (unseen operator) | 888 | 4.54 [4.06, 5.05] | 4.51 [3.90, 5.19] | 6.71 [5.51, 8.00] | 0.792 [0.750, 0.828] | 16.5 [15.5, 17.7] |
| bhm-liv/LM (seen operator) | 1641 | 4.00 [3.69, 4.36] | 4.02 [3.76, 4.35] | 4.80 [4.32, 5.42] | 0.562 [0.491, 0.633] | 11.5 [11.1, 12.1] |
| bhm-liv/VT (seen operator) | 48 | 5.83 [4.95, 6.68] | 5.23 [4.29, 6.17] | 7.38 [4.31, 11.47] | 0.417 [0.250, 0.604] | 14.7 [13.1, 16.4] |
| bhm-man/AW (unseen operator) | 33 | 7.76 [3.13, 15.86] | 6.82 [2.09, 15.12] | 11.00 [5.09, 20.14] | 0.515 [0.273, 0.728] | 12.2 [10.7, 13.7] |
| bhm-man/XC (unseen operator) | 1692 | 4.57 [4.09, 5.07] | 5.12 [4.53, 5.76] | 8.23 [7.37, 9.08] | 0.817 [0.781, 0.851] | 14.6 [14.1, 15.3] |

### By scheduled journey length (minutes, origin to destination)

| journey | rows | MAE model | MAE persistence | MAE historical | 80% coverage | width |
|---|---|---|---|---|---|---|
| 120 to 179 | 1071 | 4.26 [3.77, 4.75] | 4.45 [3.89, 4.98] | 6.36 [5.21, 7.54] | 0.813 [0.777, 0.837] | 14.4 [13.4, 15.5] |
| 180 or more | 558 | 5.75 [4.93, 6.66] | 4.82 [3.94, 5.92] | 10.50 [8.85, 12.43] | 0.729 [0.688, 0.769] | 21.7 [20.1, 23.4] |
| 75 to 119 | 5733 | 4.76 [4.46, 5.07] | 4.96 [4.57, 5.32] | 6.91 [6.43, 7.40] | 0.728 [0.691, 0.761] | 14.8 [14.1, 15.5] |
| under 30 | 1719 | 2.10 [1.74, 2.49] | 2.31 [1.82, 2.83] | 2.30 [1.89, 2.71] | 0.801 [0.765, 0.834] | 6.9 [6.6, 7.3] |

### By minutes to arrival

Own routes:

| minutes | rows | MAE model | MAE persistence | MAE historical | 80% coverage | width |
|---|---|---|---|---|---|---|
| 5 | 1017 | 2.28 [1.76, 2.74] | 2.62 [1.98, 3.18] | 4.15 [3.45, 4.89] | 0.797 [0.766, 0.832] | 7.7 [6.9, 8.5] |
| 15 | 1017 | 2.54 [2.04, 3.01] | 2.91 [2.20, 3.55] | 4.15 [3.45, 4.89] | 0.810 [0.789, 0.844] | 8.5 [7.7, 9.5] |
| 30 | 1017 | 2.99 [2.41, 3.54] | 3.33 [2.51, 4.08] | 4.15 [3.45, 4.89] | 0.832 [0.812, 0.855] | 9.7 [9.0, 10.6] |
| 60 | 1017 | 3.44 [2.82, 4.12] | 4.00 [3.17, 4.83] | 4.15 [3.45, 4.89] | 0.809 [0.785, 0.838] | 11.0 [10.4, 12.1] |
| 90 | 1017 | 3.83 [3.15, 4.60] | 4.31 [3.46, 5.19] | 4.15 [3.45, 4.89] | 0.825 [0.797, 0.856] | 11.9 [11.4, 12.7] |
| 120 | 1017 | 3.96 [3.26, 4.77] | 4.54 [3.72, 5.44] | 4.15 [3.45, 4.89] | 0.821 [0.784, 0.858] | 11.8 [11.4, 12.6] |

New routes:

| minutes | rows | MAE model | MAE persistence | MAE historical | 80% coverage | width |
|---|---|---|---|---|---|---|
| 5 | 2010 | 2.88 [2.71, 3.06] | 2.87 [2.70, 3.06] | 7.23 [6.66, 7.78] | 0.616 [0.586, 0.648] | 8.0 [7.3, 8.7] |
| 15 | 2010 | 3.50 [3.31, 3.70] | 3.20 [2.99, 3.41] | 7.23 [6.66, 7.78] | 0.648 [0.602, 0.693] | 9.9 [9.3, 10.6] |
| 30 | 2010 | 3.71 [3.47, 3.97] | 3.65 [3.39, 3.95] | 7.23 [6.66, 7.78] | 0.730 [0.685, 0.773] | 12.0 [11.3, 12.8] |
| 60 | 2010 | 5.01 [4.75, 5.28] | 4.87 [4.51, 5.21] | 7.23 [6.66, 7.78] | 0.752 [0.724, 0.780] | 16.9 [16.0, 17.8] |
| 90 | 2010 | 5.53 [5.21, 5.84] | 5.61 [5.23, 5.97] | 7.23 [6.66, 7.78] | 0.775 [0.748, 0.799] | 18.7 [18.0, 19.4] |
| 120 | 2010 | 5.70 [5.31, 6.11] | 5.98 [5.50, 6.45] | 7.23 [6.66, 7.78] | 0.761 [0.736, 0.785] | 19.1 [18.5, 19.7] |


### Interval widening that would have been needed (diagnostic, not applied)

The symmetric widening of the champion's 80% interval, in minutes, that would have given 80% coverage on these same rows. Positive means the interval was too narrow. It is computed on the scored rows themselves, so it describes the gap and is not a recalibration.

| route | 15 min | 60 min | 120 min |
|---|---|---|---|
| bhm-eus | -0.5 | -0.3 | +0.0 |
| ast-bhm | +0.0 | +0.0 | +0.0 |
| bhm-man | +0.8 | -0.8 | -1.4 |
| bhm-lds | +0.3 | +0.4 | -1.1 |
| bhm-bri | +0.5 | +0.1 | +0.6 |
| bhm-liv | +2.8 | +2.1 | +2.0 |
| bhm-glc | +0.8 | +1.1 | +0.5 |

### Feature values outside the range seen in training

Share of each route's rows (all six horizons) whose value lies outside the minimum to maximum seen on the own routes up to 2026-09-22.

| route | last_stop_seq | stops_remaining | sched_minutes_remaining | n_stops_passed | minutes_since_last_stop | hist_median_28d | hist_p80_28d | hist_n_28d |
|---|---|---|---|---|---|---|---|---|
| bhm-eus | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% |
| ast-bhm | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% |
| bhm-man | 0% | 0% | 0% | 0% | 0% | 7% | 2% | 0% |
| bhm-lds | 2% | 0% | 0% | 1% | 0% | 8% | 1% | 0% |
| bhm-bri | 1% | 0% | 0% | 1% | 0% | 6% | 1% | 0% |
| bhm-liv | 0% | 0% | 0% | 0% | 0% | 14% | 2% | 0% |
| bhm-glc | 6% | 0% | 0% | 6% | 0% | 12% | 0% | 0% |

### New routes by history length

| history | rows | MAE model | MAE persistence | MAE historical | 80% coverage |
|---|---|---|---|---|---|
| 7 or more prior days | 3954 | 4.64 [4.32, 4.98] | 4.68 [4.23, 5.12] | 6.96 [6.25, 7.67] | 0.759 [0.738, 0.781] |
| under 7 prior days | 2076 | 4.91 [4.58, 5.30] | 4.68 [4.26, 5.07] | 7.74 [6.92, 8.58] | 0.646 [0.612, 0.688] |

### An unseen route id

The champion was fitted with route ids ast-bhm, bhm-eus and operators LM, VT. The design matrix fixes the category lists at fit time, so a route id or operator code outside them becomes a missing category value (100% of new route rows for the route id, 65% for the operator). LightGBM treats a missing categorical value as missing and sends it down the branch of each split that the split's selected categories do not take, so the model sees an unseen route as no route at all. Predictions for the new routes equal predictions with a made up route id: True. The conformal offsets fall back to the pooled set, because the route has no calibration cell. Pandas already emits a Pandas4Warning for this conversion (values outside a fixed category list) and says it will raise in a future version, so the path needs an explicit fallback before a pandas upgrade.

Pooling the 15, 60 and 120 minute rows of the new routes, MAE and coverage with the route id as it is, and with the route id replaced by each trained route:

| route id given to the model | MAE | 80% coverage |
|---|---|---|
| natural_unseen | 4.74 | 0.720 |
| as_bhm-eus | 4.74 | 0.734 |
| as_ast-bhm | 4.74 | 0.657 |

### The 20 worst services

Ranked by the champion's mean absolute error over the scored horizons. Reason codes are the delay reason codes HSP attaches to the stops up to the destination, shown as code at station; none means HSP gave none. No Delay Repay amounts are given.

| route | date | operator | due (UTC) | delay min | mean abs error | model median at 60 | reason codes |
|---|---|---|---|---|---|---|---|
| bhm-eus | 2026-10-22 | VT | 17:39 | 66 | 59.6 | 6.7 | 547@EUS |
| bhm-man | 2026-10-02 | AW | 23:23 | 48 | 47.1 | 1.0 | 913@BHI, 913@BHM, 913@CRK, 913@CTR, 913@ERL, 913@FRD, 913@GOB, 913@HSB, 913@MAN, 913@NLW, 913@RUA, 913@RUE, 913@SAD, 913@SHR, 913@TFC, 913@WBQ, 913@WLN, 913@WRX, 913@WVH |
| bhm-liv | 2026-10-16 | LM | 22:17 | 65 | 39.6 | 33.4 | 887@ACB, 887@BHM, 887@CRE, 887@HTF, 887@LIV, 887@LPY, 887@PKG, 887@RUN, 887@SGB, 887@STA, 887@WSF, 887@WVH |
| bhm-glc | 2026-10-13 | VT | 12:15 | 44 | 39.4 | 6.7 | 894@GLC |
| bhm-bri | 2026-10-15 | XC | 19:32 | 64 | 39.3 | 24.2 | 777@BHM, 777@BPW, 777@BRI, 777@BUT, 777@BWK, 777@CHD, 777@CNM, 777@DAR, 777@DBY, 777@DHM, 777@EDB, 777@LDS, 777@MPT, 777@NCL, 777@SHF, 777@TAM, 777@WKF, 777@YRK |
| bhm-eus | 2026-10-22 | VT | 17:07 | 43 | 36.6 | 7.0 | 547@EUS |
| bhm-lds | 2026-10-16 | XC | 07:02 | 60 | 36.4 | 14.1 | 522@BHM, 522@BUT, 522@CHD, 522@DBY, 522@LDS, 522@SHF, 522@TAM, 522@WKF |
| bhm-eus | 2026-10-22 | VT | 18:06 | 43 | 35.5 | 6.4 | 547@EUS |
| bhm-eus | 2026-10-22 | VT | 16:54 | 44 | 35.5 | 5.2 | 547@EUS, 547@MKC, 547@WFJ |
| bhm-eus | 2026-10-16 | LM | 19:17 | 50 | 32.7 | 7.1 | 535@BHI, 535@BHM, 535@BKW, 535@CNL, 535@COV, 535@HIA, 535@LBK, 535@RUG, 535@THL, 576@BLY, 576@EUS, 576@LBZ, 576@MKC, 576@NMP, 576@WFJ, 576@WOL |
| bhm-eus | 2026-10-22 | LM | 18:17 | 53 | 31.8 | 16.9 | 811@BLY, 811@EUS, 811@LBK, 811@LBZ |
| bhm-man | 2026-10-10 | XC | 12:29 | 46 | 31.6 | 12.0 | 685@BHI, 685@COV, 685@MAC, 685@MAN, 685@SOT, 685@SPT, 685@STA |
| bhm-man | 2026-10-20 | XC | 16:29 | 53 | 31.5 | 20.9 | 705@MAC, 705@MAN, 705@SOT, 705@SPT, 705@STA, 895@WVH |
| bhm-liv | 2026-10-05 | LM | 18:17 | -33 | 30.9 | 2.6 | none |
| bhm-eus | 2026-10-17 | LM | 18:50 | 57 | 30.1 | 37.1 | 574@BHI, 574@BKW, 574@HIA, 904@BHM, 911@BLY, 911@CNL, 911@COV, 911@EUS, 911@LBK, 911@LBZ, 911@MKC, 911@NMP, 911@RUG, 911@THL, 911@WFJ, 911@WOL |
| bhm-man | 2026-10-08 | XC | 08:59 | 37 | 29.9 | 8.5 | 654@MAC, 654@MAN, 654@SOT, 654@SPT, 654@STA |
| bhm-bri | 2026-10-13 | XC | 16:31 | 44 | 29.8 | 20.2 | none |
| bhm-eus | 2026-10-20 | VT | 13:38 | 34 | 29.5 | 3.4 | 832@EUS |
| bhm-eus | 2026-10-20 | VT | 12:38 | 36 | 29.5 | 5.8 | 832@EUS |
| bhm-liv | 2026-10-15 | LM | 18:17 | -31 | 28.5 | 0.8 | 895@ACB, 895@BHM, 895@CRE, 895@HTF, 895@LIV, 895@LPY, 895@PKG, 895@RUN, 895@SGB, 895@STA, 895@WSF, 895@WVH |

## Failure patterns

1. Coverage falls short on most routes the champion has not seen, in a different direction on each. Pooling the 15, 60 and 120 minute rows, the 80% interval covered 0.83 on bhm-eus and 0.80 on ast-bhm, but 0.81 on bhm-man, 0.79 on bhm-lds, 0.75 on bhm-bri, 0.74 on bhm-glc and 0.56 on bhm-liv. At 5 and 15 minutes to arrival the new routes together covered 0.62 and 0.65, against 0.80 and 0.81 on the own routes. On bhm-liv 39.0% of observed delays fell below the lower bound and 5.2% above it, and the interval would have needed widening by +2.8, +2.1 and +2.0 minutes at 15, 60 and 120 minutes to reach 80%. On bhm-man the misses went the other way (15.9% above, 2.9% below) and the median ran 2.16 minutes early on average. The conformal offsets are the pooled ones for any route with no calibration cell, so none of this is corrected.

2. No point forecast edge over carrying the current lateness forward on the longer or more different routes. The paired difference with persistence has an interval above zero for bhm-bri at 15 minutes (+0.97 [+0.54, +1.37]), bhm-bri at 60 minutes (+1.07 [+0.34, +1.79]), bhm-glc at 15 minutes (+0.75 [+0.25, +1.16]), bhm-glc at 60 minutes (+1.65 [+0.91, +2.29]). On journeys of 180 minutes or more the champion's MAE was 5.75 against 4.82 for persistence. It does beat the historical route-hour baseline on the new routes: in 14 of 15 route and horizon cells the paired interval lies wholly below zero. Replacing the route id with a trained route's id left MAE unchanged at 4.74, 4.74 and 4.74, so the route id itself carries no point information for these routes.

3. Weak start on new routes with little history. Rows with fewer than 7 prior fetched days (2076 rows) were covered 0.646 of the time, against 0.759 with 7 or more days, and the champion's MAE was 4.91 against 4.64, while persistence scored 4.68 and 4.68. The two groups are different days, so this is a pattern and not a controlled comparison.

Tail cases. 9 of the 20 worst services are on bhm-eus, and 10 of the 20 were 30 or more minutes late with a median at 60 minutes below 10. The most repeated reason code among them is 547 on bhm-eus 2026-10-22, on 4 of the 20, which fits one incident affecting several services but does not confirm it. 2 of the 20 are arrivals 20 or more minutes early (bhm-liv 2026-10-05 due 18:17 UTC (-33 min), bhm-liv 2026-10-15 due 18:17 UTC (-31 min)); whether that is a timetable or data effect was not checked. Reason code meanings were not looked up, so no cause is named.

## Gap-free week against the live replay

On bhm-eus and ast-bhm for 16 to 22 October the live replay MAE was higher than the gap-free MAE in 6 of 6 route and horizon cells, and above the gap-free interval in 3. The two scored different services (the live replay lost 88 of 168 hours) from different sources (Darwin against HSP actual times), and this step did not separate those causes.

## Limits

- The own routes have 7 service dates, so their intervals are wide.
- bhm-bri has 16 scored dates. HSP answered every window on 3, 4, 10, 11, 17 and 18 October with no services (HTTP 200, not a failure), while a probe of 26 and 27 September returned services. The cause was not verified.
- New route history features use only dates inside the window, bhm-man included, although older bhm-man data exist. They were not used, so all five new routes start cold in the same way.
- The bundle metadata records the training block as ending 2026-09-22, with later validation and test blocks. Every scored date is after 15 October.
- The bar is the better of the two baselines on each cell, chosen after the fact, which makes it a stricter bar than the validation bar the earlier reports used.
- Delay Repay amounts are not computed.

## Recommended model changes, ranked

These are not implemented. The champion is unchanged.

1. Per-route interval recalibration, with a shared prior until a route has enough days, and separate lower and upper offsets. Evidence: coverage on new routes ranged from 0.56 to 0.81, the shortfall was on opposite sides for bhm-liv and bhm-man, and the widening needed ranged from -1.4 to +2.8 minutes. This is the cheapest change and addresses the coverage promise directly.

2. Route-agnostic features. Drop the route id category (it changed nothing for unseen routes) and express stop counts and time remaining as shares of the journey, adding scheduled journey length. Evidence: the champion was worse than persistence (paired interval above zero) on bhm-bri, bhm-glc and on 180 minute journeys. That the stop counts are the cause was not tested.

3. More training routes, chosen for length and operator. Evidence: the champion was fitted on ast-bhm, bhm-eus with operators LM, VT, and 65% of new route rows had an operator it had never seen. It still beats the historical baseline in 14 of 15 cells, so some of it transfers.

4. History features that survive a cold start: fall back to a pooled operator and hour history for a route's first weeks, and train with randomly shortened histories. Evidence: the short history rows above, and 14% of bhm-liv rows had a hist_median_28d outside the range seen in training.

5. Per-operator history keyed on a cleaner operator identity. Evidence is weaker: the operator code LM covers West Midlands Railway on ast-bhm and London Northwestern Railway on bhm-liv (see the caveat in config/delay_repay.yaml), and the same code gave coverage 0.80 on ast-bhm and 0.56 on bhm-liv, so the code alone is not a usable key.
