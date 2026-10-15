# Model results

MLflow run c73876aafecc42d9af5247c817237157, commit 0bd5b2bf0d1fb8e3006a0b01b943428c742c45b2. The model was trained on 77 complete days (2026-06-26 to 2026-09-19), tuned and calibrated on 14 validation days (2026-09-23 to 2026-10-06), the earlier half tuning and the later half calibrating, and scored once on 7 test days (2026-10-08 to 2026-10-14). Brackets are 95% bootstrap intervals over service dates. Intervals are nominal 80% intervals. Delay is minutes late at the route destination, and cancelled services are not scored.

The interval comes from LightGBM quantile models at 0.1, 0.5 and 0.9 with the horizon as a feature, widened by a conformal offset computed separately for each horizon and each observed lateness group at T (not started, 0 to 4, 5 to 14, and 15 or more minutes late). The lateness group uses only what is known at T, and a group with fewer than 30 calibration rows is merged into its neighbour. The target per horizon was chosen by tuning pinball loss between the direct delay and the delay minus the lateness at T: T-120: direct, T-60: residual, T-15: residual. Band probabilities come from a LightGBM band model with temperature calibration. P(at least 15) and P(at least 30) are then recalibrated with platt regression fitted on the calibration rows and chosen by log loss on the tuning rows, and the band probabilities are rebuilt from the recalibrated exceedances so that they stay monotone and sum to 1. The delay bands are provisional.

Two kinds of check are reported. The conformal guarantee is coverage conditional on the state known at T, so coverage by observed lateness is the interval check. For passengers and Delay Repay, what matters is whether the probabilities of reaching 15, 30 and 60 minutes late are calibrated, so threshold reliability is the passenger facing check. Coverage by true delay band is shown for description only.

## T-120 minutes, test block

| Method | MAE | 80% coverage | Mean width | Brier (5 bands) | Log loss |
|---|---|---|---|---|---|
| Timetable | 8.88 (7.59 to 10.14) | n/a | n/a | 0.366 (0.258 to 0.460) | 6.32 |
| Persistence | 7.61 (6.09 to 8.90) | 0.838 (0.782 to 0.892) | 23.0 | 0.380 (0.262 to 0.469) | 6.56 |
| Historical | 7.11 (5.87 to 8.29) | 0.866 (0.815 to 0.923) | 24.2 | 0.320 (0.245 to 0.388) | 0.62 |
| Model, global CQR | 6.31 (5.21 to 7.34) | 0.815 (0.743 to 0.894) | 19.6 | 0.309 (0.211 to 0.390) | 0.69 |
| Model, Mondrian CQR | 6.31 (5.21 to 7.34) | 0.819 (0.755 to 0.894) | 20.1 | 0.309 (0.211 to 0.390) | 0.69 |

Against the bar, historical (lowest validation MAE), the model's MAE is -0.81 minutes (-1.63 to -0.04, paired bootstrap over service dates). The interval excludes zero.

Interval coverage by observed lateness at T, which is what the conformal guarantee covers:

| Method | not_started | 0-4 | 5-14 | 15+ |
|---|---|---|---|---|
| Timetable | n/a | n/a | n/a | n/a |
| Persistence | 0.86 (n=147, width 23.0) | 0.88 (n=213, width 23.0) | 0.75 (n=55, width 23.0) | 0.35 (n=17, width 23.0) |
| Historical | 0.88 (n=147, width 23.2) | 0.87 (n=213, width 24.3) | 0.93 (n=55, width 25.6) | 0.53 (n=17, width 26.7) |
| Model, global CQR | 0.85 (n=147, width 22.9) | 0.78 (n=213, width 15.4) | 0.85 (n=55, width 18.7) | 0.82 (n=17, width 46.6) |
| Model, Mondrian CQR | 0.84 (n=147, width 22.5) | 0.78 (n=213, width 15.3) | 0.93 (n=55, width 22.9) | 0.82 (n=17, width 50.8) |

Threshold reliability, derived from the band probabilities. Brier is the binary Brier score and ECE the expected calibration error over 10 equal-count bins. Historical band frequencies are the bar.

| Threshold | Share of rows at or above | Method | Brier | ECE |
|---|---|---|---|---|
| 15 minutes | 0.183 | Model, after threshold calibration | 0.146 (0.099 to 0.188) | 0.093 (0.051 to 0.146) |
| 15 minutes | 0.183 | Model, before threshold calibration | 0.144 (0.097 to 0.185) | 0.075 (0.038 to 0.140) |
| 15 minutes | 0.183 | Historical | 0.155 (0.122 to 0.185) | 0.064 (0.049 to 0.154) |
| 30 minutes | 0.044 | Model, after threshold calibration | 0.035 (0.012 to 0.057) | 0.020 (0.011 to 0.050) |
| 30 minutes | 0.044 | Model, before threshold calibration | 0.034 (0.012 to 0.055) | 0.021 (0.011 to 0.048) |
| 30 minutes | 0.044 | Historical | 0.044 (0.018 to 0.069) | 0.050 (0.038 to 0.071) |
| 60 minutes | 0.002 | Model, after threshold calibration | 0.003 (0.000 to 0.008) | 0.001 (0.000 to 0.007) |
| 60 minutes | 0.002 | Model, before threshold calibration | 0.003 (0.000 to 0.008) | 0.001 (0.000 to 0.007) |
| 60 minutes | 0.002 | Historical | 0.002 (0.000 to 0.008) | 0.011 (0.010 to 0.017) |

Reliability for P(at least 15 minutes late) at T-120, 10 equal-count bins from the lowest predicted probability to the highest:

| Bin | Model n | Model predicted | Model observed | Historical predicted | Historical observed |
|---|---|---|---|---|---|
| 1 | 44 | 0.01 (0.01 to 0.02) | 0.14 (0.04 to 0.23) | 0.10 (0.10 to 0.11) | 0.27 (0.08 to 0.46) |
| 2 | 44 | 0.02 (0.02 to 0.02) | 0.11 (0.00 to 0.22) | 0.14 (0.14 to 0.14) | 0.14 (0.06 to 0.26) |
| 3 | 43 | 0.02 (0.02 to 0.03) | 0.09 (0.02 to 0.19) | 0.15 (0.15 to 0.16) | 0.16 (0.03 to 0.38) |
| 4 | 43 | 0.03 (0.03 to 0.04) | 0.12 (0.00 to 0.24) | 0.17 (0.16 to 0.17) | 0.19 (0.03 to 0.44) |
| 5 | 43 | 0.04 (0.04 to 0.06) | 0.16 (0.02 to 0.30) | 0.19 (0.19 to 0.21) | 0.16 (0.00 to 0.28) |
| 6 | 43 | 0.07 (0.05 to 0.08) | 0.14 (0.05 to 0.31) | 0.22 (0.22 to 0.22) | 0.21 (0.07 to 0.34) |
| 7 | 43 | 0.09 (0.07 to 0.11) | 0.23 (0.05 to 0.41) | 0.23 (0.23 to 0.23) | 0.26 (0.12 to 0.39) |
| 8 | 43 | 0.11 (0.09 to 0.14) | 0.19 (0.05 to 0.31) | 0.25 (0.25 to 0.26) | 0.14 (0.02 to 0.28) |
| 9 | 43 | 0.15 (0.11 to 0.19) | 0.16 (0.05 to 0.35) | 0.26 (0.26 to 0.27) | 0.12 (0.05 to 0.33) |
| 10 | 43 | 0.36 (0.25 to 0.48) | 0.49 (0.20 to 0.69) | 0.29 (0.29 to 0.29) | 0.19 (0.04 to 0.29) |

Reliability for P(at least 30 minutes late) at T-120, 10 equal-count bins from the lowest predicted probability to the highest:

| Bin | Model n | Model predicted | Model observed | Historical predicted | Historical observed |
|---|---|---|---|---|---|
| 1 | 44 | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) | 0.02 (0.02 to 0.03) | 0.07 (0.00 to 0.15) |
| 2 | 44 | 0.00 (0.00 to 0.01) | 0.00 (0.00 to 0.03) | 0.03 (0.03 to 0.04) | 0.00 (0.00 to 0.14) |
| 3 | 43 | 0.01 (0.00 to 0.01) | 0.02 (0.00 to 0.07) | 0.04 (0.04 to 0.04) | 0.14 (0.00 to 0.21) |
| 4 | 43 | 0.01 (0.01 to 0.01) | 0.00 (0.00 to 0.07) | 0.05 (0.05 to 0.05) | 0.00 (0.00 to 0.10) |
| 5 | 43 | 0.01 (0.01 to 0.02) | 0.05 (0.00 to 0.10) | 0.06 (0.06 to 0.06) | 0.05 (0.00 to 0.09) |
| 6 | 43 | 0.02 (0.01 to 0.02) | 0.05 (0.00 to 0.13) | 0.07 (0.06 to 0.07) | 0.05 (0.00 to 0.14) |
| 7 | 43 | 0.02 (0.02 to 0.03) | 0.05 (0.00 to 0.13) | 0.08 (0.07 to 0.08) | 0.00 (0.00 to 0.08) |
| 8 | 43 | 0.03 (0.02 to 0.04) | 0.07 (0.00 to 0.15) | 0.08 (0.08 to 0.08) | 0.09 (0.00 to 0.27) |
| 9 | 43 | 0.04 (0.03 to 0.05) | 0.05 (0.00 to 0.12) | 0.09 (0.09 to 0.09) | 0.05 (0.00 to 0.11) |
| 10 | 43 | 0.13 (0.08 to 0.19) | 0.16 (0.02 to 0.38) | 0.11 (0.11 to 0.11) | 0.00 (0.00 to 0.03) |

Reliability for P(at least 60 minutes late) at T-120, 10 equal-count bins from the lowest predicted probability to the highest:

| Bin | Model n | Model predicted | Model observed | Historical predicted | Historical observed |
|---|---|---|---|---|---|
| 1 | 44 | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) |
| 2 | 44 | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) |
| 3 | 43 | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) | 0.01 (0.00 to 0.01) | 0.02 (0.00 to 0.08) |
| 4 | 43 | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) | 0.01 (0.01 to 0.01) | 0.00 (0.00 to 0.00) |
| 5 | 43 | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) | 0.01 (0.01 to 0.01) | 0.00 (0.00 to 0.00) |
| 6 | 43 | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) | 0.01 (0.01 to 0.01) | 0.00 (0.00 to 0.00) |
| 7 | 43 | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) | 0.01 (0.01 to 0.01) | 0.00 (0.00 to 0.00) |
| 8 | 43 | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) | 0.01 (0.01 to 0.01) | 0.00 (0.00 to 0.00) |
| 9 | 43 | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) | 0.02 (0.01 to 0.02) | 0.00 (0.00 to 0.00) |
| 10 | 43 | 0.03 (0.01 to 0.05) | 0.02 (0.00 to 0.08) | 0.02 (0.02 to 0.02) | 0.00 (0.00 to 0.00) |

Interval coverage by true delay band, descriptive only. A band is selected by the outcome that the interval was built before knowing, so it cannot reach the nominal level for any forecast that is not perfect.

| Method | 0-14 | 15-29 | 30-59 | 60-119 | 120+ |
|---|---|---|---|---|---|
| Timetable | n/a | n/a | n/a | n/a | n/a |
| Persistence | 0.90 (n=353, width 23.0) | 0.68 (n=60, width 23.0) | 0.22 (n=18, width 23.0) | 1.00 (n=1, width 23.0) | n/a |
| Historical | 0.94 (n=353, width 24.3) | 0.68 (n=60, width 23.6) | 0.00 (n=18, width 23.2) | 0.00 (n=1, width 31.7) | n/a |
| Model, global CQR | 0.90 (n=353, width 18.8) | 0.47 (n=60, width 19.7) | 0.33 (n=18, width 32.6) | 0.00 (n=1, width 51.8) | n/a |
| Model, Mondrian CQR | 0.90 (n=353, width 19.3) | 0.50 (n=60, width 20.3) | 0.33 (n=18, width 34.3) | 0.00 (n=1, width 56.0) | n/a |

## T-60 minutes, test block

| Method | MAE | 80% coverage | Mean width | Brier (5 bands) | Log loss |
|---|---|---|---|---|---|
| Timetable | 8.88 (7.59 to 10.14) | n/a | n/a | 0.366 (0.258 to 0.460) | 6.32 |
| Persistence | 6.73 (5.41 to 7.92) | 0.789 (0.731 to 0.841) | 19.0 | 0.361 (0.261 to 0.433) | 6.24 |
| Historical | 7.11 (5.87 to 8.29) | 0.866 (0.815 to 0.923) | 24.2 | 0.320 (0.245 to 0.388) | 0.62 |
| Model, global CQR | 5.78 (4.84 to 6.70) | 0.782 (0.738 to 0.827) | 16.6 | 0.281 (0.197 to 0.355) | 0.59 |
| Model, Mondrian CQR | 5.78 (4.84 to 6.70) | 0.796 (0.748 to 0.842) | 17.1 | 0.281 (0.197 to 0.355) | 0.59 |

Against the bar, persistence (lowest validation MAE), the model's MAE is -0.95 minutes (-1.46 to -0.11, paired bootstrap over service dates). The interval excludes zero.

Interval coverage by observed lateness at T, which is what the conformal guarantee covers:

| Method | not_started | 0-4 | 5-14 | 15+ |
|---|---|---|---|---|
| Timetable | n/a | n/a | n/a | n/a |
| Persistence | 0.50 (n=2, width 19.0) | 0.84 (n=302, width 19.0) | 0.75 (n=103, width 19.0) | 0.40 (n=25, width 19.0) |
| Historical | 0.50 (n=2, width 19.5) | 0.87 (n=302, width 24.0) | 0.92 (n=103, width 24.6) | 0.60 (n=25, width 24.9) |
| Model, global CQR | 0.00 (n=2, width 42.6) | 0.78 (n=302, width 15.0) | 0.83 (n=103, width 17.1) | 0.68 (n=25, width 30.7) |
| Model, Mondrian CQR | 0.00 (n=2, width 42.4) | 0.77 (n=302, width 14.9) | 0.91 (n=103, width 19.2) | 0.68 (n=25, width 32.8) |

Threshold reliability, derived from the band probabilities. Brier is the binary Brier score and ECE the expected calibration error over 10 equal-count bins. Historical band frequencies are the bar.

| Threshold | Share of rows at or above | Method | Brier | ECE |
|---|---|---|---|---|
| 15 minutes | 0.183 | Model, after threshold calibration | 0.133 (0.090 to 0.171) | 0.075 (0.045 to 0.131) |
| 15 minutes | 0.183 | Model, before threshold calibration | 0.135 (0.095 to 0.171) | 0.086 (0.057 to 0.136) |
| 15 minutes | 0.183 | Historical | 0.155 (0.122 to 0.185) | 0.066 (0.048 to 0.154) |
| 30 minutes | 0.044 | Model, after threshold calibration | 0.031 (0.010 to 0.052) | 0.023 (0.012 to 0.048) |
| 30 minutes | 0.044 | Model, before threshold calibration | 0.031 (0.010 to 0.053) | 0.023 (0.009 to 0.052) |
| 30 minutes | 0.044 | Historical | 0.044 (0.018 to 0.069) | 0.050 (0.038 to 0.073) |
| 60 minutes | 0.002 | Model, after threshold calibration | 0.003 (0.000 to 0.008) | 0.001 (0.000 to 0.007) |
| 60 minutes | 0.002 | Model, before threshold calibration | 0.003 (0.000 to 0.008) | 0.001 (0.000 to 0.007) |
| 60 minutes | 0.002 | Historical | 0.002 (0.000 to 0.008) | 0.011 (0.010 to 0.017) |

Reliability for P(at least 15 minutes late) at T-60, 10 equal-count bins from the lowest predicted probability to the highest:

| Bin | Model n | Model predicted | Model observed | Historical predicted | Historical observed |
|---|---|---|---|---|---|
| 1 | 44 | 0.01 (0.01 to 0.01) | 0.09 (0.02 to 0.17) | 0.10 (0.10 to 0.11) | 0.27 (0.08 to 0.46) |
| 2 | 44 | 0.01 (0.01 to 0.01) | 0.07 (0.00 to 0.15) | 0.14 (0.14 to 0.14) | 0.14 (0.06 to 0.27) |
| 3 | 43 | 0.02 (0.01 to 0.02) | 0.12 (0.02 to 0.16) | 0.15 (0.15 to 0.16) | 0.16 (0.03 to 0.38) |
| 4 | 43 | 0.02 (0.02 to 0.02) | 0.05 (0.00 to 0.23) | 0.17 (0.16 to 0.17) | 0.19 (0.03 to 0.43) |
| 5 | 43 | 0.03 (0.02 to 0.04) | 0.23 (0.02 to 0.36) | 0.19 (0.19 to 0.21) | 0.14 (0.00 to 0.28) |
| 6 | 43 | 0.04 (0.03 to 0.06) | 0.16 (0.08 to 0.41) | 0.22 (0.22 to 0.22) | 0.23 (0.07 to 0.33) |
| 7 | 43 | 0.06 (0.04 to 0.10) | 0.14 (0.05 to 0.24) | 0.23 (0.23 to 0.23) | 0.26 (0.12 to 0.39) |
| 8 | 43 | 0.10 (0.06 to 0.14) | 0.14 (0.03 to 0.26) | 0.25 (0.25 to 0.26) | 0.14 (0.02 to 0.28) |
| 9 | 43 | 0.19 (0.11 to 0.37) | 0.21 (0.03 to 0.49) | 0.26 (0.26 to 0.27) | 0.12 (0.05 to 0.32) |
| 10 | 43 | 0.65 (0.47 to 0.76) | 0.63 (0.39 to 0.73) | 0.29 (0.29 to 0.29) | 0.19 (0.04 to 0.29) |

Reliability for P(at least 30 minutes late) at T-60, 10 equal-count bins from the lowest predicted probability to the highest:

| Bin | Model n | Model predicted | Model observed | Historical predicted | Historical observed |
|---|---|---|---|---|---|
| 1 | 44 | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) | 0.02 (0.02 to 0.03) | 0.07 (0.00 to 0.15) |
| 2 | 44 | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) | 0.03 (0.03 to 0.04) | 0.00 (0.00 to 0.14) |
| 3 | 43 | 0.00 (0.00 to 0.01) | 0.00 (0.00 to 0.05) | 0.04 (0.04 to 0.04) | 0.14 (0.00 to 0.23) |
| 4 | 43 | 0.01 (0.00 to 0.01) | 0.07 (0.00 to 0.11) | 0.05 (0.05 to 0.05) | 0.00 (0.00 to 0.09) |
| 5 | 43 | 0.01 (0.01 to 0.01) | 0.05 (0.00 to 0.11) | 0.06 (0.06 to 0.06) | 0.05 (0.00 to 0.09) |
| 6 | 43 | 0.01 (0.01 to 0.01) | 0.05 (0.00 to 0.14) | 0.07 (0.06 to 0.07) | 0.05 (0.00 to 0.14) |
| 7 | 43 | 0.01 (0.01 to 0.02) | 0.00 (0.00 to 0.11) | 0.08 (0.07 to 0.08) | 0.00 (0.00 to 0.08) |
| 8 | 43 | 0.02 (0.01 to 0.03) | 0.00 (0.00 to 0.05) | 0.08 (0.08 to 0.08) | 0.09 (0.00 to 0.27) |
| 9 | 43 | 0.04 (0.02 to 0.06) | 0.05 (0.00 to 0.10) | 0.09 (0.09 to 0.09) | 0.05 (0.00 to 0.11) |
| 10 | 43 | 0.20 (0.10 to 0.34) | 0.23 (0.02 to 0.51) | 0.11 (0.11 to 0.11) | 0.00 (0.00 to 0.03) |

Reliability for P(at least 60 minutes late) at T-60, 10 equal-count bins from the lowest predicted probability to the highest:

| Bin | Model n | Model predicted | Model observed | Historical predicted | Historical observed |
|---|---|---|---|---|---|
| 1 | 44 | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) |
| 2 | 44 | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) |
| 3 | 43 | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) | 0.01 (0.00 to 0.01) | 0.02 (0.00 to 0.08) |
| 4 | 43 | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) | 0.01 (0.01 to 0.01) | 0.00 (0.00 to 0.00) |
| 5 | 43 | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) | 0.01 (0.01 to 0.01) | 0.00 (0.00 to 0.00) |
| 6 | 43 | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) | 0.01 (0.01 to 0.01) | 0.00 (0.00 to 0.00) |
| 7 | 43 | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) | 0.01 (0.01 to 0.01) | 0.00 (0.00 to 0.00) |
| 8 | 43 | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) | 0.01 (0.01 to 0.01) | 0.00 (0.00 to 0.00) |
| 9 | 43 | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) | 0.02 (0.01 to 0.02) | 0.00 (0.00 to 0.00) |
| 10 | 43 | 0.03 (0.01 to 0.05) | 0.02 (0.00 to 0.08) | 0.02 (0.02 to 0.02) | 0.00 (0.00 to 0.00) |

Interval coverage by true delay band, descriptive only. A band is selected by the outcome that the interval was built before knowing, so it cannot reach the nominal level for any forecast that is not perfect.

| Method | 0-14 | 15-29 | 30-59 | 60-119 | 120+ |
|---|---|---|---|---|---|
| Timetable | n/a | n/a | n/a | n/a | n/a |
| Persistence | 0.86 (n=353, width 19.0) | 0.57 (n=60, width 19.0) | 0.22 (n=18, width 19.0) | 0.00 (n=1, width 19.0) | n/a |
| Historical | 0.94 (n=353, width 24.3) | 0.68 (n=60, width 23.6) | 0.00 (n=18, width 23.2) | 0.00 (n=1, width 31.7) | n/a |
| Model, global CQR | 0.85 (n=353, width 15.8) | 0.50 (n=60, width 16.0) | 0.39 (n=18, width 29.7) | 0.00 (n=1, width 67.7) | n/a |
| Model, Mondrian CQR | 0.87 (n=353, width 16.3) | 0.52 (n=60, width 16.8) | 0.39 (n=18, width 30.9) | 0.00 (n=1, width 69.8) | n/a |

## T-15 minutes, test block

| Method | MAE | 80% coverage | Mean width | Brier (5 bands) | Log loss |
|---|---|---|---|---|---|
| Timetable | 8.88 (7.59 to 10.14) | n/a | n/a | 0.366 (0.258 to 0.460) | 6.32 |
| Persistence | 4.25 (3.20 to 5.23) | 0.803 (0.783 to 0.824) | 12.0 | 0.278 (0.188 to 0.351) | 4.80 |
| Historical | 7.11 (5.87 to 8.29) | 0.866 (0.815 to 0.923) | 24.2 | 0.320 (0.245 to 0.388) | 0.62 |
| Model, global CQR | 4.06 (3.25 to 4.93) | 0.806 (0.756 to 0.863) | 12.1 | 0.217 (0.155 to 0.277) | 0.45 |
| Model, Mondrian CQR | 4.06 (3.25 to 4.93) | 0.812 (0.764 to 0.872) | 12.6 | 0.217 (0.155 to 0.277) | 0.45 |

Against the bar, persistence (lowest validation MAE), the model's MAE is -0.20 minutes (-0.53 to +0.18, paired bootstrap over service dates). The interval includes zero, so the difference is not statistically significant.

Interval coverage by observed lateness at T, which is what the conformal guarantee covers:

| Method | not_started | 0-4 | 5-14 | 15+ |
|---|---|---|---|---|
| Timetable | n/a | n/a | n/a | n/a |
| Persistence | n/a | 0.79 (n=241, width 12.0) | 0.85 (n=151, width 12.0) | 0.70 (n=40, width 12.0) |
| Historical | n/a | 0.85 (n=241, width 23.5) | 0.97 (n=151, width 25.2) | 0.53 (n=40, width 24.7) |
| Model, global CQR | n/a | 0.77 (n=241, width 10.9) | 0.86 (n=151, width 12.3) | 0.82 (n=40, width 18.9) |
| Model, Mondrian CQR | n/a | 0.77 (n=241, width 10.9) | 0.87 (n=151, width 12.4) | 0.88 (n=40, width 24.3) |

Threshold reliability, derived from the band probabilities. Brier is the binary Brier score and ECE the expected calibration error over 10 equal-count bins. Historical band frequencies are the bar.

| Threshold | Share of rows at or above | Method | Brier | ECE |
|---|---|---|---|---|
| 15 minutes | 0.183 | Model, after threshold calibration | 0.100 (0.066 to 0.130) | 0.058 (0.039 to 0.104) |
| 15 minutes | 0.183 | Model, before threshold calibration | 0.101 (0.066 to 0.132) | 0.066 (0.036 to 0.110) |
| 15 minutes | 0.183 | Historical | 0.155 (0.122 to 0.185) | 0.064 (0.049 to 0.155) |
| 30 minutes | 0.044 | Model, after threshold calibration | 0.024 (0.006 to 0.044) | 0.023 (0.008 to 0.043) |
| 30 minutes | 0.044 | Model, before threshold calibration | 0.022 (0.003 to 0.042) | 0.015 (0.004 to 0.039) |
| 30 minutes | 0.044 | Historical | 0.044 (0.018 to 0.069) | 0.050 (0.038 to 0.073) |
| 60 minutes | 0.002 | Model, after threshold calibration | 0.002 (0.000 to 0.008) | 0.001 (0.000 to 0.007) |
| 60 minutes | 0.002 | Model, before threshold calibration | 0.002 (0.000 to 0.008) | 0.001 (0.000 to 0.007) |
| 60 minutes | 0.002 | Historical | 0.002 (0.000 to 0.008) | 0.011 (0.010 to 0.017) |

Reliability for P(at least 15 minutes late) at T-15, 10 equal-count bins from the lowest predicted probability to the highest:

| Bin | Model n | Model predicted | Model observed | Historical predicted | Historical observed |
|---|---|---|---|---|---|
| 1 | 44 | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) | 0.10 (0.10 to 0.11) | 0.27 (0.08 to 0.46) |
| 2 | 44 | 0.00 (0.00 to 0.01) | 0.07 (0.00 to 0.14) | 0.14 (0.14 to 0.14) | 0.14 (0.06 to 0.27) |
| 3 | 43 | 0.01 (0.00 to 0.01) | 0.00 (0.00 to 0.13) | 0.15 (0.15 to 0.16) | 0.16 (0.03 to 0.38) |
| 4 | 43 | 0.01 (0.01 to 0.01) | 0.07 (0.00 to 0.16) | 0.17 (0.16 to 0.17) | 0.19 (0.03 to 0.44) |
| 5 | 43 | 0.02 (0.01 to 0.02) | 0.05 (0.00 to 0.19) | 0.19 (0.19 to 0.21) | 0.16 (0.00 to 0.26) |
| 6 | 43 | 0.03 (0.02 to 0.04) | 0.14 (0.05 to 0.28) | 0.22 (0.22 to 0.22) | 0.21 (0.07 to 0.34) |
| 7 | 43 | 0.04 (0.03 to 0.06) | 0.28 (0.08 to 0.39) | 0.23 (0.23 to 0.23) | 0.26 (0.12 to 0.39) |
| 8 | 43 | 0.08 (0.05 to 0.13) | 0.12 (0.02 to 0.27) | 0.25 (0.25 to 0.26) | 0.14 (0.02 to 0.28) |
| 9 | 43 | 0.28 (0.13 to 0.47) | 0.28 (0.05 to 0.65) | 0.26 (0.26 to 0.27) | 0.12 (0.05 to 0.33) |
| 10 | 43 | 0.80 (0.66 to 0.88) | 0.84 (0.57 to 0.95) | 0.29 (0.29 to 0.29) | 0.19 (0.04 to 0.29) |

Reliability for P(at least 30 minutes late) at T-15, 10 equal-count bins from the lowest predicted probability to the highest:

| Bin | Model n | Model predicted | Model observed | Historical predicted | Historical observed |
|---|---|---|---|---|---|
| 1 | 44 | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) | 0.02 (0.02 to 0.03) | 0.07 (0.00 to 0.15) |
| 2 | 44 | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) | 0.03 (0.03 to 0.04) | 0.00 (0.00 to 0.14) |
| 3 | 43 | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) | 0.04 (0.04 to 0.04) | 0.14 (0.00 to 0.21) |
| 4 | 43 | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) | 0.05 (0.05 to 0.05) | 0.00 (0.00 to 0.09) |
| 5 | 43 | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.04) | 0.06 (0.06 to 0.06) | 0.05 (0.00 to 0.09) |
| 6 | 43 | 0.00 (0.00 to 0.01) | 0.05 (0.00 to 0.11) | 0.07 (0.06 to 0.07) | 0.05 (0.00 to 0.14) |
| 7 | 43 | 0.01 (0.00 to 0.01) | 0.05 (0.00 to 0.17) | 0.08 (0.07 to 0.08) | 0.00 (0.00 to 0.08) |
| 8 | 43 | 0.01 (0.01 to 0.02) | 0.05 (0.00 to 0.14) | 0.08 (0.08 to 0.08) | 0.09 (0.00 to 0.27) |
| 9 | 43 | 0.02 (0.01 to 0.04) | 0.00 (0.00 to 0.09) | 0.09 (0.09 to 0.09) | 0.05 (0.00 to 0.12) |
| 10 | 43 | 0.22 (0.09 to 0.40) | 0.30 (0.08 to 0.57) | 0.11 (0.11 to 0.11) | 0.00 (0.00 to 0.03) |

Reliability for P(at least 60 minutes late) at T-15, 10 equal-count bins from the lowest predicted probability to the highest:

| Bin | Model n | Model predicted | Model observed | Historical predicted | Historical observed |
|---|---|---|---|---|---|
| 1 | 44 | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) |
| 2 | 44 | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) |
| 3 | 43 | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) | 0.01 (0.00 to 0.01) | 0.02 (0.00 to 0.08) |
| 4 | 43 | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) | 0.01 (0.01 to 0.01) | 0.00 (0.00 to 0.00) |
| 5 | 43 | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) | 0.01 (0.01 to 0.01) | 0.00 (0.00 to 0.00) |
| 6 | 43 | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) | 0.01 (0.01 to 0.01) | 0.00 (0.00 to 0.00) |
| 7 | 43 | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) | 0.01 (0.01 to 0.01) | 0.00 (0.00 to 0.00) |
| 8 | 43 | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) | 0.01 (0.01 to 0.01) | 0.00 (0.00 to 0.00) |
| 9 | 43 | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) | 0.02 (0.01 to 0.02) | 0.00 (0.00 to 0.00) |
| 10 | 43 | 0.01 (0.01 to 0.03) | 0.02 (0.00 to 0.08) | 0.02 (0.02 to 0.02) | 0.00 (0.00 to 0.00) |

Interval coverage by true delay band, descriptive only. A band is selected by the outcome that the interval was built before knowing, so it cannot reach the nominal level for any forecast that is not perfect.

| Method | 0-14 | 15-29 | 30-59 | 60-119 | 120+ |
|---|---|---|---|---|---|
| Timetable | n/a | n/a | n/a | n/a | n/a |
| Persistence | 0.85 (n=353, width 12.0) | 0.68 (n=60, width 12.0) | 0.28 (n=18, width 12.0) | 0.00 (n=1, width 12.0) | n/a |
| Historical | 0.94 (n=353, width 24.3) | 0.68 (n=60, width 23.6) | 0.00 (n=18, width 23.2) | 0.00 (n=1, width 31.7) | n/a |
| Model, global CQR | 0.85 (n=353, width 11.5) | 0.60 (n=60, width 12.3) | 0.56 (n=18, width 21.6) | 1.00 (n=1, width 40.3) | n/a |
| Model, Mondrian CQR | 0.86 (n=353, width 11.7) | 0.62 (n=60, width 14.1) | 0.56 (n=18, width 24.9) | 1.00 (n=1, width 45.7) | n/a |

## Limitations

These results cover one route, BHM to EUS. BHM to MAN and AST to EUS via BHM have not been ingested yet, so nothing here says the model transfers to them. The history runs from 2026-06-26 to 2026-10-14, which is summer and the first weeks of autumn, so there is no winter, leaf-fall or storm season, and a seasonal shift would change both the delays and the calibration. The 15 or more minutes late lateness group is small: it held 19 calibration rows at T-120, 20 calibration rows at T-60, 38 calibration rows at T-15 and 17 test rows at T-120, 25 test rows at T-60, 40 test rows at T-15, so its offsets are noisy and are merged into a neighbour where under 30 rows, and its coverage has a wide interval. The model's MAE gain over the bar is not statistically significant at T-15. The test block is 7 days, so the bootstrap intervals are wide, and the 60 minute and longer delays are too rare for the 60 minute threshold table to say much. The conformal guarantee holds on average within each observed lateness group if the days are exchangeable, which a time split only approximates. The delay bands are provisional until the per-operator compensation rules are checked.
