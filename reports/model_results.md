# Model results

MLflow run ccaafb52a93b46e2a8c463b18efb8c42, commit a41812638d8d3e3783f289431e6172c999c87a30. The model was trained on 98 complete days (2026-06-11 to 2026-09-19), tuned and calibrated on 14 validation days (2026-09-23 to 2026-10-06), the earlier half tuning and the later half calibrating, and scored once on 7 test days (2026-10-08 to 2026-10-14). Brackets are 95% bootstrap intervals over service dates. Intervals are nominal 80% intervals. Delay is minutes late at the route destination, and cancelled services are not scored.

The interval comes from LightGBM quantile models at 0.1, 0.5 and 0.9 with the horizon as a feature, widened by a conformal offset computed separately for each horizon and each observed lateness group at T (not started, 0 to 4, 5 to 14, and 15 or more minutes late). The lateness group uses only what is known at T, and a group with fewer than 30 calibration rows is merged into its neighbour. The target per horizon was chosen by tuning pinball loss between the direct delay and the delay minus the lateness at T: T-120: residual, T-60: direct, T-15: direct. Band probabilities come from a LightGBM band model with temperature calibration. P(at least 15) and P(at least 30) are then recalibrated separately per horizon, choosing among isotonic regression, Platt scaling and no recalibration by log loss on the tuning rows, with the methods fitted on the calibration rows. The choice was T-120 platt, T-60 none, T-15 none. The band probabilities are rebuilt from the recalibrated exceedances so that they stay monotone and sum to 1. The delay bands are provisional.

Two kinds of check are reported. The conformal guarantee is coverage conditional on the state known at T, so coverage by observed lateness is the interval check. For passengers and Delay Repay, what matters is whether the probabilities of reaching 15, 30 and 60 minutes late are calibrated, so threshold reliability is the passenger facing check. Coverage by true delay band is shown for description only.

## T-120 minutes, test block

| Method | MAE | 80% coverage | Mean width | Brier (5 bands) | Log loss |
|---|---|---|---|---|---|
| Timetable | 8.88 (7.59 to 10.14) | n/a | n/a | 0.366 (0.258 to 0.460) | 6.32 |
| Persistence | 7.61 (6.09 to 8.90) | 0.831 (0.774 to 0.889) | 22.0 | 0.380 (0.262 to 0.469) | 6.56 |
| Historical | 7.11 (5.87 to 8.29) | 0.856 (0.808 to 0.907) | 22.9 | 0.321 (0.243 to 0.390) | 0.62 |
| Model, global CQR | 6.21 (5.04 to 7.26) | 0.787 (0.724 to 0.857) | 18.8 | 0.316 (0.221 to 0.393) | 0.75 |
| Model, Mondrian CQR | 6.21 (5.04 to 7.26) | 0.789 (0.736 to 0.859) | 19.3 | 0.316 (0.221 to 0.393) | 0.75 |

Against the bar, historical (lowest validation MAE), the model's MAE is -0.90 minutes (-1.83 to +0.02, paired bootstrap over service dates). The interval includes zero, so the difference is not statistically significant.

Interval coverage by observed lateness at T, which is what the conformal guarantee covers:

| Method | not_started | 0-4 | 5-14 | 15+ |
|---|---|---|---|---|
| Timetable | n/a | n/a | n/a | n/a |
| Persistence | 0.85 (n=147, width 22.0) | 0.88 (n=213, width 22.0) | 0.75 (n=55, width 22.0) | 0.35 (n=17, width 22.0) |
| Historical | 0.86 (n=147, width 22.0) | 0.86 (n=213, width 22.9) | 0.91 (n=55, width 24.1) | 0.53 (n=17, width 26.0) |
| Model, global CQR | 0.83 (n=147, width 21.7) | 0.75 (n=213, width 14.4) | 0.89 (n=55, width 20.5) | 0.59 (n=17, width 42.5) |
| Model, Mondrian CQR | 0.78 (n=147, width 20.5) | 0.74 (n=213, width 14.3) | 0.96 (n=55, width 26.5) | 0.88 (n=17, width 48.5) |

Threshold reliability, derived from the band probabilities. Brier is the binary Brier score and ECE the expected calibration error over 10 equal-count bins. Historical band frequencies are the bar.

| Threshold | Share of rows at or above | Method | Brier | ECE |
|---|---|---|---|---|
| 15 minutes | 0.183 | Model, after threshold calibration | 0.148 (0.101 to 0.186) | 0.083 (0.052 to 0.137) |
| 15 minutes | 0.183 | Model, before threshold calibration | 0.148 (0.101 to 0.189) | 0.089 (0.048 to 0.139) |
| 15 minutes | 0.183 | Historical | 0.155 (0.120 to 0.187) | 0.095 (0.051 to 0.155) |
| 30 minutes | 0.044 | Model, after threshold calibration | 0.039 (0.014 to 0.063) | 0.032 (0.016 to 0.060) |
| 30 minutes | 0.044 | Model, before threshold calibration | 0.034 (0.014 to 0.054) | 0.032 (0.016 to 0.055) |
| 30 minutes | 0.044 | Historical | 0.044 (0.018 to 0.070) | 0.040 (0.036 to 0.076) |
| 60 minutes | 0.002 | Model, after threshold calibration | 0.002 (0.000 to 0.008) | 0.001 (0.001 to 0.007) |
| 60 minutes | 0.002 | Model, before threshold calibration | 0.003 (0.000 to 0.009) | 0.002 (0.001 to 0.007) |
| 60 minutes | 0.002 | Historical | 0.002 (0.000 to 0.008) | 0.010 (0.009 to 0.016) |

Reliability for P(at least 15 minutes late) at T-120, 10 equal-count bins from the lowest predicted probability to the highest:

| Bin | Model n | Model predicted | Model observed | Historical predicted | Historical observed |
|---|---|---|---|---|---|
| 1 | 44 | 0.02 (0.02 to 0.03) | 0.09 (0.00 to 0.21) | 0.10 (0.10 to 0.11) | 0.27 (0.05 to 0.45) |
| 2 | 44 | 0.03 (0.03 to 0.03) | 0.16 (0.00 to 0.24) | 0.13 (0.13 to 0.13) | 0.09 (0.03 to 0.36) |
| 3 | 43 | 0.04 (0.03 to 0.04) | 0.07 (0.00 to 0.21) | 0.14 (0.14 to 0.14) | 0.30 (0.07 to 0.46) |
| 4 | 43 | 0.05 (0.04 to 0.06) | 0.05 (0.00 to 0.19) | 0.15 (0.15 to 0.16) | 0.09 (0.03 to 0.24) |
| 5 | 43 | 0.07 (0.05 to 0.08) | 0.28 (0.05 to 0.37) | 0.19 (0.18 to 0.20) | 0.16 (0.00 to 0.28) |
| 6 | 43 | 0.08 (0.07 to 0.10) | 0.19 (0.00 to 0.35) | 0.21 (0.21 to 0.21) | 0.16 (0.07 to 0.30) |
| 7 | 43 | 0.10 (0.08 to 0.12) | 0.14 (0.05 to 0.36) | 0.22 (0.22 to 0.22) | 0.30 (0.10 to 0.43) |
| 8 | 43 | 0.12 (0.11 to 0.14) | 0.19 (0.06 to 0.44) | 0.24 (0.24 to 0.25) | 0.12 (0.02 to 0.30) |
| 9 | 43 | 0.15 (0.13 to 0.18) | 0.30 (0.10 to 0.40) | 0.26 (0.26 to 0.26) | 0.16 (0.03 to 0.33) |
| 10 | 43 | 0.34 (0.25 to 0.43) | 0.37 (0.17 to 0.65) | 0.28 (0.28 to 0.28) | 0.16 (0.06 to 0.23) |

Reliability for P(at least 30 minutes late) at T-120, 10 equal-count bins from the lowest predicted probability to the highest:

| Bin | Model n | Model predicted | Model observed | Historical predicted | Historical observed |
|---|---|---|---|---|---|
| 1 | 44 | 0.01 (0.01 to 0.01) | 0.00 (0.00 to 0.00) | 0.02 (0.02 to 0.02) | 0.02 (0.00 to 0.09) |
| 2 | 44 | 0.01 (0.01 to 0.01) | 0.02 (0.00 to 0.06) | 0.03 (0.03 to 0.03) | 0.09 (0.00 to 0.25) |
| 3 | 43 | 0.01 (0.01 to 0.01) | 0.00 (0.00 to 0.05) | 0.04 (0.04 to 0.04) | 0.09 (0.00 to 0.16) |
| 4 | 43 | 0.01 (0.01 to 0.02) | 0.00 (0.00 to 0.02) | 0.05 (0.05 to 0.05) | 0.05 (0.00 to 0.14) |
| 5 | 43 | 0.02 (0.02 to 0.02) | 0.05 (0.00 to 0.09) | 0.06 (0.06 to 0.06) | 0.00 (0.00 to 0.05) |
| 6 | 43 | 0.02 (0.02 to 0.03) | 0.09 (0.00 to 0.15) | 0.06 (0.06 to 0.06) | 0.05 (0.00 to 0.13) |
| 7 | 43 | 0.03 (0.02 to 0.03) | 0.05 (0.00 to 0.19) | 0.07 (0.07 to 0.07) | 0.07 (0.00 to 0.26) |
| 8 | 43 | 0.03 (0.03 to 0.04) | 0.07 (0.00 to 0.14) | 0.08 (0.08 to 0.08) | 0.02 (0.00 to 0.08) |
| 9 | 43 | 0.04 (0.03 to 0.04) | 0.00 (0.00 to 0.07) | 0.09 (0.09 to 0.09) | 0.05 (0.00 to 0.09) |
| 10 | 43 | 0.08 (0.06 to 0.11) | 0.16 (0.02 to 0.38) | 0.10 (0.10 to 0.11) | 0.00 (0.00 to 0.03) |

Reliability for P(at least 60 minutes late) at T-120, 10 equal-count bins from the lowest predicted probability to the highest:

| Bin | Model n | Model predicted | Model observed | Historical predicted | Historical observed |
|---|---|---|---|---|---|
| 1 | 44 | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) |
| 2 | 44 | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) |
| 3 | 43 | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) | 0.01 (0.01 to 0.01) | 0.02 (0.00 to 0.08) |
| 4 | 43 | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) | 0.01 (0.01 to 0.01) | 0.00 (0.00 to 0.00) |
| 5 | 43 | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) | 0.01 (0.01 to 0.01) | 0.00 (0.00 to 0.00) |
| 6 | 43 | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) | 0.01 (0.01 to 0.01) | 0.00 (0.00 to 0.00) |
| 7 | 43 | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) | 0.01 (0.01 to 0.01) | 0.00 (0.00 to 0.00) |
| 8 | 43 | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) | 0.01 (0.01 to 0.01) | 0.00 (0.00 to 0.00) |
| 9 | 43 | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) | 0.01 (0.01 to 0.01) | 0.00 (0.00 to 0.00) |
| 10 | 43 | 0.03 (0.01 to 0.04) | 0.02 (0.00 to 0.08) | 0.02 (0.02 to 0.02) | 0.00 (0.00 to 0.00) |

Interval coverage by true delay band, descriptive only. A band is selected by the outcome that the interval was built before knowing, so it cannot reach the nominal level for any forecast that is not perfect.

| Method | 0-14 | 15-29 | 30-59 | 60-119 | 120+ |
|---|---|---|---|---|---|
| Timetable | n/a | n/a | n/a | n/a | n/a |
| Persistence | 0.90 (n=353, width 22.0) | 0.63 (n=60, width 22.0) | 0.22 (n=18, width 22.0) | 1.00 (n=1, width 22.0) | n/a |
| Historical | 0.93 (n=353, width 23.0) | 0.68 (n=60, width 22.4) | 0.00 (n=18, width 21.5) | 0.00 (n=1, width 32.7) | n/a |
| Model, global CQR | 0.86 (n=353, width 17.9) | 0.50 (n=60, width 19.7) | 0.33 (n=18, width 30.3) | 0.00 (n=1, width 62.4) | n/a |
| Model, Mondrian CQR | 0.87 (n=353, width 18.3) | 0.45 (n=60, width 20.3) | 0.39 (n=18, width 32.6) | 1.00 (n=1, width 68.4) | n/a |

## T-60 minutes, test block

| Method | MAE | 80% coverage | Mean width | Brier (5 bands) | Log loss |
|---|---|---|---|---|---|
| Timetable | 8.88 (7.59 to 10.14) | n/a | n/a | 0.366 (0.258 to 0.460) | 6.32 |
| Persistence | 6.73 (5.41 to 7.92) | 0.752 (0.679 to 0.808) | 17.0 | 0.361 (0.261 to 0.433) | 6.24 |
| Historical | 7.11 (5.87 to 8.29) | 0.856 (0.808 to 0.907) | 22.9 | 0.321 (0.243 to 0.390) | 0.62 |
| Model, global CQR | 5.54 (4.61 to 6.51) | 0.782 (0.724 to 0.846) | 16.1 | 0.284 (0.206 to 0.353) | 0.58 |
| Model, Mondrian CQR | 5.54 (4.61 to 6.51) | 0.796 (0.744 to 0.855) | 16.4 | 0.284 (0.206 to 0.353) | 0.58 |

Against the bar, persistence (lowest validation MAE), the model's MAE is -1.19 minutes (-1.65 to -0.43, paired bootstrap over service dates). The interval excludes zero.

Interval coverage by observed lateness at T, which is what the conformal guarantee covers:

| Method | not_started | 0-4 | 5-14 | 15+ |
|---|---|---|---|---|
| Timetable | n/a | n/a | n/a | n/a |
| Persistence | 0.50 (n=2, width 17.0) | 0.81 (n=302, width 17.0) | 0.67 (n=103, width 17.0) | 0.36 (n=25, width 17.0) |
| Historical | 0.50 (n=2, width 19.8) | 0.85 (n=302, width 22.6) | 0.93 (n=103, width 23.5) | 0.64 (n=25, width 24.0) |
| Model, global CQR | 0.00 (n=2, width 35.7) | 0.76 (n=302, width 14.0) | 0.84 (n=103, width 17.7) | 0.84 (n=25, width 32.5) |
| Model, Mondrian CQR | 0.00 (n=2, width 35.7) | 0.76 (n=302, width 14.0) | 0.90 (n=103, width 18.9) | 0.84 (n=25, width 33.7) |

Threshold reliability, derived from the band probabilities. Brier is the binary Brier score and ECE the expected calibration error over 10 equal-count bins. Historical band frequencies are the bar.

| Threshold | Share of rows at or above | Method | Brier | ECE |
|---|---|---|---|---|
| 15 minutes | 0.183 | Model, after threshold calibration | 0.137 (0.097 to 0.173) | 0.092 (0.060 to 0.145) |
| 15 minutes | 0.183 | Model, before threshold calibration | 0.137 (0.097 to 0.173) | 0.092 (0.060 to 0.145) |
| 15 minutes | 0.183 | Historical | 0.155 (0.120 to 0.187) | 0.095 (0.052 to 0.155) |
| 30 minutes | 0.044 | Model, after threshold calibration | 0.032 (0.011 to 0.054) | 0.023 (0.010 to 0.055) |
| 30 minutes | 0.044 | Model, before threshold calibration | 0.032 (0.011 to 0.054) | 0.023 (0.010 to 0.055) |
| 30 minutes | 0.044 | Historical | 0.044 (0.018 to 0.070) | 0.040 (0.037 to 0.078) |
| 60 minutes | 0.002 | Model, after threshold calibration | 0.003 (0.000 to 0.009) | 0.001 (0.000 to 0.007) |
| 60 minutes | 0.002 | Model, before threshold calibration | 0.003 (0.000 to 0.009) | 0.001 (0.000 to 0.007) |
| 60 minutes | 0.002 | Historical | 0.002 (0.000 to 0.008) | 0.010 (0.009 to 0.016) |

Reliability for P(at least 15 minutes late) at T-60, 10 equal-count bins from the lowest predicted probability to the highest:

| Bin | Model n | Model predicted | Model observed | Historical predicted | Historical observed |
|---|---|---|---|---|---|
| 1 | 44 | 0.01 (0.01 to 0.01) | 0.07 (0.00 to 0.14) | 0.10 (0.10 to 0.11) | 0.27 (0.04 to 0.45) |
| 2 | 44 | 0.01 (0.01 to 0.01) | 0.09 (0.00 to 0.17) | 0.13 (0.13 to 0.13) | 0.09 (0.03 to 0.35) |
| 3 | 43 | 0.02 (0.02 to 0.02) | 0.02 (0.00 to 0.11) | 0.14 (0.14 to 0.14) | 0.30 (0.07 to 0.47) |
| 4 | 43 | 0.02 (0.02 to 0.03) | 0.14 (0.00 to 0.21) | 0.15 (0.15 to 0.16) | 0.09 (0.03 to 0.24) |
| 5 | 43 | 0.03 (0.03 to 0.04) | 0.26 (0.09 to 0.36) | 0.19 (0.18 to 0.20) | 0.14 (0.00 to 0.28) |
| 6 | 43 | 0.04 (0.04 to 0.06) | 0.12 (0.03 to 0.24) | 0.21 (0.21 to 0.21) | 0.19 (0.07 to 0.30) |
| 7 | 43 | 0.06 (0.05 to 0.10) | 0.21 (0.10 to 0.35) | 0.22 (0.22 to 0.22) | 0.30 (0.10 to 0.42) |
| 8 | 43 | 0.11 (0.07 to 0.18) | 0.14 (0.05 to 0.29) | 0.24 (0.24 to 0.25) | 0.12 (0.02 to 0.30) |
| 9 | 43 | 0.24 (0.13 to 0.44) | 0.21 (0.02 to 0.53) | 0.26 (0.26 to 0.26) | 0.16 (0.03 to 0.35) |
| 10 | 43 | 0.74 (0.58 to 0.84) | 0.58 (0.38 to 0.73) | 0.28 (0.28 to 0.28) | 0.16 (0.07 to 0.23) |

Reliability for P(at least 30 minutes late) at T-60, 10 equal-count bins from the lowest predicted probability to the highest:

| Bin | Model n | Model predicted | Model observed | Historical predicted | Historical observed |
|---|---|---|---|---|---|
| 1 | 44 | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) | 0.02 (0.02 to 0.02) | 0.02 (0.00 to 0.09) |
| 2 | 44 | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) | 0.03 (0.03 to 0.03) | 0.09 (0.00 to 0.25) |
| 3 | 43 | 0.00 (0.00 to 0.00) | 0.02 (0.00 to 0.07) | 0.04 (0.04 to 0.04) | 0.09 (0.00 to 0.16) |
| 4 | 43 | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.10) | 0.05 (0.05 to 0.05) | 0.05 (0.00 to 0.14) |
| 5 | 43 | 0.00 (0.00 to 0.01) | 0.07 (0.00 to 0.15) | 0.06 (0.06 to 0.06) | 0.00 (0.00 to 0.05) |
| 6 | 43 | 0.01 (0.00 to 0.01) | 0.05 (0.00 to 0.15) | 0.06 (0.06 to 0.06) | 0.05 (0.00 to 0.14) |
| 7 | 43 | 0.01 (0.01 to 0.02) | 0.02 (0.00 to 0.09) | 0.07 (0.07 to 0.07) | 0.07 (0.00 to 0.27) |
| 8 | 43 | 0.02 (0.01 to 0.03) | 0.02 (0.00 to 0.07) | 0.08 (0.08 to 0.08) | 0.02 (0.00 to 0.07) |
| 9 | 43 | 0.04 (0.02 to 0.08) | 0.00 (0.00 to 0.08) | 0.09 (0.09 to 0.09) | 0.05 (0.00 to 0.09) |
| 10 | 43 | 0.29 (0.15 to 0.50) | 0.26 (0.02 to 0.53) | 0.10 (0.10 to 0.11) | 0.00 (0.00 to 0.03) |

Reliability for P(at least 60 minutes late) at T-60, 10 equal-count bins from the lowest predicted probability to the highest:

| Bin | Model n | Model predicted | Model observed | Historical predicted | Historical observed |
|---|---|---|---|---|---|
| 1 | 44 | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) |
| 2 | 44 | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) |
| 3 | 43 | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) | 0.01 (0.01 to 0.01) | 0.02 (0.00 to 0.08) |
| 4 | 43 | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) | 0.01 (0.01 to 0.01) | 0.00 (0.00 to 0.00) |
| 5 | 43 | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) | 0.01 (0.01 to 0.01) | 0.00 (0.00 to 0.00) |
| 6 | 43 | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) | 0.01 (0.01 to 0.01) | 0.00 (0.00 to 0.00) |
| 7 | 43 | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) | 0.01 (0.01 to 0.01) | 0.00 (0.00 to 0.00) |
| 8 | 43 | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) | 0.01 (0.01 to 0.01) | 0.00 (0.00 to 0.00) |
| 9 | 43 | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) | 0.01 (0.01 to 0.01) | 0.00 (0.00 to 0.00) |
| 10 | 43 | 0.03 (0.01 to 0.06) | 0.02 (0.00 to 0.08) | 0.02 (0.02 to 0.02) | 0.00 (0.00 to 0.00) |

Interval coverage by true delay band, descriptive only. A band is selected by the outcome that the interval was built before knowing, so it cannot reach the nominal level for any forecast that is not perfect.

| Method | 0-14 | 15-29 | 30-59 | 60-119 | 120+ |
|---|---|---|---|---|---|
| Timetable | n/a | n/a | n/a | n/a | n/a |
| Persistence | 0.82 (n=353, width 17.0) | 0.50 (n=60, width 17.0) | 0.22 (n=18, width 17.0) | 0.00 (n=1, width 17.0) | n/a |
| Historical | 0.93 (n=353, width 23.0) | 0.68 (n=60, width 22.4) | 0.00 (n=18, width 21.5) | 0.00 (n=1, width 32.7) | n/a |
| Model, global CQR | 0.85 (n=353, width 15.1) | 0.50 (n=60, width 16.8) | 0.39 (n=18, width 30.4) | 0.00 (n=1, width 57.4) | n/a |
| Model, Mondrian CQR | 0.87 (n=353, width 15.4) | 0.52 (n=60, width 17.4) | 0.39 (n=18, width 31.1) | 0.00 (n=1, width 58.6) | n/a |

## T-15 minutes, test block

| Method | MAE | 80% coverage | Mean width | Brier (5 bands) | Log loss |
|---|---|---|---|---|---|
| Timetable | 8.88 (7.59 to 10.14) | n/a | n/a | 0.366 (0.258 to 0.460) | 6.32 |
| Persistence | 4.25 (3.20 to 5.23) | 0.803 (0.783 to 0.824) | 12.0 | 0.278 (0.188 to 0.351) | 4.80 |
| Historical | 7.11 (5.87 to 8.29) | 0.856 (0.808 to 0.907) | 22.9 | 0.321 (0.243 to 0.390) | 0.62 |
| Model, global CQR | 3.86 (3.07 to 4.80) | 0.780 (0.733 to 0.833) | 11.4 | 0.213 (0.149 to 0.270) | 0.44 |
| Model, Mondrian CQR | 3.86 (3.07 to 4.80) | 0.787 (0.743 to 0.834) | 11.3 | 0.213 (0.149 to 0.270) | 0.44 |

Against the bar, persistence (lowest validation MAE), the model's MAE is -0.40 minutes (-0.72 to +0.02, paired bootstrap over service dates). The interval includes zero, so the difference is not statistically significant.

Interval coverage by observed lateness at T, which is what the conformal guarantee covers:

| Method | not_started | 0-4 | 5-14 | 15+ |
|---|---|---|---|---|
| Timetable | n/a | n/a | n/a | n/a |
| Persistence | n/a | 0.79 (n=241, width 12.0) | 0.85 (n=151, width 12.0) | 0.70 (n=40, width 12.0) |
| Historical | n/a | 0.83 (n=241, width 22.0) | 0.98 (n=151, width 24.0) | 0.55 (n=40, width 23.6) |
| Model, global CQR | n/a | 0.74 (n=241, width 9.6) | 0.81 (n=151, width 11.0) | 0.90 (n=40, width 23.5) |
| Model, Mondrian CQR | n/a | 0.76 (n=241, width 9.9) | 0.81 (n=151, width 10.7) | 0.88 (n=40, width 21.9) |

Threshold reliability, derived from the band probabilities. Brier is the binary Brier score and ECE the expected calibration error over 10 equal-count bins. Historical band frequencies are the bar.

| Threshold | Share of rows at or above | Method | Brier | ECE |
|---|---|---|---|---|
| 15 minutes | 0.183 | Model, after threshold calibration | 0.099 (0.066 to 0.129) | 0.061 (0.039 to 0.103) |
| 15 minutes | 0.183 | Model, before threshold calibration | 0.099 (0.066 to 0.129) | 0.061 (0.039 to 0.103) |
| 15 minutes | 0.183 | Historical | 0.155 (0.120 to 0.187) | 0.095 (0.051 to 0.154) |
| 30 minutes | 0.044 | Model, after threshold calibration | 0.022 (0.004 to 0.043) | 0.017 (0.004 to 0.041) |
| 30 minutes | 0.044 | Model, before threshold calibration | 0.022 (0.004 to 0.043) | 0.017 (0.004 to 0.041) |
| 30 minutes | 0.044 | Historical | 0.044 (0.018 to 0.070) | 0.040 (0.037 to 0.077) |
| 60 minutes | 0.002 | Model, after threshold calibration | 0.003 (0.000 to 0.008) | 0.001 (0.000 to 0.006) |
| 60 minutes | 0.002 | Model, before threshold calibration | 0.003 (0.000 to 0.008) | 0.001 (0.000 to 0.006) |
| 60 minutes | 0.002 | Historical | 0.002 (0.000 to 0.008) | 0.010 (0.009 to 0.016) |

Reliability for P(at least 15 minutes late) at T-15, 10 equal-count bins from the lowest predicted probability to the highest:

| Bin | Model n | Model predicted | Model observed | Historical predicted | Historical observed |
|---|---|---|---|---|---|
| 1 | 44 | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) | 0.10 (0.10 to 0.11) | 0.27 (0.05 to 0.45) |
| 2 | 44 | 0.00 (0.00 to 0.01) | 0.00 (0.00 to 0.09) | 0.13 (0.13 to 0.13) | 0.09 (0.03 to 0.35) |
| 3 | 43 | 0.01 (0.00 to 0.01) | 0.07 (0.00 to 0.09) | 0.14 (0.14 to 0.14) | 0.30 (0.07 to 0.47) |
| 4 | 43 | 0.01 (0.01 to 0.02) | 0.02 (0.00 to 0.12) | 0.15 (0.15 to 0.16) | 0.09 (0.03 to 0.24) |
| 5 | 43 | 0.02 (0.01 to 0.02) | 0.09 (0.02 to 0.20) | 0.19 (0.18 to 0.20) | 0.16 (0.00 to 0.27) |
| 6 | 43 | 0.03 (0.02 to 0.04) | 0.19 (0.09 to 0.28) | 0.21 (0.21 to 0.21) | 0.16 (0.07 to 0.30) |
| 7 | 43 | 0.05 (0.03 to 0.07) | 0.23 (0.10 to 0.37) | 0.22 (0.22 to 0.22) | 0.30 (0.10 to 0.42) |
| 8 | 43 | 0.09 (0.06 to 0.16) | 0.12 (0.02 to 0.19) | 0.24 (0.24 to 0.25) | 0.12 (0.02 to 0.30) |
| 9 | 43 | 0.33 (0.16 to 0.54) | 0.30 (0.03 to 0.65) | 0.26 (0.26 to 0.26) | 0.14 (0.03 to 0.33) |
| 10 | 43 | 0.88 (0.75 to 0.95) | 0.81 (0.59 to 0.95) | 0.28 (0.28 to 0.28) | 0.19 (0.06 to 0.24) |

Reliability for P(at least 30 minutes late) at T-15, 10 equal-count bins from the lowest predicted probability to the highest:

| Bin | Model n | Model predicted | Model observed | Historical predicted | Historical observed |
|---|---|---|---|---|---|
| 1 | 44 | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) | 0.02 (0.02 to 0.02) | 0.02 (0.00 to 0.10) |
| 2 | 44 | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) | 0.03 (0.03 to 0.03) | 0.09 (0.00 to 0.25) |
| 3 | 43 | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) | 0.04 (0.04 to 0.04) | 0.09 (0.00 to 0.16) |
| 4 | 43 | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) | 0.05 (0.05 to 0.05) | 0.05 (0.00 to 0.13) |
| 5 | 43 | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.06) | 0.06 (0.06 to 0.06) | 0.00 (0.00 to 0.05) |
| 6 | 43 | 0.00 (0.00 to 0.00) | 0.02 (0.00 to 0.07) | 0.06 (0.06 to 0.06) | 0.05 (0.00 to 0.13) |
| 7 | 43 | 0.00 (0.00 to 0.01) | 0.07 (0.00 to 0.20) | 0.07 (0.07 to 0.07) | 0.07 (0.00 to 0.27) |
| 8 | 43 | 0.01 (0.01 to 0.01) | 0.05 (0.00 to 0.14) | 0.08 (0.08 to 0.08) | 0.02 (0.00 to 0.07) |
| 9 | 43 | 0.02 (0.01 to 0.05) | 0.00 (0.00 to 0.09) | 0.09 (0.09 to 0.09) | 0.05 (0.00 to 0.10) |
| 10 | 43 | 0.32 (0.15 to 0.55) | 0.30 (0.08 to 0.57) | 0.10 (0.10 to 0.11) | 0.00 (0.00 to 0.03) |

Reliability for P(at least 60 minutes late) at T-15, 10 equal-count bins from the lowest predicted probability to the highest:

| Bin | Model n | Model predicted | Model observed | Historical predicted | Historical observed |
|---|---|---|---|---|---|
| 1 | 44 | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) |
| 2 | 44 | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) |
| 3 | 43 | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) | 0.01 (0.01 to 0.01) | 0.02 (0.00 to 0.08) |
| 4 | 43 | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) | 0.01 (0.01 to 0.01) | 0.00 (0.00 to 0.00) |
| 5 | 43 | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) | 0.01 (0.01 to 0.01) | 0.00 (0.00 to 0.00) |
| 6 | 43 | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) | 0.01 (0.01 to 0.01) | 0.00 (0.00 to 0.00) |
| 7 | 43 | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) | 0.01 (0.01 to 0.01) | 0.00 (0.00 to 0.00) |
| 8 | 43 | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) | 0.01 (0.01 to 0.01) | 0.00 (0.00 to 0.00) |
| 9 | 43 | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) | 0.01 (0.01 to 0.01) | 0.00 (0.00 to 0.00) |
| 10 | 43 | 0.03 (0.01 to 0.05) | 0.02 (0.00 to 0.08) | 0.02 (0.02 to 0.02) | 0.00 (0.00 to 0.00) |

Interval coverage by true delay band, descriptive only. A band is selected by the outcome that the interval was built before knowing, so it cannot reach the nominal level for any forecast that is not perfect.

| Method | 0-14 | 15-29 | 30-59 | 60-119 | 120+ |
|---|---|---|---|---|---|
| Timetable | n/a | n/a | n/a | n/a | n/a |
| Persistence | 0.85 (n=353, width 12.0) | 0.68 (n=60, width 12.0) | 0.28 (n=18, width 12.0) | 0.00 (n=1, width 12.0) | n/a |
| Historical | 0.93 (n=353, width 23.0) | 0.68 (n=60, width 22.4) | 0.00 (n=18, width 21.5) | 0.00 (n=1, width 32.7) | n/a |
| Model, global CQR | 0.82 (n=353, width 10.1) | 0.65 (n=60, width 13.3) | 0.56 (n=18, width 29.3) | 0.00 (n=1, width 53.6) | n/a |
| Model, Mondrian CQR | 0.82 (n=353, width 10.1) | 0.65 (n=60, width 12.7) | 0.56 (n=18, width 28.4) | 0.00 (n=1, width 51.9) | n/a |

## Limitations

These results cover one route, BHM to EUS. BHM to MAN and AST to EUS via BHM have not been ingested yet, so nothing here says the model transfers to them. The history runs from 2026-06-11 to 2026-10-14, which is summer and the first weeks of autumn, so there is no winter, leaf-fall or storm season, and a seasonal shift would change both the delays and the calibration. The 15 or more minutes late lateness group is small: it held 19 calibration rows at T-120, 20 calibration rows at T-60, 38 calibration rows at T-15 and 17 test rows at T-120, 25 test rows at T-60, 40 test rows at T-15, so its offsets are noisy and are merged into a neighbour where under 30 rows, and its coverage has a wide interval. The model's MAE gain over the bar is not statistically significant at T-120, T-15. The test block is 7 days, so the bootstrap intervals are wide, and the 60 minute and longer delays are too rare for the 60 minute threshold table to say much. The conformal guarantee holds on average within each observed lateness group if the days are exchangeable, which a time split only approximates. The delay bands are provisional until the per-operator compensation rules are checked.
