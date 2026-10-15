# Model results

MLflow run 4105b5f763b547a7b008aebf5be7cbe6, commit 1ca88eac99bee45ef00e59811b774662540a5516. The model was trained on 70 complete days (2026-06-30 to 2026-09-18), tuned and calibrated on 14 validation days (2026-09-23 to 2026-10-06), the earlier half tuning and the later half calibrating, and scored once on 7 test days (2026-10-08 to 2026-10-14). Brackets are 95% bootstrap intervals over service dates. Intervals are nominal 80% intervals. Delay is minutes late at the route destination, and cancelled services are not scored.

The interval comes from LightGBM quantile models at 0.1, 0.5 and 0.9 with the horizon as a feature, widened by a conformal offset computed separately for each horizon and each observed lateness group at T (not started, 0 to 4, 5 to 14, and 15 or more minutes late). The lateness group uses only what is known at T, and a group with fewer than 30 calibration rows is merged into its neighbour. The target per horizon was chosen by tuning pinball loss between the direct delay and the delay minus the lateness at T: T-120: direct, T-60: residual, T-15: residual. Band probabilities come from a LightGBM band model with temperature calibration. The delay bands are provisional.

Two kinds of check are reported. The conformal guarantee is coverage conditional on the state known at T, so coverage by observed lateness is the interval check. For passengers and Delay Repay, what matters is whether the probabilities of reaching 15, 30 and 60 minutes late are calibrated, so threshold reliability is the passenger facing check. Coverage by true delay band is shown for description only.

## T-120 minutes, test block

| Method | MAE | 80% coverage | Mean width | Brier (5 bands) | Log loss |
|---|---|---|---|---|---|
| Timetable | 8.88 (7.59 to 10.14) | n/a | n/a | 0.366 (0.258 to 0.460) | 6.32 |
| Persistence | 7.61 (6.09 to 8.90) | 0.838 (0.782 to 0.892) | 23.0 | 0.380 (0.262 to 0.469) | 6.56 |
| Historical | 7.14 (5.90 to 8.32) | 0.859 (0.815 to 0.905) | 23.9 | 0.320 (0.245 to 0.387) | 0.62 |
| Model, global CQR | 6.32 (5.24 to 7.30) | 0.801 (0.731 to 0.877) | 20.2 | 0.306 (0.213 to 0.383) | 0.60 |
| Model, Mondrian CQR | 6.32 (5.24 to 7.30) | 0.806 (0.745 to 0.872) | 20.6 | 0.306 (0.213 to 0.383) | 0.60 |

Against the bar, historical (lowest validation MAE), the model's MAE is -0.83 minutes (-1.51 to -0.20, paired bootstrap over service dates). The interval excludes zero.

Interval coverage by observed lateness at T, which is what the conformal guarantee covers:

| Method | not_started | 0-4 | 5-14 | 15+ |
|---|---|---|---|---|
| Timetable | n/a | n/a | n/a | n/a |
| Persistence | 0.86 (n=147, width 23.0) | 0.88 (n=213, width 23.0) | 0.75 (n=55, width 23.0) | 0.35 (n=17, width 23.0) |
| Historical | 0.86 (n=147, width 22.9) | 0.86 (n=213, width 23.8) | 0.96 (n=55, width 25.6) | 0.53 (n=17, width 27.0) |
| Model, global CQR | 0.83 (n=147, width 22.7) | 0.76 (n=213, width 16.1) | 0.89 (n=55, width 21.1) | 0.82 (n=17, width 47.8) |
| Model, Mondrian CQR | 0.82 (n=147, width 22.2) | 0.76 (n=213, width 15.9) | 0.95 (n=55, width 24.7) | 0.88 (n=17, width 51.5) |

Threshold reliability, derived from the band probabilities. Brier is the binary Brier score and ECE the expected calibration error over 10 equal-count bins. Historical band frequencies are the bar.

| Threshold | Share of rows at or above | Method | Brier | ECE |
|---|---|---|---|---|
| 15 minutes | 0.183 | Model | 0.146 (0.100 to 0.185) | 0.080 (0.048 to 0.133) |
| 15 minutes | 0.183 | Historical | 0.155 (0.121 to 0.184) | 0.058 (0.052 to 0.152) |
| 30 minutes | 0.044 | Model | 0.033 (0.011 to 0.054) | 0.020 (0.010 to 0.051) |
| 30 minutes | 0.044 | Historical | 0.044 (0.018 to 0.070) | 0.054 (0.038 to 0.072) |
| 60 minutes | 0.002 | Model | 0.003 (0.000 to 0.008) | 0.001 (0.001 to 0.007) |
| 60 minutes | 0.002 | Historical | 0.002 (0.000 to 0.008) | 0.012 (0.010 to 0.018) |

Reliability for P(at least 15 minutes late) at T-120, 10 equal-count bins from the lowest predicted probability to the highest:

| Bin | Model n | Model predicted | Model observed | Historical predicted | Historical observed |
|---|---|---|---|---|---|
| 1 | 44 | 0.02 (0.01 to 0.02) | 0.16 (0.02 to 0.24) | 0.10 (0.10 to 0.11) | 0.25 (0.09 to 0.50) |
| 2 | 44 | 0.02 (0.02 to 0.02) | 0.11 (0.02 to 0.29) | 0.13 (0.13 to 0.14) | 0.16 (0.05 to 0.24) |
| 3 | 43 | 0.03 (0.02 to 0.03) | 0.09 (0.00 to 0.17) | 0.16 (0.16 to 0.16) | 0.16 (0.03 to 0.38) |
| 4 | 43 | 0.04 (0.03 to 0.05) | 0.07 (0.00 to 0.17) | 0.17 (0.17 to 0.17) | 0.19 (0.03 to 0.44) |
| 5 | 43 | 0.06 (0.04 to 0.08) | 0.16 (0.02 to 0.30) | 0.20 (0.19 to 0.21) | 0.16 (0.00 to 0.28) |
| 6 | 43 | 0.09 (0.06 to 0.12) | 0.21 (0.03 to 0.39) | 0.22 (0.22 to 0.22) | 0.23 (0.10 to 0.43) |
| 7 | 43 | 0.12 (0.09 to 0.15) | 0.19 (0.05 to 0.41) | 0.22 (0.22 to 0.23) | 0.21 (0.06 to 0.35) |
| 8 | 43 | 0.15 (0.12 to 0.17) | 0.12 (0.04 to 0.47) | 0.25 (0.24 to 0.25) | 0.12 (0.03 to 0.26) |
| 9 | 43 | 0.18 (0.15 to 0.21) | 0.30 (0.07 to 0.42) | 0.26 (0.26 to 0.27) | 0.16 (0.07 to 0.28) |
| 10 | 43 | 0.43 (0.30 to 0.57) | 0.42 (0.17 to 0.65) | 0.29 (0.29 to 0.29) | 0.19 (0.04 to 0.29) |

Reliability for P(at least 30 minutes late) at T-120, 10 equal-count bins from the lowest predicted probability to the highest:

| Bin | Model n | Model predicted | Model observed | Historical predicted | Historical observed |
|---|---|---|---|---|---|
| 1 | 44 | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) | 0.02 (0.02 to 0.03) | 0.07 (0.00 to 0.15) |
| 2 | 44 | 0.00 (0.00 to 0.00) | 0.02 (0.00 to 0.05) | 0.04 (0.04 to 0.04) | 0.00 (0.00 to 0.14) |
| 3 | 43 | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.06) | 0.04 (0.04 to 0.04) | 0.14 (0.00 to 0.21) |
| 4 | 43 | 0.00 (0.00 to 0.01) | 0.02 (0.00 to 0.06) | 0.05 (0.05 to 0.05) | 0.00 (0.00 to 0.10) |
| 5 | 43 | 0.01 (0.00 to 0.01) | 0.02 (0.00 to 0.11) | 0.06 (0.06 to 0.06) | 0.09 (0.00 to 0.17) |
| 6 | 43 | 0.01 (0.01 to 0.02) | 0.09 (0.00 to 0.21) | 0.07 (0.07 to 0.07) | 0.00 (0.00 to 0.02) |
| 7 | 43 | 0.02 (0.01 to 0.02) | 0.05 (0.00 to 0.17) | 0.07 (0.07 to 0.08) | 0.02 (0.00 to 0.16) |
| 8 | 43 | 0.02 (0.02 to 0.03) | 0.02 (0.00 to 0.09) | 0.08 (0.08 to 0.08) | 0.07 (0.00 to 0.22) |
| 9 | 43 | 0.03 (0.02 to 0.04) | 0.02 (0.00 to 0.07) | 0.09 (0.09 to 0.09) | 0.02 (0.00 to 0.06) |
| 10 | 43 | 0.17 (0.09 to 0.26) | 0.19 (0.02 to 0.44) | 0.11 (0.11 to 0.11) | 0.02 (0.00 to 0.08) |

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
| 8 | 43 | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) | 0.02 (0.02 to 0.02) | 0.00 (0.00 to 0.00) |
| 9 | 43 | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) | 0.02 (0.02 to 0.02) | 0.00 (0.00 to 0.00) |
| 10 | 43 | 0.03 (0.01 to 0.06) | 0.02 (0.00 to 0.08) | 0.02 (0.02 to 0.02) | 0.00 (0.00 to 0.00) |

Interval coverage by true delay band, descriptive only. A band is selected by the outcome that the interval was built before knowing, so it cannot reach the nominal level for any forecast that is not perfect.

| Method | 0-14 | 15-29 | 30-59 | 60-119 | 120+ |
|---|---|---|---|---|---|
| Timetable | n/a | n/a | n/a | n/a | n/a |
| Persistence | 0.90 (n=353, width 23.0) | 0.68 (n=60, width 23.0) | 0.22 (n=18, width 23.0) | 1.00 (n=1, width 23.0) | n/a |
| Historical | 0.93 (n=353, width 24.0) | 0.72 (n=60, width 23.2) | 0.00 (n=18, width 22.5) | 0.00 (n=1, width 43.0) | n/a |
| Model, global CQR | 0.88 (n=353, width 19.4) | 0.47 (n=60, width 20.3) | 0.33 (n=18, width 33.9) | 0.00 (n=1, width 60.6) | n/a |
| Model, Mondrian CQR | 0.88 (n=353, width 19.6) | 0.50 (n=60, width 20.8) | 0.39 (n=18, width 35.4) | 0.00 (n=1, width 64.2) | n/a |

## T-60 minutes, test block

| Method | MAE | 80% coverage | Mean width | Brier (5 bands) | Log loss |
|---|---|---|---|---|---|
| Timetable | 8.88 (7.59 to 10.14) | n/a | n/a | 0.366 (0.258 to 0.460) | 6.32 |
| Persistence | 6.73 (5.41 to 7.92) | 0.759 (0.685 to 0.814) | 18.0 | 0.361 (0.261 to 0.433) | 6.24 |
| Historical | 7.14 (5.90 to 8.32) | 0.859 (0.815 to 0.905) | 23.9 | 0.320 (0.245 to 0.387) | 0.62 |
| Model, global CQR | 5.70 (4.78 to 6.54) | 0.771 (0.723 to 0.830) | 16.2 | 0.283 (0.206 to 0.348) | 0.58 |
| Model, Mondrian CQR | 5.70 (4.78 to 6.54) | 0.789 (0.742 to 0.840) | 17.0 | 0.283 (0.206 to 0.348) | 0.58 |

Against the bar, persistence (lowest validation MAE), the model's MAE is -1.03 minutes (-1.50 to -0.22, paired bootstrap over service dates). The interval excludes zero.

Interval coverage by observed lateness at T, which is what the conformal guarantee covers:

| Method | not_started | 0-4 | 5-14 | 15+ |
|---|---|---|---|---|
| Timetable | n/a | n/a | n/a | n/a |
| Persistence | 0.50 (n=2, width 18.0) | 0.82 (n=302, width 18.0) | 0.67 (n=103, width 18.0) | 0.36 (n=25, width 18.0) |
| Historical | 0.50 (n=2, width 19.2) | 0.86 (n=302, width 23.7) | 0.92 (n=103, width 24.2) | 0.60 (n=25, width 24.7) |
| Model, global CQR | 0.00 (n=2, width 36.9) | 0.77 (n=302, width 14.6) | 0.82 (n=103, width 17.4) | 0.64 (n=25, width 30.1) |
| Model, Mondrian CQR | 0.00 (n=2, width 36.6) | 0.75 (n=302, width 14.3) | 0.93 (n=103, width 20.7) | 0.68 (n=25, width 33.5) |

Threshold reliability, derived from the band probabilities. Brier is the binary Brier score and ECE the expected calibration error over 10 equal-count bins. Historical band frequencies are the bar.

| Threshold | Share of rows at or above | Method | Brier | ECE |
|---|---|---|---|---|
| 15 minutes | 0.183 | Model | 0.135 (0.097 to 0.170) | 0.088 (0.054 to 0.131) |
| 15 minutes | 0.183 | Historical | 0.155 (0.121 to 0.184) | 0.063 (0.053 to 0.152) |
| 30 minutes | 0.044 | Model | 0.030 (0.008 to 0.052) | 0.018 (0.008 to 0.051) |
| 30 minutes | 0.044 | Historical | 0.044 (0.018 to 0.070) | 0.054 (0.038 to 0.073) |
| 60 minutes | 0.002 | Model | 0.003 (0.000 to 0.008) | 0.001 (0.000 to 0.006) |
| 60 minutes | 0.002 | Historical | 0.002 (0.000 to 0.008) | 0.012 (0.010 to 0.018) |

Reliability for P(at least 15 minutes late) at T-60, 10 equal-count bins from the lowest predicted probability to the highest:

| Bin | Model n | Model predicted | Model observed | Historical predicted | Historical observed |
|---|---|---|---|---|---|
| 1 | 44 | 0.01 (0.01 to 0.01) | 0.09 (0.00 to 0.19) | 0.10 (0.10 to 0.11) | 0.25 (0.08 to 0.50) |
| 2 | 44 | 0.01 (0.01 to 0.02) | 0.07 (0.00 to 0.17) | 0.13 (0.13 to 0.14) | 0.16 (0.05 to 0.24) |
| 3 | 43 | 0.02 (0.02 to 0.02) | 0.07 (0.00 to 0.15) | 0.16 (0.16 to 0.16) | 0.16 (0.03 to 0.38) |
| 4 | 43 | 0.03 (0.02 to 0.03) | 0.07 (0.00 to 0.20) | 0.17 (0.17 to 0.17) | 0.19 (0.03 to 0.43) |
| 5 | 43 | 0.04 (0.03 to 0.05) | 0.26 (0.05 to 0.30) | 0.20 (0.19 to 0.21) | 0.14 (0.00 to 0.28) |
| 6 | 43 | 0.05 (0.04 to 0.07) | 0.14 (0.05 to 0.25) | 0.22 (0.22 to 0.22) | 0.26 (0.10 to 0.43) |
| 7 | 43 | 0.07 (0.06 to 0.11) | 0.16 (0.05 to 0.33) | 0.22 (0.22 to 0.23) | 0.21 (0.05 to 0.35) |
| 8 | 43 | 0.11 (0.08 to 0.17) | 0.19 (0.05 to 0.33) | 0.25 (0.24 to 0.25) | 0.12 (0.03 to 0.26) |
| 9 | 43 | 0.25 (0.13 to 0.47) | 0.19 (0.03 to 0.54) | 0.26 (0.26 to 0.27) | 0.16 (0.06 to 0.28) |
| 10 | 43 | 0.73 (0.56 to 0.84) | 0.60 (0.39 to 0.72) | 0.29 (0.29 to 0.29) | 0.19 (0.04 to 0.29) |

Reliability for P(at least 30 minutes late) at T-60, 10 equal-count bins from the lowest predicted probability to the highest:

| Bin | Model n | Model predicted | Model observed | Historical predicted | Historical observed |
|---|---|---|---|---|---|
| 1 | 44 | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.02) | 0.02 (0.02 to 0.03) | 0.07 (0.00 to 0.15) |
| 2 | 44 | 0.00 (0.00 to 0.00) | 0.02 (0.00 to 0.07) | 0.04 (0.04 to 0.04) | 0.00 (0.00 to 0.14) |
| 3 | 43 | 0.00 (0.00 to 0.00) | 0.02 (0.00 to 0.07) | 0.04 (0.04 to 0.04) | 0.14 (0.00 to 0.23) |
| 4 | 43 | 0.00 (0.00 to 0.00) | 0.05 (0.00 to 0.12) | 0.05 (0.05 to 0.05) | 0.00 (0.00 to 0.09) |
| 5 | 43 | 0.00 (0.00 to 0.01) | 0.05 (0.00 to 0.10) | 0.06 (0.06 to 0.06) | 0.09 (0.00 to 0.17) |
| 6 | 43 | 0.00 (0.00 to 0.01) | 0.02 (0.00 to 0.12) | 0.07 (0.07 to 0.07) | 0.00 (0.00 to 0.04) |
| 7 | 43 | 0.01 (0.00 to 0.01) | 0.00 (0.00 to 0.07) | 0.07 (0.07 to 0.08) | 0.02 (0.00 to 0.16) |
| 8 | 43 | 0.01 (0.01 to 0.02) | 0.02 (0.00 to 0.07) | 0.08 (0.08 to 0.08) | 0.07 (0.00 to 0.22) |
| 9 | 43 | 0.03 (0.01 to 0.06) | 0.02 (0.00 to 0.10) | 0.09 (0.09 to 0.09) | 0.02 (0.00 to 0.06) |
| 10 | 43 | 0.25 (0.11 to 0.48) | 0.23 (0.02 to 0.51) | 0.11 (0.11 to 0.11) | 0.02 (0.00 to 0.08) |

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
| 8 | 43 | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) | 0.02 (0.02 to 0.02) | 0.00 (0.00 to 0.00) |
| 9 | 43 | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) | 0.02 (0.02 to 0.02) | 0.00 (0.00 to 0.00) |
| 10 | 43 | 0.03 (0.01 to 0.06) | 0.02 (0.00 to 0.08) | 0.02 (0.02 to 0.02) | 0.00 (0.00 to 0.00) |

Interval coverage by true delay band, descriptive only. A band is selected by the outcome that the interval was built before knowing, so it cannot reach the nominal level for any forecast that is not perfect.

| Method | 0-14 | 15-29 | 30-59 | 60-119 | 120+ |
|---|---|---|---|---|---|
| Timetable | n/a | n/a | n/a | n/a | n/a |
| Persistence | 0.82 (n=353, width 18.0) | 0.55 (n=60, width 18.0) | 0.22 (n=18, width 18.0) | 0.00 (n=1, width 18.0) | n/a |
| Historical | 0.93 (n=353, width 24.0) | 0.72 (n=60, width 23.2) | 0.00 (n=18, width 22.5) | 0.00 (n=1, width 43.0) | n/a |
| Model, global CQR | 0.84 (n=353, width 15.6) | 0.48 (n=60, width 15.8) | 0.33 (n=18, width 28.3) | 0.00 (n=1, width 61.1) | n/a |
| Model, Mondrian CQR | 0.86 (n=353, width 16.2) | 0.50 (n=60, width 17.1) | 0.39 (n=18, width 30.3) | 0.00 (n=1, width 64.4) | n/a |

## T-15 minutes, test block

| Method | MAE | 80% coverage | Mean width | Brier (5 bands) | Log loss |
|---|---|---|---|---|---|
| Timetable | 8.88 (7.59 to 10.14) | n/a | n/a | 0.366 (0.258 to 0.460) | 6.32 |
| Persistence | 4.25 (3.20 to 5.23) | 0.803 (0.783 to 0.824) | 12.0 | 0.278 (0.188 to 0.351) | 4.80 |
| Historical | 7.14 (5.90 to 8.32) | 0.859 (0.815 to 0.905) | 23.9 | 0.320 (0.245 to 0.387) | 0.62 |
| Model, global CQR | 4.09 (3.29 to 4.92) | 0.794 (0.749 to 0.840) | 12.1 | 0.214 (0.150 to 0.270) | 0.44 |
| Model, Mondrian CQR | 4.09 (3.29 to 4.92) | 0.806 (0.760 to 0.858) | 12.8 | 0.214 (0.150 to 0.270) | 0.44 |

Against the bar, persistence (lowest validation MAE), the model's MAE is -0.17 minutes (-0.46 to +0.17, paired bootstrap over service dates). The interval includes zero, so the difference is not statistically significant.

Interval coverage by observed lateness at T, which is what the conformal guarantee covers:

| Method | not_started | 0-4 | 5-14 | 15+ |
|---|---|---|---|---|
| Timetable | n/a | n/a | n/a | n/a |
| Persistence | n/a | 0.79 (n=241, width 12.0) | 0.85 (n=151, width 12.0) | 0.70 (n=40, width 12.0) |
| Historical | n/a | 0.84 (n=241, width 23.3) | 0.97 (n=151, width 24.7) | 0.55 (n=40, width 24.3) |
| Model, global CQR | n/a | 0.78 (n=241, width 11.0) | 0.82 (n=151, width 12.0) | 0.80 (n=40, width 19.5) |
| Model, Mondrian CQR | n/a | 0.78 (n=241, width 11.0) | 0.84 (n=151, width 12.1) | 0.85 (n=40, width 26.0) |

Threshold reliability, derived from the band probabilities. Brier is the binary Brier score and ECE the expected calibration error over 10 equal-count bins. Historical band frequencies are the bar.

| Threshold | Share of rows at or above | Method | Brier | ECE |
|---|---|---|---|---|
| 15 minutes | 0.183 | Model | 0.099 (0.066 to 0.127) | 0.069 (0.039 to 0.106) |
| 15 minutes | 0.183 | Historical | 0.155 (0.121 to 0.184) | 0.058 (0.053 to 0.154) |
| 30 minutes | 0.044 | Model | 0.022 (0.003 to 0.044) | 0.015 (0.004 to 0.039) |
| 30 minutes | 0.044 | Historical | 0.044 (0.018 to 0.070) | 0.054 (0.038 to 0.072) |
| 60 minutes | 0.002 | Model | 0.003 (0.000 to 0.008) | 0.000 (0.000 to 0.007) |
| 60 minutes | 0.002 | Historical | 0.002 (0.000 to 0.008) | 0.012 (0.010 to 0.018) |

Reliability for P(at least 15 minutes late) at T-15, 10 equal-count bins from the lowest predicted probability to the highest:

| Bin | Model n | Model predicted | Model observed | Historical predicted | Historical observed |
|---|---|---|---|---|---|
| 1 | 44 | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) | 0.10 (0.10 to 0.11) | 0.27 (0.08 to 0.50) |
| 2 | 44 | 0.00 (0.00 to 0.01) | 0.07 (0.00 to 0.14) | 0.13 (0.13 to 0.14) | 0.14 (0.05 to 0.24) |
| 3 | 43 | 0.01 (0.00 to 0.01) | 0.00 (0.00 to 0.12) | 0.16 (0.16 to 0.16) | 0.16 (0.03 to 0.38) |
| 4 | 43 | 0.01 (0.01 to 0.02) | 0.00 (0.00 to 0.13) | 0.17 (0.17 to 0.17) | 0.19 (0.03 to 0.44) |
| 5 | 43 | 0.02 (0.01 to 0.03) | 0.12 (0.00 to 0.20) | 0.20 (0.19 to 0.21) | 0.16 (0.00 to 0.26) |
| 6 | 43 | 0.04 (0.02 to 0.05) | 0.14 (0.03 to 0.28) | 0.22 (0.22 to 0.22) | 0.23 (0.10 to 0.43) |
| 7 | 43 | 0.06 (0.04 to 0.08) | 0.33 (0.10 to 0.49) | 0.22 (0.22 to 0.23) | 0.21 (0.06 to 0.35) |
| 8 | 43 | 0.11 (0.07 to 0.17) | 0.05 (0.00 to 0.28) | 0.25 (0.24 to 0.25) | 0.12 (0.03 to 0.26) |
| 9 | 43 | 0.35 (0.17 to 0.57) | 0.35 (0.04 to 0.64) | 0.26 (0.26 to 0.27) | 0.16 (0.06 to 0.28) |
| 10 | 43 | 0.87 (0.75 to 0.93) | 0.79 (0.59 to 0.95) | 0.29 (0.29 to 0.29) | 0.19 (0.04 to 0.29) |

Reliability for P(at least 30 minutes late) at T-15, 10 equal-count bins from the lowest predicted probability to the highest:

| Bin | Model n | Model predicted | Model observed | Historical predicted | Historical observed |
|---|---|---|---|---|---|
| 1 | 44 | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) | 0.02 (0.02 to 0.03) | 0.07 (0.00 to 0.15) |
| 2 | 44 | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) | 0.04 (0.04 to 0.04) | 0.00 (0.00 to 0.14) |
| 3 | 43 | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) | 0.04 (0.04 to 0.04) | 0.14 (0.00 to 0.21) |
| 4 | 43 | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) | 0.05 (0.05 to 0.05) | 0.00 (0.00 to 0.09) |
| 5 | 43 | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.04) | 0.06 (0.06 to 0.06) | 0.09 (0.00 to 0.16) |
| 6 | 43 | 0.00 (0.00 to 0.00) | 0.05 (0.00 to 0.15) | 0.07 (0.07 to 0.07) | 0.00 (0.00 to 0.02) |
| 7 | 43 | 0.00 (0.00 to 0.01) | 0.09 (0.00 to 0.20) | 0.07 (0.07 to 0.08) | 0.02 (0.00 to 0.16) |
| 8 | 43 | 0.01 (0.00 to 0.01) | 0.00 (0.00 to 0.11) | 0.08 (0.08 to 0.08) | 0.07 (0.00 to 0.22) |
| 9 | 43 | 0.02 (0.01 to 0.03) | 0.02 (0.00 to 0.09) | 0.09 (0.09 to 0.09) | 0.02 (0.00 to 0.07) |
| 10 | 43 | 0.28 (0.12 to 0.52) | 0.28 (0.04 to 0.57) | 0.11 (0.11 to 0.11) | 0.02 (0.00 to 0.08) |

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
| 8 | 43 | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) | 0.02 (0.02 to 0.02) | 0.00 (0.00 to 0.00) |
| 9 | 43 | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) | 0.02 (0.02 to 0.02) | 0.00 (0.00 to 0.00) |
| 10 | 43 | 0.02 (0.01 to 0.04) | 0.02 (0.00 to 0.08) | 0.02 (0.02 to 0.02) | 0.00 (0.00 to 0.00) |

Interval coverage by true delay band, descriptive only. A band is selected by the outcome that the interval was built before knowing, so it cannot reach the nominal level for any forecast that is not perfect.

| Method | 0-14 | 15-29 | 30-59 | 60-119 | 120+ |
|---|---|---|---|---|---|
| Timetable | n/a | n/a | n/a | n/a | n/a |
| Persistence | 0.85 (n=353, width 12.0) | 0.68 (n=60, width 12.0) | 0.28 (n=18, width 12.0) | 0.00 (n=1, width 12.0) | n/a |
| Historical | 0.93 (n=353, width 24.0) | 0.72 (n=60, width 23.2) | 0.00 (n=18, width 22.5) | 0.00 (n=1, width 43.0) | n/a |
| Model, global CQR | 0.83 (n=353, width 11.5) | 0.67 (n=60, width 12.6) | 0.50 (n=18, width 21.1) | 0.00 (n=1, width 42.8) | n/a |
| Model, Mondrian CQR | 0.84 (n=353, width 11.7) | 0.67 (n=60, width 14.8) | 0.56 (n=18, width 25.1) | 0.00 (n=1, width 49.3) | n/a |

## Limitations

These results cover one route, BHM to EUS. BHM to MAN and AST to EUS via BHM have not been ingested yet, so nothing here says the model transfers to them. The history runs from 2026-06-30 to 2026-10-14, which is summer and the first weeks of autumn, so there is no winter, leaf-fall or storm season, and a seasonal shift would change both the delays and the calibration. The 15 or more minutes late lateness group is small: it held 19 calibration rows at T-120, 20 calibration rows at T-60, 38 calibration rows at T-15 and 17 test rows at T-120, 25 test rows at T-60, 40 test rows at T-15, so its offsets are noisy and are merged into a neighbour where under 30 rows, and its coverage has a wide interval. The model's MAE gain over the bar is not statistically significant at T-15. The test block is 7 days, so the bootstrap intervals are wide, and the 60 minute and longer delays are too rare for the 60 minute threshold table to say much. The conformal guarantee holds on average within each observed lateness group if the days are exchangeable, which a time split only approximates. The delay bands are provisional until the per-operator compensation rules are checked.
