# Model results

MLflow run 4c88d55bbbed49018e3ce05e8199ce5b, commit fc20d828cc2e999031dd5eba9a9f52ecda8f23f1. One model was trained on 159 complete days (2026-04-15 to 2026-09-22) of ast-bhm, bhm-eus, tuned and calibrated on 14 validation days (2026-09-24 to 2026-10-07), the earlier half tuning and the later half calibrating, and scored once on 7 test days (2026-10-09 to 2026-10-15). Brackets are 95% bootstrap intervals over service dates. Intervals are nominal 80% intervals. Delay is minutes late at the route destination, and cancelled services are not scored. Each route is scored against baselines fitted on that route's own training days.

The model takes the minutes to scheduled arrival as a number, trained on snapshots at 5, 15, 30, 60, 90, 120 minutes before arrival, with the route and per-route 28 day history as features. The interval comes from LightGBM quantile models at 0.1, 0.5 and 0.9, widened by a conformal offset computed separately for each route, horizon bucket (under 20, 20 to 74, 75 or more minutes) and observed lateness group at T (not started, 0 to 4, 5 to 14, and 15 or more minutes late). The lateness group uses only what is known at T, and a group with fewer than 30 calibration rows is merged into its neighbour and logged. The target per horizon bucket was chosen by tuning pinball loss between the direct delay and the delay minus the lateness at T: under 20: direct, 20 to 74: direct, 75 or more: residual. Band probabilities come from a LightGBM band model with temperature calibration. P(at least 15) and P(at least 30) are then recalibrated separately per horizon bucket, choosing among isotonic regression, Platt scaling and no recalibration by log loss on the tuning rows, with the methods fitted on the calibration rows. The choice was under 20 platt, 20 to 74 none, 75 or more platt. The band probabilities are rebuilt from the recalibrated exceedances so that they stay monotone and sum to 1. The delay bands are provisional.

Two kinds of check are reported. The conformal guarantee is coverage conditional on the state known at T, so coverage by observed lateness is the interval check. For passengers and Delay Repay, what matters is whether the probabilities of reaching 15, 30 and 60 minutes late are calibrated, so threshold reliability is the passenger facing check. Coverage by true delay band is shown for description only. Results for every grid horizon are in the run's results.json; this page shows 120, 60 and 15 minutes.

## ast-bhm, T-120 minutes, test block

| Method | MAE | 80% coverage | Mean width | Brier (5 bands) | Log loss |
|---|---|---|---|---|---|
| Timetable | 2.48 (2.01 to 2.95) | n/a | n/a | 0.007 (0.000 to 0.020) | 0.12 |
| Persistence | 2.48 (2.01 to 2.95) | 0.815 (0.756 to 0.868) | 6.0 | 0.007 (0.000 to 0.020) | 0.12 |
| Historical | 2.31 (1.95 to 2.63) | 0.832 (0.791 to 0.865) | 6.9 | 0.007 (0.000 to 0.020) | 0.03 |
| Model, global CQR | 2.30 (1.91 to 2.68) | 0.830 (0.789 to 0.866) | 7.1 | 0.007 (0.000 to 0.020) | 0.02 |
| Model, Mondrian CQR | 2.30 (1.91 to 2.68) | 0.830 (0.789 to 0.866) | 7.1 | 0.007 (0.000 to 0.020) | 0.02 |

Against the bar, historical (lowest validation MAE), the model's MAE is -0.01 minutes (-0.08 to +0.07, paired bootstrap over service dates). The interval includes zero, so the difference is not statistically significant.

Interval coverage by observed lateness at T, which is what the conformal guarantee covers:

| Method | not_started | 0-4 | 5-14 | 15+ |
|---|---|---|---|---|
| Timetable | n/a | n/a | n/a | n/a |
| Persistence | 0.81 (n=589, width 6.0) | n/a | n/a | n/a |
| Historical | 0.83 (n=589, width 6.9) | n/a | n/a | n/a |
| Model, global CQR | 0.83 (n=589, width 7.1) | n/a | n/a | n/a |
| Model, Mondrian CQR | 0.83 (n=589, width 7.1) | n/a | n/a | n/a |

Threshold reliability, derived from the band probabilities. Brier is the binary Brier score and ECE the expected calibration error over 10 equal-count bins. Historical band frequencies are the bar.

| Threshold | Share of rows at or above | Method | Brier | ECE |
|---|---|---|---|---|
| 15 minutes | 0.003 | Model, after threshold calibration | 0.003 (0.000 to 0.010) | 0.009 (0.005 to 0.014) |
| 15 minutes | 0.003 | Model, before threshold calibration | 0.003 (0.000 to 0.010) | 0.007 (0.003 to 0.013) |
| 15 minutes | 0.003 | Historical | 0.003 (0.000 to 0.010) | 0.010 (0.008 to 0.013) |
| 30 minutes | 0.000 | Model, after threshold calibration | 0.000 (0.000 to 0.000) | 0.001 (0.001 to 0.001) |
| 30 minutes | 0.000 | Model, before threshold calibration | 0.000 (0.000 to 0.000) | 0.000 (0.000 to 0.000) |
| 30 minutes | 0.000 | Historical | 0.000 (0.000 to 0.000) | 0.001 (0.001 to 0.001) |
| 60 minutes | 0.000 | Model, after threshold calibration | 0.000 (0.000 to 0.000) | 0.000 (0.000 to 0.000) |
| 60 minutes | 0.000 | Model, before threshold calibration | 0.000 (0.000 to 0.000) | 0.000 (0.000 to 0.000) |
| 60 minutes | 0.000 | Historical | 0.000 (0.000 to 0.000) | 0.000 (0.000 to 0.000) |

Reliability for P(at least 15 minutes late) at T-120, 10 equal-count bins from the lowest predicted probability to the highest:

| Bin | Model n | Model predicted | Model observed | Historical predicted | Historical observed |
|---|---|---|---|---|---|
| 1 | 59 | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) |
| 2 | 59 | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) |
| 3 | 59 | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) |
| 4 | 59 | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) | 0.01 (0.01 to 0.01) | 0.02 (0.00 to 0.04) |
| 5 | 59 | 0.00 (0.00 to 0.01) | 0.00 (0.00 to 0.00) | 0.01 (0.01 to 0.01) | 0.00 (0.00 to 0.03) |
| 6 | 59 | 0.01 (0.00 to 0.01) | 0.00 (0.00 to 0.00) | 0.01 (0.01 to 0.01) | 0.00 (0.00 to 0.00) |
| 7 | 59 | 0.01 (0.01 to 0.01) | 0.00 (0.00 to 0.00) | 0.01 (0.01 to 0.01) | 0.02 (0.00 to 0.03) |
| 8 | 59 | 0.01 (0.01 to 0.01) | 0.00 (0.00 to 0.05) | 0.01 (0.01 to 0.01) | 0.00 (0.00 to 0.04) |
| 9 | 59 | 0.01 (0.01 to 0.01) | 0.03 (0.00 to 0.05) | 0.02 (0.02 to 0.02) | 0.00 (0.00 to 0.00) |
| 10 | 58 | 0.03 (0.02 to 0.03) | 0.00 (0.00 to 0.02) | 0.02 (0.02 to 0.02) | 0.00 (0.00 to 0.00) |

Reliability for P(at least 30 minutes late) at T-120, 10 equal-count bins from the lowest predicted probability to the highest:

| Bin | Model n | Model predicted | Model observed | Historical predicted | Historical observed |
|---|---|---|---|---|---|
| 1 | 59 | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) |
| 2 | 59 | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) |
| 3 | 59 | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) |
| 4 | 59 | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) |
| 5 | 59 | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) |
| 6 | 59 | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) |
| 7 | 59 | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) |
| 8 | 59 | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) |
| 9 | 59 | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) |
| 10 | 58 | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) |

Reliability for P(at least 60 minutes late) at T-120, 10 equal-count bins from the lowest predicted probability to the highest:

| Bin | Model n | Model predicted | Model observed | Historical predicted | Historical observed |
|---|---|---|---|---|---|
| 1 | 59 | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) |
| 2 | 59 | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) |
| 3 | 59 | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) |
| 4 | 59 | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) |
| 5 | 59 | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) |
| 6 | 59 | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) |
| 7 | 59 | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) |
| 8 | 59 | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) |
| 9 | 59 | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) |
| 10 | 58 | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) |

Interval coverage by true delay band, descriptive only. A band is selected by the outcome that the interval was built before knowing, so it cannot reach the nominal level for any forecast that is not perfect.

| Method | 0-14 | 15-29 | 30-59 | 60-119 | 120+ |
|---|---|---|---|---|---|
| Timetable | n/a | n/a | n/a | n/a | n/a |
| Persistence | 0.82 (n=587, width 6.0) | 0.00 (n=2, width 6.0) | n/a | n/a | n/a |
| Historical | 0.83 (n=587, width 6.9) | 0.00 (n=2, width 7.0) | n/a | n/a | n/a |
| Model, global CQR | 0.83 (n=587, width 7.1) | 0.00 (n=2, width 8.0) | n/a | n/a | n/a |
| Model, Mondrian CQR | 0.83 (n=587, width 7.1) | 0.00 (n=2, width 8.0) | n/a | n/a | n/a |

## ast-bhm, T-60 minutes, test block

| Method | MAE | 80% coverage | Mean width | Brier (5 bands) | Log loss |
|---|---|---|---|---|---|
| Timetable | 2.48 (2.01 to 2.95) | n/a | n/a | 0.007 (0.000 to 0.020) | 0.12 |
| Persistence | 2.48 (2.01 to 2.95) | 0.815 (0.756 to 0.868) | 6.0 | 0.007 (0.000 to 0.020) | 0.12 |
| Historical | 2.31 (1.95 to 2.63) | 0.832 (0.791 to 0.865) | 6.9 | 0.007 (0.000 to 0.020) | 0.03 |
| Model, global CQR | 2.30 (1.96 to 2.62) | 0.846 (0.790 to 0.882) | 7.5 | 0.007 (0.000 to 0.020) | 0.02 |
| Model, Mondrian CQR | 2.30 (1.96 to 2.62) | 0.788 (0.754 to 0.820) | 7.4 | 0.007 (0.000 to 0.020) | 0.02 |

Against the bar, historical (lowest validation MAE), the model's MAE is -0.00 minutes (-0.07 to +0.03, paired bootstrap over service dates). The interval includes zero, so the difference is not statistically significant.

Interval coverage by observed lateness at T, which is what the conformal guarantee covers:

| Method | not_started | 0-4 | 5-14 | 15+ |
|---|---|---|---|---|
| Timetable | n/a | n/a | n/a | n/a |
| Persistence | 0.81 (n=584, width 6.0) | 1.00 (n=5, width 6.0) | n/a | n/a |
| Historical | 0.83 (n=584, width 6.9) | 1.00 (n=5, width 5.0) | n/a | n/a |
| Model, global CQR | 0.85 (n=584, width 7.5) | 0.80 (n=5, width 5.9) | n/a | n/a |
| Model, Mondrian CQR | 0.79 (n=584, width 7.4) | 0.80 (n=5, width 5.8) | n/a | n/a |

Threshold reliability, derived from the band probabilities. Brier is the binary Brier score and ECE the expected calibration error over 10 equal-count bins. Historical band frequencies are the bar.

| Threshold | Share of rows at or above | Method | Brier | ECE |
|---|---|---|---|---|
| 15 minutes | 0.003 | Model, after threshold calibration | 0.003 (0.000 to 0.010) | 0.007 (0.004 to 0.014) |
| 15 minutes | 0.003 | Model, before threshold calibration | 0.003 (0.000 to 0.010) | 0.007 (0.004 to 0.014) |
| 15 minutes | 0.003 | Historical | 0.003 (0.000 to 0.010) | 0.010 (0.008 to 0.013) |
| 30 minutes | 0.000 | Model, after threshold calibration | 0.000 (0.000 to 0.000) | 0.000 (0.000 to 0.000) |
| 30 minutes | 0.000 | Model, before threshold calibration | 0.000 (0.000 to 0.000) | 0.000 (0.000 to 0.000) |
| 30 minutes | 0.000 | Historical | 0.000 (0.000 to 0.000) | 0.001 (0.001 to 0.001) |
| 60 minutes | 0.000 | Model, after threshold calibration | 0.000 (0.000 to 0.000) | 0.000 (0.000 to 0.000) |
| 60 minutes | 0.000 | Model, before threshold calibration | 0.000 (0.000 to 0.000) | 0.000 (0.000 to 0.000) |
| 60 minutes | 0.000 | Historical | 0.000 (0.000 to 0.000) | 0.000 (0.000 to 0.000) |

Reliability for P(at least 15 minutes late) at T-60, 10 equal-count bins from the lowest predicted probability to the highest:

| Bin | Model n | Model predicted | Model observed | Historical predicted | Historical observed |
|---|---|---|---|---|---|
| 1 | 59 | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) |
| 2 | 59 | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) |
| 3 | 59 | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) |
| 4 | 59 | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) | 0.01 (0.01 to 0.01) | 0.02 (0.00 to 0.04) |
| 5 | 59 | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) | 0.01 (0.01 to 0.01) | 0.00 (0.00 to 0.03) |
| 6 | 59 | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) | 0.01 (0.01 to 0.01) | 0.00 (0.00 to 0.00) |
| 7 | 59 | 0.00 (0.00 to 0.01) | 0.00 (0.00 to 0.00) | 0.01 (0.01 to 0.01) | 0.02 (0.00 to 0.03) |
| 8 | 59 | 0.01 (0.00 to 0.01) | 0.00 (0.00 to 0.05) | 0.01 (0.01 to 0.01) | 0.00 (0.00 to 0.04) |
| 9 | 59 | 0.01 (0.01 to 0.01) | 0.03 (0.00 to 0.07) | 0.02 (0.02 to 0.02) | 0.00 (0.00 to 0.00) |
| 10 | 58 | 0.03 (0.02 to 0.03) | 0.00 (0.00 to 0.02) | 0.02 (0.02 to 0.02) | 0.00 (0.00 to 0.00) |

Reliability for P(at least 30 minutes late) at T-60, 10 equal-count bins from the lowest predicted probability to the highest:

| Bin | Model n | Model predicted | Model observed | Historical predicted | Historical observed |
|---|---|---|---|---|---|
| 1 | 59 | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) |
| 2 | 59 | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) |
| 3 | 59 | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) |
| 4 | 59 | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) |
| 5 | 59 | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) |
| 6 | 59 | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) |
| 7 | 59 | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) |
| 8 | 59 | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) |
| 9 | 59 | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) |
| 10 | 58 | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) |

Reliability for P(at least 60 minutes late) at T-60, 10 equal-count bins from the lowest predicted probability to the highest:

| Bin | Model n | Model predicted | Model observed | Historical predicted | Historical observed |
|---|---|---|---|---|---|
| 1 | 59 | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) |
| 2 | 59 | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) |
| 3 | 59 | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) |
| 4 | 59 | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) |
| 5 | 59 | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) |
| 6 | 59 | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) |
| 7 | 59 | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) |
| 8 | 59 | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) |
| 9 | 59 | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) |
| 10 | 58 | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) |

Interval coverage by true delay band, descriptive only. A band is selected by the outcome that the interval was built before knowing, so it cannot reach the nominal level for any forecast that is not perfect.

| Method | 0-14 | 15-29 | 30-59 | 60-119 | 120+ |
|---|---|---|---|---|---|
| Timetable | n/a | n/a | n/a | n/a | n/a |
| Persistence | 0.82 (n=587, width 6.0) | 0.00 (n=2, width 6.0) | n/a | n/a | n/a |
| Historical | 0.83 (n=587, width 6.9) | 0.00 (n=2, width 7.0) | n/a | n/a | n/a |
| Model, global CQR | 0.85 (n=587, width 7.5) | 0.00 (n=2, width 10.6) | n/a | n/a | n/a |
| Model, Mondrian CQR | 0.79 (n=587, width 7.4) | 0.00 (n=2, width 10.5) | n/a | n/a | n/a |

## ast-bhm, T-15 minutes, test block

| Method | MAE | 80% coverage | Mean width | Brier (5 bands) | Log loss |
|---|---|---|---|---|---|
| Timetable | 2.48 (2.01 to 2.95) | n/a | n/a | 0.007 (0.000 to 0.020) | 0.12 |
| Persistence | 2.19 (1.83 to 2.56) | 0.839 (0.800 to 0.877) | 6.0 | 0.007 (0.000 to 0.020) | 0.12 |
| Historical | 2.31 (1.95 to 2.63) | 0.832 (0.791 to 0.865) | 6.9 | 0.007 (0.000 to 0.020) | 0.03 |
| Model, global CQR | 1.93 (1.65 to 2.22) | 0.823 (0.782 to 0.857) | 6.2 | 0.007 (0.000 to 0.020) | 0.02 |
| Model, Mondrian CQR | 1.93 (1.65 to 2.22) | 0.805 (0.761 to 0.841) | 6.2 | 0.007 (0.000 to 0.020) | 0.02 |

Against the bar, persistence (lowest validation MAE), the model's MAE is -0.26 minutes (-0.37 to -0.15, paired bootstrap over service dates). The interval excludes zero.

Interval coverage by observed lateness at T, which is what the conformal guarantee covers:

| Method | not_started | 0-4 | 5-14 | 15+ |
|---|---|---|---|---|
| Timetable | n/a | n/a | n/a | n/a |
| Persistence | 0.00 (n=4, width 6.0) | 0.85 (n=550, width 6.0) | 0.71 (n=35, width 6.0) | n/a |
| Historical | 0.00 (n=4, width 6.9) | 0.86 (n=550, width 6.9) | 0.54 (n=35, width 6.9) | n/a |
| Model, global CQR | 0.50 (n=4, width 14.7) | 0.83 (n=550, width 6.0) | 0.83 (n=35, width 8.2) | n/a |
| Model, Mondrian CQR | 0.50 (n=4, width 14.6) | 0.81 (n=550, width 6.0) | 0.83 (n=35, width 8.6) | n/a |

Threshold reliability, derived from the band probabilities. Brier is the binary Brier score and ECE the expected calibration error over 10 equal-count bins. Historical band frequencies are the bar.

| Threshold | Share of rows at or above | Method | Brier | ECE |
|---|---|---|---|---|
| 15 minutes | 0.003 | Model, after threshold calibration | 0.003 (0.000 to 0.010) | 0.003 (0.001 to 0.010) |
| 15 minutes | 0.003 | Model, before threshold calibration | 0.003 (0.000 to 0.010) | 0.003 (0.001 to 0.009) |
| 15 minutes | 0.003 | Historical | 0.003 (0.000 to 0.010) | 0.010 (0.008 to 0.013) |
| 30 minutes | 0.000 | Model, after threshold calibration | 0.000 (0.000 to 0.000) | 0.000 (0.000 to 0.000) |
| 30 minutes | 0.000 | Model, before threshold calibration | 0.000 (0.000 to 0.000) | 0.000 (0.000 to 0.000) |
| 30 minutes | 0.000 | Historical | 0.000 (0.000 to 0.000) | 0.001 (0.001 to 0.001) |
| 60 minutes | 0.000 | Model, after threshold calibration | 0.000 (0.000 to 0.000) | 0.000 (0.000 to 0.000) |
| 60 minutes | 0.000 | Model, before threshold calibration | 0.000 (0.000 to 0.000) | 0.000 (0.000 to 0.000) |
| 60 minutes | 0.000 | Historical | 0.000 (0.000 to 0.000) | 0.000 (0.000 to 0.000) |

Reliability for P(at least 15 minutes late) at T-15, 10 equal-count bins from the lowest predicted probability to the highest:

| Bin | Model n | Model predicted | Model observed | Historical predicted | Historical observed |
|---|---|---|---|---|---|
| 1 | 59 | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) |
| 2 | 59 | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) |
| 3 | 59 | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) |
| 4 | 59 | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) | 0.01 (0.01 to 0.01) | 0.02 (0.00 to 0.04) |
| 5 | 59 | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) | 0.01 (0.01 to 0.01) | 0.00 (0.00 to 0.03) |
| 6 | 59 | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) | 0.01 (0.01 to 0.01) | 0.00 (0.00 to 0.00) |
| 7 | 59 | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.02) | 0.01 (0.01 to 0.01) | 0.02 (0.00 to 0.03) |
| 8 | 59 | 0.00 (0.00 to 0.00) | 0.02 (0.00 to 0.05) | 0.01 (0.01 to 0.01) | 0.00 (0.00 to 0.04) |
| 9 | 59 | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.02) | 0.02 (0.02 to 0.02) | 0.00 (0.00 to 0.00) |
| 10 | 58 | 0.01 (0.00 to 0.01) | 0.02 (0.00 to 0.05) | 0.02 (0.02 to 0.02) | 0.00 (0.00 to 0.00) |

Reliability for P(at least 30 minutes late) at T-15, 10 equal-count bins from the lowest predicted probability to the highest:

| Bin | Model n | Model predicted | Model observed | Historical predicted | Historical observed |
|---|---|---|---|---|---|
| 1 | 59 | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) |
| 2 | 59 | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) |
| 3 | 59 | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) |
| 4 | 59 | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) |
| 5 | 59 | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) |
| 6 | 59 | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) |
| 7 | 59 | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) |
| 8 | 59 | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) |
| 9 | 59 | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) |
| 10 | 58 | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) |

Reliability for P(at least 60 minutes late) at T-15, 10 equal-count bins from the lowest predicted probability to the highest:

| Bin | Model n | Model predicted | Model observed | Historical predicted | Historical observed |
|---|---|---|---|---|---|
| 1 | 59 | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) |
| 2 | 59 | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) |
| 3 | 59 | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) |
| 4 | 59 | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) |
| 5 | 59 | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) |
| 6 | 59 | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) |
| 7 | 59 | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) |
| 8 | 59 | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) |
| 9 | 59 | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) |
| 10 | 58 | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) |

Interval coverage by true delay band, descriptive only. A band is selected by the outcome that the interval was built before knowing, so it cannot reach the nominal level for any forecast that is not perfect.

| Method | 0-14 | 15-29 | 30-59 | 60-119 | 120+ |
|---|---|---|---|---|---|
| Timetable | n/a | n/a | n/a | n/a | n/a |
| Persistence | 0.84 (n=587, width 6.0) | 0.00 (n=2, width 6.0) | n/a | n/a | n/a |
| Historical | 0.83 (n=587, width 6.9) | 0.00 (n=2, width 7.0) | n/a | n/a | n/a |
| Model, global CQR | 0.82 (n=587, width 6.2) | 0.50 (n=2, width 12.3) | n/a | n/a | n/a |
| Model, Mondrian CQR | 0.81 (n=587, width 6.2) | 0.50 (n=2, width 12.2) | n/a | n/a | n/a |

## bhm-eus, T-120 minutes, test block

| Method | MAE | 80% coverage | Mean width | Brier (5 bands) | Log loss |
|---|---|---|---|---|---|
| Timetable | 8.74 (7.38 to 10.21) | n/a | n/a | 0.353 (0.240 to 0.455) | 6.10 |
| Persistence | 7.49 (5.88 to 8.86) | 0.830 (0.764 to 0.885) | 22.0 | 0.363 (0.240 to 0.465) | 6.27 |
| Historical | 6.94 (5.56 to 8.42) | 0.844 (0.782 to 0.902) | 22.4 | 0.304 (0.220 to 0.391) | 0.59 |
| Model, global CQR | 6.27 (4.94 to 7.51) | 0.786 (0.713 to 0.857) | 17.6 | 0.301 (0.202 to 0.389) | 0.66 |
| Model, Mondrian CQR | 6.27 (4.94 to 7.51) | 0.814 (0.744 to 0.883) | 18.5 | 0.301 (0.202 to 0.389) | 0.66 |

Against the bar, historical (lowest validation MAE), the model's MAE is -0.67 minutes (-1.69 to +0.10, paired bootstrap over service dates). The interval includes zero, so the difference is not statistically significant.

Interval coverage by observed lateness at T, which is what the conformal guarantee covers:

| Method | not_started | 0-4 | 5-14 | 15+ |
|---|---|---|---|---|
| Timetable | n/a | n/a | n/a | n/a |
| Persistence | 0.85 (n=150, width 22.0) | 0.87 (n=214, width 22.0) | 0.72 (n=50, width 22.0) | 0.38 (n=16, width 22.0) |
| Historical | 0.87 (n=150, width 21.7) | 0.84 (n=214, width 22.3) | 0.88 (n=50, width 23.7) | 0.50 (n=16, width 25.5) |
| Model, global CQR | 0.86 (n=150, width 21.2) | 0.75 (n=214, width 13.5) | 0.80 (n=50, width 18.9) | 0.50 (n=16, width 35.3) |
| Model, Mondrian CQR | 0.85 (n=150, width 19.8) | 0.76 (n=214, width 14.0) | 0.88 (n=50, width 22.2) | 0.94 (n=16, width 57.0) |

Threshold reliability, derived from the band probabilities. Brier is the binary Brier score and ECE the expected calibration error over 10 equal-count bins. Historical band frequencies are the bar.

| Threshold | Share of rows at or above | Method | Brier | ECE |
|---|---|---|---|---|
| 15 minutes | 0.177 | Model, after threshold calibration | 0.141 (0.092 to 0.186) | 0.083 (0.049 to 0.140) |
| 15 minutes | 0.177 | Model, before threshold calibration | 0.141 (0.093 to 0.186) | 0.085 (0.048 to 0.141) |
| 15 minutes | 0.177 | Historical | 0.146 (0.108 to 0.186) | 0.047 (0.044 to 0.133) |
| 30 minutes | 0.044 | Model, after threshold calibration | 0.038 (0.015 to 0.064) | 0.032 (0.017 to 0.064) |
| 30 minutes | 0.044 | Model, before threshold calibration | 0.036 (0.015 to 0.059) | 0.035 (0.018 to 0.058) |
| 30 minutes | 0.044 | Historical | 0.044 (0.018 to 0.073) | 0.046 (0.037 to 0.080) |
| 60 minutes | 0.002 | Model, after threshold calibration | 0.003 (0.000 to 0.008) | 0.001 (0.001 to 0.007) |
| 60 minutes | 0.002 | Model, before threshold calibration | 0.004 (0.000 to 0.009) | 0.002 (0.001 to 0.008) |
| 60 minutes | 0.002 | Historical | 0.002 (0.000 to 0.008) | 0.008 (0.008 to 0.014) |

Reliability for P(at least 15 minutes late) at T-120, 10 equal-count bins from the lowest predicted probability to the highest:

| Bin | Model n | Model predicted | Model observed | Historical predicted | Historical observed |
|---|---|---|---|---|---|
| 1 | 43 | 0.02 (0.02 to 0.02) | 0.09 (0.00 to 0.21) | 0.09 (0.09 to 0.10) | 0.14 (0.03 to 0.30) |
| 2 | 43 | 0.03 (0.02 to 0.03) | 0.02 (0.00 to 0.09) | 0.12 (0.12 to 0.12) | 0.12 (0.00 to 0.41) |
| 3 | 43 | 0.04 (0.03 to 0.04) | 0.12 (0.00 to 0.26) | 0.13 (0.13 to 0.14) | 0.16 (0.02 to 0.30) |
| 4 | 43 | 0.05 (0.04 to 0.06) | 0.12 (0.03 to 0.26) | 0.14 (0.14 to 0.15) | 0.21 (0.05 to 0.39) |
| 5 | 43 | 0.06 (0.05 to 0.07) | 0.23 (0.02 to 0.43) | 0.17 (0.16 to 0.19) | 0.12 (0.05 to 0.25) |
| 6 | 43 | 0.07 (0.06 to 0.09) | 0.23 (0.02 to 0.51) | 0.19 (0.19 to 0.20) | 0.19 (0.05 to 0.33) |
| 7 | 43 | 0.09 (0.07 to 0.12) | 0.07 (0.00 to 0.28) | 0.21 (0.21 to 0.21) | 0.23 (0.06 to 0.41) |
| 8 | 43 | 0.12 (0.09 to 0.14) | 0.26 (0.04 to 0.39) | 0.22 (0.22 to 0.23) | 0.28 (0.10 to 0.47) |
| 9 | 43 | 0.16 (0.14 to 0.18) | 0.26 (0.19 to 0.35) | 0.24 (0.23 to 0.24) | 0.16 (0.06 to 0.30) |
| 10 | 43 | 0.35 (0.25 to 0.47) | 0.37 (0.16 to 0.65) | 0.27 (0.26 to 0.27) | 0.16 (0.04 to 0.25) |

Reliability for P(at least 30 minutes late) at T-120, 10 equal-count bins from the lowest predicted probability to the highest:

| Bin | Model n | Model predicted | Model observed | Historical predicted | Historical observed |
|---|---|---|---|---|---|
| 1 | 43 | 0.01 (0.01 to 0.01) | 0.02 (0.00 to 0.07) | 0.02 (0.02 to 0.02) | 0.12 (0.02 to 0.24) |
| 2 | 43 | 0.01 (0.01 to 0.01) | 0.00 (0.00 to 0.05) | 0.03 (0.03 to 0.03) | 0.00 (0.00 to 0.08) |
| 3 | 43 | 0.01 (0.01 to 0.01) | 0.02 (0.00 to 0.07) | 0.03 (0.03 to 0.03) | 0.09 (0.00 to 0.19) |
| 4 | 43 | 0.01 (0.01 to 0.02) | 0.02 (0.00 to 0.19) | 0.04 (0.03 to 0.04) | 0.05 (0.00 to 0.15) |
| 5 | 43 | 0.02 (0.01 to 0.02) | 0.09 (0.00 to 0.19) | 0.04 (0.04 to 0.05) | 0.00 (0.00 to 0.00) |
| 6 | 43 | 0.02 (0.02 to 0.03) | 0.00 (0.00 to 0.13) | 0.06 (0.06 to 0.06) | 0.07 (0.00 to 0.18) |
| 7 | 43 | 0.03 (0.02 to 0.03) | 0.09 (0.00 to 0.16) | 0.07 (0.07 to 0.07) | 0.07 (0.00 to 0.23) |
| 8 | 43 | 0.03 (0.03 to 0.04) | 0.00 (0.00 to 0.14) | 0.08 (0.08 to 0.08) | 0.02 (0.00 to 0.11) |
| 9 | 43 | 0.04 (0.04 to 0.05) | 0.02 (0.00 to 0.07) | 0.08 (0.08 to 0.09) | 0.02 (0.00 to 0.06) |
| 10 | 43 | 0.11 (0.08 to 0.15) | 0.16 (0.02 to 0.38) | 0.10 (0.10 to 0.10) | 0.00 (0.00 to 0.03) |

Reliability for P(at least 60 minutes late) at T-120, 10 equal-count bins from the lowest predicted probability to the highest:

| Bin | Model n | Model predicted | Model observed | Historical predicted | Historical observed |
|---|---|---|---|---|---|
| 1 | 43 | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) |
| 2 | 43 | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) |
| 3 | 43 | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) |
| 4 | 43 | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) | 0.01 (0.01 to 0.01) | 0.00 (0.00 to 0.00) |
| 5 | 43 | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) | 0.01 (0.01 to 0.01) | 0.00 (0.00 to 0.00) |
| 6 | 43 | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) | 0.01 (0.01 to 0.01) | 0.00 (0.00 to 0.00) |
| 7 | 43 | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) | 0.01 (0.01 to 0.01) | 0.00 (0.00 to 0.00) |
| 8 | 43 | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) | 0.01 (0.01 to 0.01) | 0.02 (0.00 to 0.08) |
| 9 | 43 | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) | 0.01 (0.01 to 0.01) | 0.00 (0.00 to 0.00) |
| 10 | 43 | 0.03 (0.01 to 0.05) | 0.02 (0.00 to 0.08) | 0.02 (0.02 to 0.02) | 0.00 (0.00 to 0.00) |

Interval coverage by true delay band, descriptive only. A band is selected by the outcome that the interval was built before knowing, so it cannot reach the nominal level for any forecast that is not perfect.

| Method | 0-14 | 15-29 | 30-59 | 60-119 | 120+ |
|---|---|---|---|---|---|
| Timetable | n/a | n/a | n/a | n/a | n/a |
| Persistence | 0.89 (n=354, width 22.0) | 0.63 (n=57, width 22.0) | 0.22 (n=18, width 22.0) | 1.00 (n=1, width 22.0) | n/a |
| Historical | 0.91 (n=354, width 22.3) | 0.70 (n=57, width 23.3) | 0.00 (n=18, width 20.9) | 0.00 (n=1, width 31.5) | n/a |
| Model, global CQR | 0.85 (n=354, width 16.8) | 0.54 (n=57, width 19.4) | 0.33 (n=18, width 26.2) | 0.00 (n=1, width 53.8) | n/a |
| Model, Mondrian CQR | 0.88 (n=354, width 17.4) | 0.53 (n=57, width 20.0) | 0.39 (n=18, width 34.4) | 1.00 (n=1, width 75.5) | n/a |

## bhm-eus, T-60 minutes, test block

| Method | MAE | 80% coverage | Mean width | Brier (5 bands) | Log loss |
|---|---|---|---|---|---|
| Timetable | 8.74 (7.38 to 10.21) | n/a | n/a | 0.353 (0.240 to 0.455) | 6.10 |
| Persistence | 6.56 (5.24 to 7.88) | 0.795 (0.719 to 0.861) | 18.0 | 0.344 (0.229 to 0.437) | 5.94 |
| Historical | 6.94 (5.56 to 8.42) | 0.844 (0.782 to 0.902) | 22.4 | 0.304 (0.220 to 0.391) | 0.59 |
| Model, global CQR | 5.44 (4.44 to 6.58) | 0.774 (0.713 to 0.847) | 15.7 | 0.275 (0.191 to 0.353) | 0.55 |
| Model, Mondrian CQR | 5.44 (4.44 to 6.58) | 0.812 (0.749 to 0.881) | 16.5 | 0.275 (0.191 to 0.353) | 0.55 |

Against the bar, persistence (lowest validation MAE), the model's MAE is -1.13 minutes (-1.60 to -0.43, paired bootstrap over service dates). The interval excludes zero.

Interval coverage by observed lateness at T, which is what the conformal guarantee covers:

| Method | not_started | 0-4 | 5-14 | 15+ |
|---|---|---|---|---|
| Timetable | n/a | n/a | n/a | n/a |
| Persistence | 0.50 (n=2, width 18.0) | 0.84 (n=309, width 18.0) | 0.75 (n=93, width 18.0) | 0.42 (n=26, width 18.0) |
| Historical | 0.50 (n=2, width 20.5) | 0.84 (n=309, width 22.1) | 0.91 (n=93, width 23.1) | 0.62 (n=26, width 23.6) |
| Model, global CQR | 0.50 (n=2, width 31.3) | 0.76 (n=309, width 13.9) | 0.82 (n=93, width 17.6) | 0.81 (n=26, width 29.1) |
| Model, Mondrian CQR | 0.50 (n=2, width 32.1) | 0.80 (n=309, width 14.8) | 0.85 (n=93, width 18.1) | 0.81 (n=26, width 29.9) |

Threshold reliability, derived from the band probabilities. Brier is the binary Brier score and ECE the expected calibration error over 10 equal-count bins. Historical band frequencies are the bar.

| Threshold | Share of rows at or above | Method | Brier | ECE |
|---|---|---|---|---|
| 15 minutes | 0.177 | Model, after threshold calibration | 0.132 (0.091 to 0.172) | 0.077 (0.055 to 0.130) |
| 15 minutes | 0.177 | Model, before threshold calibration | 0.132 (0.091 to 0.172) | 0.077 (0.055 to 0.130) |
| 15 minutes | 0.177 | Historical | 0.146 (0.108 to 0.186) | 0.047 (0.043 to 0.136) |
| 30 minutes | 0.044 | Model, after threshold calibration | 0.032 (0.012 to 0.054) | 0.023 (0.009 to 0.054) |
| 30 minutes | 0.044 | Model, before threshold calibration | 0.032 (0.012 to 0.054) | 0.023 (0.009 to 0.054) |
| 30 minutes | 0.044 | Historical | 0.044 (0.018 to 0.073) | 0.049 (0.037 to 0.080) |
| 60 minutes | 0.002 | Model, after threshold calibration | 0.003 (0.000 to 0.008) | 0.002 (0.001 to 0.007) |
| 60 minutes | 0.002 | Model, before threshold calibration | 0.003 (0.000 to 0.008) | 0.002 (0.001 to 0.007) |
| 60 minutes | 0.002 | Historical | 0.002 (0.000 to 0.008) | 0.008 (0.008 to 0.014) |

Reliability for P(at least 15 minutes late) at T-60, 10 equal-count bins from the lowest predicted probability to the highest:

| Bin | Model n | Model predicted | Model observed | Historical predicted | Historical observed |
|---|---|---|---|---|---|
| 1 | 43 | 0.01 (0.01 to 0.02) | 0.02 (0.00 to 0.08) | 0.09 (0.09 to 0.10) | 0.14 (0.03 to 0.30) |
| 2 | 43 | 0.02 (0.02 to 0.02) | 0.07 (0.00 to 0.19) | 0.12 (0.12 to 0.12) | 0.12 (0.00 to 0.41) |
| 3 | 43 | 0.03 (0.02 to 0.03) | 0.05 (0.02 to 0.20) | 0.13 (0.13 to 0.14) | 0.16 (0.02 to 0.30) |
| 4 | 43 | 0.03 (0.03 to 0.04) | 0.21 (0.00 to 0.29) | 0.14 (0.14 to 0.15) | 0.21 (0.05 to 0.40) |
| 5 | 43 | 0.04 (0.03 to 0.05) | 0.19 (0.05 to 0.33) | 0.17 (0.16 to 0.19) | 0.12 (0.05 to 0.24) |
| 6 | 43 | 0.06 (0.04 to 0.08) | 0.12 (0.05 to 0.28) | 0.19 (0.19 to 0.20) | 0.19 (0.05 to 0.33) |
| 7 | 43 | 0.08 (0.06 to 0.12) | 0.16 (0.04 to 0.28) | 0.21 (0.21 to 0.21) | 0.23 (0.07 to 0.41) |
| 8 | 43 | 0.12 (0.09 to 0.21) | 0.14 (0.04 to 0.33) | 0.22 (0.22 to 0.23) | 0.28 (0.10 to 0.47) |
| 9 | 43 | 0.26 (0.13 to 0.47) | 0.26 (0.05 to 0.49) | 0.24 (0.23 to 0.24) | 0.16 (0.06 to 0.30) |
| 10 | 43 | 0.75 (0.61 to 0.87) | 0.56 (0.42 to 0.72) | 0.27 (0.26 to 0.27) | 0.16 (0.04 to 0.26) |

Reliability for P(at least 30 minutes late) at T-60, 10 equal-count bins from the lowest predicted probability to the highest:

| Bin | Model n | Model predicted | Model observed | Historical predicted | Historical observed |
|---|---|---|---|---|---|
| 1 | 43 | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.06) | 0.02 (0.02 to 0.02) | 0.12 (0.02 to 0.24) |
| 2 | 43 | 0.00 (0.00 to 0.00) | 0.05 (0.00 to 0.10) | 0.03 (0.03 to 0.03) | 0.00 (0.00 to 0.08) |
| 3 | 43 | 0.00 (0.00 to 0.00) | 0.05 (0.00 to 0.13) | 0.03 (0.03 to 0.03) | 0.09 (0.00 to 0.17) |
| 4 | 43 | 0.00 (0.00 to 0.00) | 0.05 (0.00 to 0.17) | 0.04 (0.03 to 0.04) | 0.05 (0.00 to 0.15) |
| 5 | 43 | 0.00 (0.00 to 0.01) | 0.00 (0.00 to 0.03) | 0.04 (0.04 to 0.05) | 0.00 (0.00 to 0.00) |
| 6 | 43 | 0.01 (0.01 to 0.01) | 0.02 (0.00 to 0.05) | 0.06 (0.06 to 0.06) | 0.05 (0.00 to 0.18) |
| 7 | 43 | 0.01 (0.01 to 0.02) | 0.02 (0.00 to 0.08) | 0.07 (0.07 to 0.07) | 0.09 (0.00 to 0.23) |
| 8 | 43 | 0.02 (0.01 to 0.03) | 0.00 (0.00 to 0.04) | 0.08 (0.08 to 0.08) | 0.02 (0.00 to 0.11) |
| 9 | 43 | 0.03 (0.02 to 0.06) | 0.02 (0.00 to 0.14) | 0.08 (0.08 to 0.09) | 0.02 (0.00 to 0.06) |
| 10 | 43 | 0.26 (0.13 to 0.46) | 0.23 (0.02 to 0.53) | 0.10 (0.10 to 0.10) | 0.00 (0.00 to 0.03) |

Reliability for P(at least 60 minutes late) at T-60, 10 equal-count bins from the lowest predicted probability to the highest:

| Bin | Model n | Model predicted | Model observed | Historical predicted | Historical observed |
|---|---|---|---|---|---|
| 1 | 43 | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) |
| 2 | 43 | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) |
| 3 | 43 | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) |
| 4 | 43 | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) | 0.01 (0.01 to 0.01) | 0.00 (0.00 to 0.00) |
| 5 | 43 | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) | 0.01 (0.01 to 0.01) | 0.00 (0.00 to 0.00) |
| 6 | 43 | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) | 0.01 (0.01 to 0.01) | 0.00 (0.00 to 0.00) |
| 7 | 43 | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) | 0.01 (0.01 to 0.01) | 0.00 (0.00 to 0.00) |
| 8 | 43 | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) | 0.01 (0.01 to 0.01) | 0.02 (0.00 to 0.08) |
| 9 | 43 | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) | 0.01 (0.01 to 0.01) | 0.00 (0.00 to 0.00) |
| 10 | 43 | 0.04 (0.01 to 0.08) | 0.02 (0.00 to 0.08) | 0.02 (0.02 to 0.02) | 0.00 (0.00 to 0.00) |

Interval coverage by true delay band, descriptive only. A band is selected by the outcome that the interval was built before knowing, so it cannot reach the nominal level for any forecast that is not perfect.

| Method | 0-14 | 15-29 | 30-59 | 60-119 | 120+ |
|---|---|---|---|---|---|
| Timetable | n/a | n/a | n/a | n/a | n/a |
| Persistence | 0.87 (n=354, width 18.0) | 0.54 (n=57, width 18.0) | 0.22 (n=18, width 18.0) | 0.00 (n=1, width 18.0) | n/a |
| Historical | 0.91 (n=354, width 22.3) | 0.70 (n=57, width 23.3) | 0.00 (n=18, width 20.9) | 0.00 (n=1, width 31.5) | n/a |
| Model, global CQR | 0.84 (n=354, width 14.9) | 0.51 (n=57, width 17.0) | 0.39 (n=18, width 25.4) | 0.00 (n=1, width 46.7) | n/a |
| Model, Mondrian CQR | 0.88 (n=354, width 15.7) | 0.51 (n=57, width 17.7) | 0.39 (n=18, width 26.1) | 0.00 (n=1, width 47.6) | n/a |

## bhm-eus, T-15 minutes, test block

| Method | MAE | 80% coverage | Mean width | Brier (5 bands) | Log loss |
|---|---|---|---|---|---|
| Timetable | 8.74 (7.38 to 10.21) | n/a | n/a | 0.353 (0.240 to 0.455) | 6.10 |
| Persistence | 4.13 (3.12 to 5.19) | 0.844 (0.815 to 0.874) | 12.0 | 0.265 (0.173 to 0.346) | 4.58 |
| Historical | 6.94 (5.56 to 8.42) | 0.844 (0.782 to 0.902) | 22.4 | 0.304 (0.220 to 0.391) | 0.59 |
| Model, global CQR | 3.54 (2.83 to 4.46) | 0.809 (0.778 to 0.846) | 11.1 | 0.199 (0.139 to 0.260) | 0.40 |
| Model, Mondrian CQR | 3.54 (2.83 to 4.46) | 0.840 (0.817 to 0.864) | 11.7 | 0.199 (0.139 to 0.260) | 0.40 |

Against the bar, persistence (lowest validation MAE), the model's MAE is -0.59 minutes (-0.94 to -0.21, paired bootstrap over service dates). The interval excludes zero.

Interval coverage by observed lateness at T, which is what the conformal guarantee covers:

| Method | not_started | 0-4 | 5-14 | 15+ |
|---|---|---|---|---|
| Timetable | n/a | n/a | n/a | n/a |
| Persistence | n/a | 0.83 (n=238, width 12.0) | 0.88 (n=152, width 12.0) | 0.78 (n=40, width 12.0) |
| Historical | n/a | 0.82 (n=238, width 21.5) | 0.98 (n=152, width 23.5) | 0.50 (n=40, width 23.5) |
| Model, global CQR | n/a | 0.77 (n=238, width 9.3) | 0.86 (n=152, width 11.7) | 0.85 (n=40, width 19.3) |
| Model, Mondrian CQR | n/a | 0.82 (n=238, width 9.9) | 0.86 (n=152, width 11.7) | 0.88 (n=40, width 21.9) |

Threshold reliability, derived from the band probabilities. Brier is the binary Brier score and ECE the expected calibration error over 10 equal-count bins. Historical band frequencies are the bar.

| Threshold | Share of rows at or above | Method | Brier | ECE |
|---|---|---|---|---|
| 15 minutes | 0.177 | Model, after threshold calibration | 0.090 (0.061 to 0.122) | 0.049 (0.040 to 0.094) |
| 15 minutes | 0.177 | Model, before threshold calibration | 0.088 (0.059 to 0.120) | 0.050 (0.034 to 0.093) |
| 15 minutes | 0.177 | Historical | 0.146 (0.108 to 0.186) | 0.049 (0.044 to 0.136) |
| 30 minutes | 0.044 | Model, after threshold calibration | 0.023 (0.006 to 0.044) | 0.019 (0.007 to 0.042) |
| 30 minutes | 0.044 | Model, before threshold calibration | 0.022 (0.005 to 0.043) | 0.015 (0.005 to 0.039) |
| 30 minutes | 0.044 | Historical | 0.044 (0.018 to 0.073) | 0.049 (0.038 to 0.080) |
| 60 minutes | 0.002 | Model, after threshold calibration | 0.002 (0.000 to 0.006) | 0.001 (0.000 to 0.006) |
| 60 minutes | 0.002 | Model, before threshold calibration | 0.002 (0.000 to 0.006) | 0.001 (0.000 to 0.006) |
| 60 minutes | 0.002 | Historical | 0.002 (0.000 to 0.008) | 0.008 (0.008 to 0.014) |

Reliability for P(at least 15 minutes late) at T-15, 10 equal-count bins from the lowest predicted probability to the highest:

| Bin | Model n | Model predicted | Model observed | Historical predicted | Historical observed |
|---|---|---|---|---|---|
| 1 | 43 | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) | 0.09 (0.09 to 0.10) | 0.14 (0.03 to 0.30) |
| 2 | 43 | 0.00 (0.00 to 0.00) | 0.02 (0.00 to 0.07) | 0.12 (0.12 to 0.12) | 0.12 (0.00 to 0.41) |
| 3 | 43 | 0.01 (0.00 to 0.01) | 0.00 (0.00 to 0.06) | 0.13 (0.13 to 0.14) | 0.16 (0.02 to 0.30) |
| 4 | 43 | 0.01 (0.01 to 0.02) | 0.05 (0.00 to 0.09) | 0.14 (0.14 to 0.15) | 0.21 (0.05 to 0.39) |
| 5 | 43 | 0.02 (0.01 to 0.02) | 0.02 (0.00 to 0.27) | 0.17 (0.16 to 0.19) | 0.09 (0.05 to 0.25) |
| 6 | 43 | 0.03 (0.02 to 0.04) | 0.26 (0.00 to 0.37) | 0.19 (0.19 to 0.20) | 0.21 (0.06 to 0.33) |
| 7 | 43 | 0.05 (0.04 to 0.07) | 0.16 (0.05 to 0.27) | 0.21 (0.21 to 0.21) | 0.23 (0.07 to 0.41) |
| 8 | 43 | 0.08 (0.06 to 0.13) | 0.14 (0.03 to 0.24) | 0.22 (0.22 to 0.23) | 0.28 (0.10 to 0.47) |
| 9 | 43 | 0.25 (0.10 to 0.47) | 0.28 (0.04 to 0.74) | 0.24 (0.23 to 0.24) | 0.16 (0.06 to 0.31) |
| 10 | 43 | 0.84 (0.66 to 0.95) | 0.84 (0.61 to 0.95) | 0.27 (0.26 to 0.27) | 0.16 (0.04 to 0.24) |

Reliability for P(at least 30 minutes late) at T-15, 10 equal-count bins from the lowest predicted probability to the highest:

| Bin | Model n | Model predicted | Model observed | Historical predicted | Historical observed |
|---|---|---|---|---|---|
| 1 | 43 | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) | 0.02 (0.02 to 0.02) | 0.12 (0.02 to 0.24) |
| 2 | 43 | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) | 0.03 (0.03 to 0.03) | 0.00 (0.00 to 0.08) |
| 3 | 43 | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) | 0.03 (0.03 to 0.03) | 0.09 (0.00 to 0.18) |
| 4 | 43 | 0.00 (0.00 to 0.00) | 0.05 (0.00 to 0.14) | 0.04 (0.03 to 0.04) | 0.05 (0.00 to 0.15) |
| 5 | 43 | 0.00 (0.00 to 0.01) | 0.02 (0.00 to 0.11) | 0.04 (0.04 to 0.05) | 0.00 (0.00 to 0.00) |
| 6 | 43 | 0.01 (0.00 to 0.01) | 0.00 (0.00 to 0.06) | 0.06 (0.06 to 0.06) | 0.05 (0.00 to 0.18) |
| 7 | 43 | 0.01 (0.01 to 0.01) | 0.07 (0.00 to 0.13) | 0.07 (0.07 to 0.07) | 0.09 (0.00 to 0.23) |
| 8 | 43 | 0.01 (0.01 to 0.02) | 0.00 (0.00 to 0.10) | 0.08 (0.08 to 0.08) | 0.02 (0.00 to 0.11) |
| 9 | 43 | 0.03 (0.02 to 0.04) | 0.02 (0.00 to 0.07) | 0.08 (0.08 to 0.09) | 0.02 (0.00 to 0.06) |
| 10 | 43 | 0.24 (0.10 to 0.46) | 0.28 (0.07 to 0.60) | 0.10 (0.10 to 0.10) | 0.00 (0.00 to 0.03) |

Reliability for P(at least 60 minutes late) at T-15, 10 equal-count bins from the lowest predicted probability to the highest:

| Bin | Model n | Model predicted | Model observed | Historical predicted | Historical observed |
|---|---|---|---|---|---|
| 1 | 43 | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) |
| 2 | 43 | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) |
| 3 | 43 | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) |
| 4 | 43 | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) | 0.01 (0.01 to 0.01) | 0.00 (0.00 to 0.00) |
| 5 | 43 | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) | 0.01 (0.01 to 0.01) | 0.00 (0.00 to 0.00) |
| 6 | 43 | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) | 0.01 (0.01 to 0.01) | 0.00 (0.00 to 0.00) |
| 7 | 43 | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) | 0.01 (0.01 to 0.01) | 0.00 (0.00 to 0.00) |
| 8 | 43 | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) | 0.01 (0.01 to 0.01) | 0.02 (0.00 to 0.08) |
| 9 | 43 | 0.00 (0.00 to 0.00) | 0.00 (0.00 to 0.00) | 0.01 (0.01 to 0.01) | 0.00 (0.00 to 0.00) |
| 10 | 43 | 0.03 (0.01 to 0.06) | 0.02 (0.00 to 0.08) | 0.02 (0.02 to 0.02) | 0.00 (0.00 to 0.00) |

Interval coverage by true delay band, descriptive only. A band is selected by the outcome that the interval was built before knowing, so it cannot reach the nominal level for any forecast that is not perfect.

| Method | 0-14 | 15-29 | 30-59 | 60-119 | 120+ |
|---|---|---|---|---|---|
| Timetable | n/a | n/a | n/a | n/a | n/a |
| Persistence | 0.90 (n=354, width 12.0) | 0.65 (n=57, width 12.0) | 0.39 (n=18, width 12.0) | 0.00 (n=1, width 12.0) | n/a |
| Historical | 0.91 (n=354, width 22.3) | 0.70 (n=57, width 23.3) | 0.00 (n=18, width 20.9) | 0.00 (n=1, width 31.5) | n/a |
| Model, global CQR | 0.84 (n=354, width 10.1) | 0.72 (n=57, width 12.8) | 0.56 (n=18, width 22.6) | 0.00 (n=1, width 35.4) | n/a |
| Model, Mondrian CQR | 0.88 (n=354, width 10.6) | 0.72 (n=57, width 13.8) | 0.56 (n=18, width 24.4) | 0.00 (n=1, width 38.0) | n/a |

## Limitations

These results cover 2 routes: ast-bhm, bhm-eus. BHM to MAN and AST to EUS via BHM are not in the model, so nothing here says it transfers to them. The history runs from 2026-04-15 to 2026-10-15, which is summer and the first weeks of autumn, so there is no winter, leaf-fall or storm season, and a seasonal shift would change both the delays and the calibration. The 15 or more minutes late lateness group is small. On ast-bhm it held 0 calibration rows at T-120, 0 calibration rows at T-60, 1 calibration rows at T-15 and 0 test rows at T-120, 0 test rows at T-60, 0 test rows at T-15. The model's MAE gain over the bar is not statistically significant at T-120, T-60. On bhm-eus it held 17 calibration rows at T-120, 20 calibration rows at T-60, 34 calibration rows at T-15 and 16 test rows at T-120, 26 test rows at T-60, 40 test rows at T-15. The model's MAE gain over the bar is not statistically significant at T-120. Offsets for a small group are noisy and are merged into a neighbour below 30 rows. The test block is 7 days, so the bootstrap intervals are wide, and the 60 minute and longer delays are too rare for the 60 minute threshold table to say much. The conformal guarantee holds on average within each cell if the days are exchangeable, which a time split only approximates. The delay bands are provisional until the per-operator compensation rules are checked.

<!-- connections:start -->

## Connections: AST to BHM inbound, BHM to EUS outbound

This section scores the connection engine out of sample. A pair is one AST to BHM service with one BHM to EUS service that leaves between 10 and 30 minutes after it is due at Birmingham. The minimum connection time of 10 minutes is assumed, not a published figure. The AST to BHM days split as 159 train, 14 validation and 7 test. There were 18270 training pairs, 1746 validation pairs and 813 test pairs, of which 748 test pairs had a known outcome. Only test pairs are scored below. The delay model, the tail model, the outbound delay history and the baseline are all fitted on training dates, and the validation pairs are used only to choose alert thresholds.

Each forecast is made 60 or 15 minutes before the outbound is due to leave, with the exact number of minutes to the inbound's scheduled arrival at that moment given to the model. If the inbound had already arrived by then, its delay is known and only the outbound is uncertain. If it was still approaching, the model was given the exact minutes. If its scheduled arrival had passed or was within the model's shortest trained horizon of 5 minutes, the model saw the true state at that moment and, for an overdue train, the draws were limited to delays at least as large as the time already elapsed. The baseline is the historical miss frequency by scheduled gap bucket (5 minute steps above the minimum time, the last bucket open), fitted on 16269 training pairs with an overall miss rate of 6.4%.

| Forecast time | Pairs | Realised miss rate | Mean predicted | Model Brier | Baseline Brier | Model ECE | Baseline ECE | 80% range coverage |
|---|---|---|---|---|---|---|---|---|
| T-60 before outbound | 748 | 6.0% | 7.9% | 0.0525 (0.0395 to 0.0687) | 0.0528 (0.0382 to 0.0706) | 0.0235 (0.0152 to 0.0448) | 0.0248 (0.0121 to 0.0333) | 0.874 (0.823 to 0.920) of 737 |
| T-15 before outbound | 748 | 6.0% | 8.5% | 0.0406 (0.0287 to 0.0566) | 0.0528 (0.0382 to 0.0706) | 0.0285 (0.0156 to 0.0466) | 0.0248 (0.0121 to 0.0333) | 0.877 (0.825 to 0.923) of 737 |

Brackets are 95% bootstrap intervals over service dates. Lower Brier and ECE are better. The coverage column is the share of pairs whose realised arrival at Euston fell inside the predicted 80% range, counting only pairs that reached Euston.

At T-60 before outbound, by what was known about the inbound: approaching 748 pairs (miss rate 6.0%, mean predicted 7.9%).

Reliability for P(miss) at T-60 before outbound, 10 equal-count bins from the lowest predicted probability to the highest:

| Bin | n | Model predicted | Model observed | Baseline predicted | Baseline observed |
|---|---|---|---|---|---|
| 1 | 75 | 0.001 | 0.000 (0.000 to 0.000) | 0.004 | 0.000 (0.000 to 0.000) |
| 2 | 75 | 0.003 | 0.013 (0.000 to 0.046) | 0.004 | 0.000 (0.000 to 0.000) |
| 3 | 75 | 0.004 | 0.000 (0.000 to 0.017) | 0.004 | 0.000 (0.000 to 0.000) |
| 4 | 75 | 0.005 | 0.000 (0.000 to 0.000) | 0.006 | 0.013 (0.000 to 0.018) |
| 5 | 75 | 0.020 | 0.027 (0.000 to 0.060) | 0.036 | 0.067 (0.000 to 0.089) |
| 6 | 75 | 0.033 | 0.040 (0.000 to 0.092) | 0.048 | 0.080 (0.013 to 0.109) |
| 7 | 75 | 0.060 | 0.053 (0.013 to 0.121) | 0.048 | 0.027 (0.000 to 0.113) |
| 8 | 75 | 0.129 | 0.067 (0.012 to 0.189) | 0.119 | 0.160 (0.041 to 0.211) |
| 9 | 74 | 0.204 | 0.162 (0.036 to 0.250) | 0.180 | 0.149 (0.080 to 0.277) |
| 10 | 74 | 0.336 | 0.243 (0.187 to 0.300) | 0.180 | 0.108 (0.076 to 0.277) |

At T-15 before outbound, by what was known about the inbound: approaching 1 pairs (miss rate 100.0%, mean predicted 61.1%), arrived 448 pairs (miss rate 0.0%, mean predicted 0.0%), due or late 299 pairs (miss rate 14.7%, mean predicted 21.1%).

Reliability for P(miss) at T-15 before outbound, 10 equal-count bins from the lowest predicted probability to the highest:

| Bin | n | Model predicted | Model observed | Baseline predicted | Baseline observed |
|---|---|---|---|---|---|
| 1 | 75 | 0.000 | 0.000 (0.000 to 0.000) | 0.004 | 0.000 (0.000 to 0.000) |
| 2 | 75 | 0.000 | 0.000 (0.000 to 0.000) | 0.004 | 0.000 (0.000 to 0.000) |
| 3 | 75 | 0.000 | 0.000 (0.000 to 0.000) | 0.004 | 0.000 (0.000 to 0.000) |
| 4 | 75 | 0.000 | 0.000 (0.000 to 0.000) | 0.006 | 0.013 (0.000 to 0.018) |
| 5 | 75 | 0.000 | 0.000 (0.000 to 0.000) | 0.036 | 0.067 (0.000 to 0.089) |
| 6 | 75 | 0.001 | 0.000 (0.000 to 0.051) | 0.048 | 0.080 (0.013 to 0.109) |
| 7 | 75 | 0.077 | 0.093 (0.027 to 0.176) | 0.048 | 0.027 (0.000 to 0.113) |
| 8 | 75 | 0.136 | 0.067 (0.014 to 0.138) | 0.119 | 0.160 (0.041 to 0.211) |
| 9 | 74 | 0.201 | 0.068 (0.000 to 0.133) | 0.180 | 0.149 (0.080 to 0.277) |
| 10 | 74 | 0.446 | 0.378 (0.268 to 0.470) | 0.180 | 0.108 (0.076 to 0.277) |

### Alert thresholds

The cost of a false alert is set to 1 and the cost of a missed alert to 5. Both are assumptions in config/alerts.yaml and should be revisited with product input. Each threshold is the one with the lowest expected cost per pair on validation pairs, on a grid of 0.01, and the test pairs were not used to choose it. An alert fires when the probability is at or above the threshold.

Likely to miss your connection, on P(miss). The chosen tau is 0.22. Test results with both forecast times pooled:

| Policy | Threshold | Alerts | Precision | Recall | Alert rate | Cost per pair |
|---|---|---|---|---|---|---|
| Chosen tau | 0.22 | 197 of 1496 | 0.264 | 0.578 | 0.132 | 0.224 |
| Always alert | 0.00 | 1496 of 1496 | 0.060 | 1.000 | 1.000 | 0.940 |
| Never alert | 1.01 | 0 of 1496 | n/a | 0.000 | 0.000 | 0.301 |

On validation the chosen tau cost 0.225 per pair against 0.933 for always alerting and 0.335 for never alerting. The event happened in 6.7% of validation pairs and 6.0% of test pairs.

At T-60 before outbound the chosen tau gave precision 0.218, recall 0.489, alert rate 0.135 and cost 0.259 per pair over 748 pairs.
At T-15 before outbound the chosen tau gave precision 0.312, recall 0.667, alert rate 0.128 and cost 0.189 per pair over 748 pairs.

Sensitivity of tau to the missed-alert cost: missed-alert cost 5: tau 0.22 (a calibrated forecast would give 0.17), missed-alert cost 3: tau 0.33 (a calibrated forecast would give 0.25), missed-alert cost 10: tau 0.08 (a calibrated forecast would give 0.09).

You may be owed Delay Repay, on P(final arrival at least 15 minutes late). The chosen tau2 is 0.26. Test results with both forecast times pooled:

| Policy | Threshold | Alerts | Precision | Recall | Alert rate | Cost per pair |
|---|---|---|---|---|---|---|
| Chosen tau2 | 0.26 | 350 of 1474 | 0.357 | 0.391 | 0.237 | 0.814 |
| Always alert | 0.00 | 1474 of 1474 | 0.217 | 1.000 | 1.000 | 0.783 |
| Never alert | 1.01 | 0 of 1474 | n/a | 0.000 | 0.000 | 1.085 |

On validation the chosen tau2 cost 0.454 per pair against 0.866 for always alerting and 0.670 for never alerting. The event happened in 13.4% of validation pairs and 21.7% of test pairs, so a threshold chosen on the quieter validation weeks did not beat always alerting on test.

At T-60 before outbound the chosen tau2 gave precision 0.358, recall 0.394, alert rate 0.239 and cost 0.811 per pair over 737 pairs.
At T-15 before outbound the chosen tau2 gave precision 0.356, recall 0.388, alert rate 0.236 and cost 0.817 per pair over 737 pairs.

Sensitivity of tau2 to the missed-alert cost: missed-alert cost 5: tau2 0.26 (a calibrated forecast would give 0.17), missed-alert cost 3: tau2 0.32 (a calibrated forecast would give 0.25), missed-alert cost 10: tau2 0.22 (a calibrated forecast would give 0.09).

### Limitations

The minimum connection time is assumed, and the miss probabilities move with it. The inbound and outbound delays are drawn independently, so shared causes such as a signalling failure or a strike day are not modelled. In the history the two delays are only weakly correlated (0.11), and for the tightest changes, 10 to 14 minutes, independence overstated the miss rate (18.4% realised against 20.3% with the pairing broken), so the bias is not in the optimistic direction there. The history is summer and the first weeks of autumn, with no winter or storm season. Only two routes are in the model, and the connection test uses one hub and one pair of routes. The test block is 7 days, so the intervals are wide, and a small number of pairs drives the tails of every score. The Delay Repay alert counts a stranded passenger as at least 15 minutes late, which is an assumption. The costs behind both thresholds are assumed.

<!-- connections:end -->
