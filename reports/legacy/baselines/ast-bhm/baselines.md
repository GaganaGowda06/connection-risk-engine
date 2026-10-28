# Baseline scores

Delay is the minutes between scheduled and actual arrival at the route destination. Training covers 2026-04-15 to 2026-09-22 (159 days), validation 2026-09-24 to 2026-10-07, and test 2026-10-09 to 2026-10-15, with one day dropped between blocks. Intervals are nominal 80% intervals. Brackets are 95% bootstrap intervals from 1000 resamples of whole service dates. Cancelled services have no delay and are not scored.

The delay bands are 0-14, 15-29, 30-59, 60-119, 120+ minutes, and they are provisional until the per-operator rules are checked in Stage 5. Timetable and persistence give a hard band with probability 1, so their log loss is dominated by the 1e-15 clip. The historical band frequencies are smoothed by half a count per band. The timetable has no interval. The persistence interval is its point plus the 10th and 90th percentile of training residuals per horizon.

## T-120 minutes

| Baseline | Block | Rows | MAE | RMSE | 80% coverage | Mean width | Brier | Log loss |
|---|---|---|---|---|---|---|---|---|
| Timetable | validation | 1180 | 2.24 (2.08 to 2.41) | 3.34 (3.09 to 3.67) | n/a | n/a | 0.007 (0.001 to 0.015) | 0.12 |
| Timetable | test | 589 | 2.48 (2.01 to 2.95) | 3.66 (3.01 to 4.26) | n/a | n/a | 0.007 (0.000 to 0.020) | 0.12 |
| Persistence | validation | 1180 | 2.24 (2.08 to 2.41) | 3.34 (3.09 to 3.67) | 0.827 (0.799 to 0.851) | 6.0 | 0.007 (0.001 to 0.015) | 0.12 |
| Persistence | test | 589 | 2.48 (2.01 to 2.95) | 3.66 (3.01 to 4.26) | 0.815 (0.756 to 0.868) | 6.0 | 0.007 (0.000 to 0.020) | 0.12 |
| Historical | validation | 1180 | 2.11 (2.01 to 2.25) | 3.01 (2.79 to 3.31) | 0.849 (0.821 to 0.873) | 6.9 | 0.007 (0.002 to 0.015) | 0.03 |
| Historical | test | 589 | 2.31 (1.95 to 2.63) | 3.21 (2.71 to 3.65) | 0.832 (0.791 to 0.865) | 6.9 | 0.007 (0.000 to 0.020) | 0.03 |

At T-120 the bar to beat for the point forecast is historical, the lowest validation MAE, which scores 2.31 (1.95 to 2.63) minutes on test. For the delay bands it is timetable, the lowest validation Brier score. A model has to beat these on the test block with an interval that does not overlap theirs to count as an improvement.

## T-90 minutes

| Baseline | Block | Rows | MAE | RMSE | 80% coverage | Mean width | Brier | Log loss |
|---|---|---|---|---|---|---|---|---|
| Timetable | validation | 1180 | 2.24 (2.08 to 2.41) | 3.34 (3.09 to 3.67) | n/a | n/a | 0.007 (0.001 to 0.015) | 0.12 |
| Timetable | test | 589 | 2.48 (2.01 to 2.95) | 3.66 (3.01 to 4.26) | n/a | n/a | 0.007 (0.000 to 0.020) | 0.12 |
| Persistence | validation | 1180 | 2.24 (2.08 to 2.41) | 3.34 (3.09 to 3.67) | 0.827 (0.799 to 0.851) | 6.0 | 0.007 (0.001 to 0.015) | 0.12 |
| Persistence | test | 589 | 2.48 (2.01 to 2.95) | 3.66 (3.01 to 4.26) | 0.815 (0.756 to 0.868) | 6.0 | 0.007 (0.000 to 0.020) | 0.12 |
| Historical | validation | 1180 | 2.11 (2.01 to 2.25) | 3.01 (2.79 to 3.31) | 0.849 (0.821 to 0.873) | 6.9 | 0.007 (0.002 to 0.015) | 0.03 |
| Historical | test | 589 | 2.31 (1.95 to 2.63) | 3.21 (2.71 to 3.65) | 0.832 (0.791 to 0.865) | 6.9 | 0.007 (0.000 to 0.020) | 0.03 |

At T-90 the bar to beat for the point forecast is historical, the lowest validation MAE, which scores 2.31 (1.95 to 2.63) minutes on test. For the delay bands it is timetable, the lowest validation Brier score. A model has to beat these on the test block with an interval that does not overlap theirs to count as an improvement.

## T-60 minutes

| Baseline | Block | Rows | MAE | RMSE | 80% coverage | Mean width | Brier | Log loss |
|---|---|---|---|---|---|---|---|---|
| Timetable | validation | 1180 | 2.24 (2.08 to 2.41) | 3.34 (3.09 to 3.67) | n/a | n/a | 0.007 (0.001 to 0.015) | 0.12 |
| Timetable | test | 589 | 2.48 (2.01 to 2.95) | 3.66 (3.01 to 4.26) | n/a | n/a | 0.007 (0.000 to 0.020) | 0.12 |
| Persistence | validation | 1180 | 2.24 (2.08 to 2.41) | 3.34 (3.10 to 3.67) | 0.826 (0.798 to 0.850) | 6.0 | 0.007 (0.001 to 0.015) | 0.12 |
| Persistence | test | 589 | 2.48 (2.01 to 2.95) | 3.66 (3.01 to 4.26) | 0.815 (0.756 to 0.868) | 6.0 | 0.007 (0.000 to 0.020) | 0.12 |
| Historical | validation | 1180 | 2.11 (2.01 to 2.25) | 3.01 (2.79 to 3.31) | 0.849 (0.821 to 0.873) | 6.9 | 0.007 (0.002 to 0.015) | 0.03 |
| Historical | test | 589 | 2.31 (1.95 to 2.63) | 3.21 (2.71 to 3.65) | 0.832 (0.791 to 0.865) | 6.9 | 0.007 (0.000 to 0.020) | 0.03 |

At T-60 the bar to beat for the point forecast is historical, the lowest validation MAE, which scores 2.31 (1.95 to 2.63) minutes on test. For the delay bands it is timetable, the lowest validation Brier score. A model has to beat these on the test block with an interval that does not overlap theirs to count as an improvement.

## T-30 minutes

| Baseline | Block | Rows | MAE | RMSE | 80% coverage | Mean width | Brier | Log loss |
|---|---|---|---|---|---|---|---|---|
| Timetable | validation | 1180 | 2.24 (2.08 to 2.41) | 3.34 (3.09 to 3.67) | n/a | n/a | 0.007 (0.001 to 0.015) | 0.12 |
| Timetable | test | 589 | 2.48 (2.01 to 2.95) | 3.66 (3.01 to 4.26) | n/a | n/a | 0.007 (0.000 to 0.020) | 0.12 |
| Persistence | validation | 1180 | 2.30 (2.14 to 2.48) | 3.41 (3.14 to 3.72) | 0.873 (0.851 to 0.893) | 7.0 | 0.005 (0.000 to 0.013) | 0.09 |
| Persistence | test | 589 | 2.54 (2.05 to 3.03) | 3.71 (3.07 to 4.31) | 0.862 (0.826 to 0.897) | 7.0 | 0.007 (0.000 to 0.020) | 0.12 |
| Historical | validation | 1180 | 2.11 (2.01 to 2.25) | 3.01 (2.79 to 3.31) | 0.849 (0.821 to 0.873) | 6.9 | 0.007 (0.002 to 0.015) | 0.03 |
| Historical | test | 589 | 2.31 (1.95 to 2.63) | 3.21 (2.71 to 3.65) | 0.832 (0.791 to 0.865) | 6.9 | 0.007 (0.000 to 0.020) | 0.03 |

At T-30 the bar to beat for the point forecast is historical, the lowest validation MAE, which scores 2.31 (1.95 to 2.63) minutes on test. For the delay bands it is persistence, the lowest validation Brier score. A model has to beat these on the test block with an interval that does not overlap theirs to count as an improvement.

## T-15 minutes

| Baseline | Block | Rows | MAE | RMSE | 80% coverage | Mean width | Brier | Log loss |
|---|---|---|---|---|---|---|---|---|
| Timetable | validation | 1180 | 2.24 (2.08 to 2.41) | 3.34 (3.09 to 3.67) | n/a | n/a | 0.007 (0.001 to 0.015) | 0.12 |
| Timetable | test | 589 | 2.48 (2.01 to 2.95) | 3.66 (3.01 to 4.26) | n/a | n/a | 0.007 (0.000 to 0.020) | 0.12 |
| Persistence | validation | 1180 | 2.09 (1.96 to 2.22) | 3.07 (2.85 to 3.35) | 0.851 (0.828 to 0.874) | 6.0 | 0.005 (0.000 to 0.013) | 0.09 |
| Persistence | test | 589 | 2.19 (1.83 to 2.56) | 3.25 (2.73 to 3.79) | 0.839 (0.800 to 0.877) | 6.0 | 0.007 (0.000 to 0.020) | 0.12 |
| Historical | validation | 1180 | 2.11 (2.01 to 2.25) | 3.01 (2.79 to 3.31) | 0.849 (0.821 to 0.873) | 6.9 | 0.007 (0.002 to 0.015) | 0.03 |
| Historical | test | 589 | 2.31 (1.95 to 2.63) | 3.21 (2.71 to 3.65) | 0.832 (0.791 to 0.865) | 6.9 | 0.007 (0.000 to 0.020) | 0.03 |

At T-15 the bar to beat for the point forecast is persistence, the lowest validation MAE, which scores 2.19 (1.83 to 2.56) minutes on test. For the delay bands it is persistence, the lowest validation Brier score. A model has to beat these on the test block with an interval that does not overlap theirs to count as an improvement.

## T-5 minutes

| Baseline | Block | Rows | MAE | RMSE | 80% coverage | Mean width | Brier | Log loss |
|---|---|---|---|---|---|---|---|---|
| Timetable | validation | 1180 | 2.24 (2.08 to 2.41) | 3.34 (3.09 to 3.67) | n/a | n/a | 0.007 (0.001 to 0.015) | 0.12 |
| Timetable | test | 589 | 2.48 (2.01 to 2.95) | 3.66 (3.01 to 4.26) | n/a | n/a | 0.007 (0.000 to 0.020) | 0.12 |
| Persistence | validation | 1180 | 1.66 (1.53 to 1.80) | 2.48 (2.27 to 2.71) | 0.906 (0.888 to 0.926) | 6.0 | 0.007 (0.000 to 0.015) | 0.12 |
| Persistence | test | 589 | 1.78 (1.51 to 2.05) | 2.64 (2.22 to 3.09) | 0.896 (0.861 to 0.929) | 6.0 | 0.003 (0.000 to 0.010) | 0.06 |
| Historical | validation | 1180 | 2.11 (2.01 to 2.25) | 3.01 (2.79 to 3.31) | 0.849 (0.821 to 0.873) | 6.9 | 0.007 (0.002 to 0.015) | 0.03 |
| Historical | test | 589 | 2.31 (1.95 to 2.63) | 3.21 (2.71 to 3.65) | 0.832 (0.791 to 0.865) | 6.9 | 0.007 (0.000 to 0.020) | 0.03 |

At T-5 the bar to beat for the point forecast is persistence, the lowest validation MAE, which scores 1.78 (1.51 to 2.05) minutes on test. For the delay bands it is timetable, the lowest validation Brier score. A model has to beat these on the test block with an interval that does not overlap theirs to count as an improvement.
