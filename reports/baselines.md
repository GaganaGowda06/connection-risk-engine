# Baseline scores

Delay is the minutes between scheduled and actual arrival at the route destination. Training covers 2026-09-18 to 2026-09-28 (8 days), validation 2026-09-30 to 2026-10-06, and test 2026-10-08 to 2026-10-14, with one day dropped between blocks. Intervals are nominal 80% intervals. Brackets are 95% bootstrap intervals from 1000 resamples of whole service dates. Cancelled services have no delay and are not scored.

The delay bands are 0-14, 15-29, 30-59, 60-119, 120+ minutes, and they are provisional until the per-operator rules are checked in Stage 5. Timetable and persistence give a hard band with probability 1, so their log loss is dominated by the 1e-15 clip. The historical band frequencies are smoothed by half a count per band. Only the historical baseline has an interval.

## T-120 minutes

| Baseline | Block | Rows | MAE | RMSE | 80% coverage | Mean width | Brier | Log loss |
|---|---|---|---|---|---|---|---|---|
| Timetable | validation | 488 | 6.30 (5.02 to 7.76) | 10.23 (8.37 to 11.99) | n/a | n/a | 0.201 (0.124 to 0.286) | 3.47 |
| Timetable | test | 432 | 8.88 (7.59 to 10.14) | 13.49 (11.02 to 15.40) | n/a | n/a | 0.366 (0.258 to 0.460) | 6.32 |
| Persistence | validation | 488 | 6.25 (4.94 to 7.76) | 10.42 (8.35 to 12.39) | n/a | n/a | 0.213 (0.118 to 0.314) | 3.68 |
| Persistence | test | 432 | 7.61 (6.09 to 8.90) | 11.38 (8.83 to 13.30) | n/a | n/a | 0.380 (0.262 to 0.469) | 6.56 |
| Historical | validation | 488 | 5.30 (4.43 to 6.25) | 8.68 (7.10 to 10.13) | 0.811 (0.755 to 0.863) | 16.5 | 0.185 (0.123 to 0.253) | 0.41 |
| Historical | test | 432 | 7.23 (6.00 to 8.40) | 11.69 (9.12 to 13.81) | 0.738 (0.690 to 0.794) | 16.6 | 0.327 (0.237 to 0.405) | 0.68 |

At T-120 the bar to beat for the point forecast is historical, the lowest validation MAE, which scores 7.23 (6.00 to 8.40) minutes on test. For the delay bands it is historical, the lowest validation Brier score. A model has to beat these on the test block with an interval that does not overlap theirs to count as an improvement.

## T-60 minutes

| Baseline | Block | Rows | MAE | RMSE | 80% coverage | Mean width | Brier | Log loss |
|---|---|---|---|---|---|---|---|---|
| Timetable | validation | 488 | 6.30 (5.02 to 7.76) | 10.23 (8.37 to 11.99) | n/a | n/a | 0.201 (0.124 to 0.286) | 3.47 |
| Timetable | test | 432 | 8.88 (7.59 to 10.14) | 13.49 (11.02 to 15.40) | n/a | n/a | 0.366 (0.258 to 0.460) | 6.32 |
| Persistence | validation | 488 | 4.91 (3.95 to 5.91) | 8.36 (6.87 to 9.49) | n/a | n/a | 0.201 (0.125 to 0.288) | 3.47 |
| Persistence | test | 432 | 6.73 (5.41 to 7.92) | 10.46 (7.86 to 12.37) | n/a | n/a | 0.361 (0.261 to 0.433) | 6.24 |
| Historical | validation | 488 | 5.30 (4.43 to 6.25) | 8.68 (7.10 to 10.13) | 0.811 (0.755 to 0.863) | 16.5 | 0.185 (0.123 to 0.253) | 0.41 |
| Historical | test | 432 | 7.23 (6.00 to 8.40) | 11.69 (9.12 to 13.81) | 0.738 (0.690 to 0.794) | 16.6 | 0.327 (0.237 to 0.405) | 0.68 |

At T-60 the bar to beat for the point forecast is persistence, the lowest validation MAE, which scores 6.73 (5.41 to 7.92) minutes on test. For the delay bands it is historical, the lowest validation Brier score. A model has to beat these on the test block with an interval that does not overlap theirs to count as an improvement.

## T-15 minutes

| Baseline | Block | Rows | MAE | RMSE | 80% coverage | Mean width | Brier | Log loss |
|---|---|---|---|---|---|---|---|---|
| Timetable | validation | 488 | 6.30 (5.02 to 7.76) | 10.23 (8.37 to 11.99) | n/a | n/a | 0.201 (0.124 to 0.286) | 3.47 |
| Timetable | test | 432 | 8.88 (7.59 to 10.14) | 13.49 (11.02 to 15.40) | n/a | n/a | 0.366 (0.258 to 0.460) | 6.32 |
| Persistence | validation | 488 | 3.31 (2.70 to 3.89) | 6.31 (4.68 to 7.65) | n/a | n/a | 0.119 (0.071 to 0.161) | 2.05 |
| Persistence | test | 432 | 4.25 (3.20 to 5.23) | 7.89 (5.10 to 9.91) | n/a | n/a | 0.278 (0.188 to 0.351) | 4.80 |
| Historical | validation | 488 | 5.30 (4.43 to 6.25) | 8.68 (7.10 to 10.13) | 0.811 (0.755 to 0.863) | 16.5 | 0.185 (0.123 to 0.253) | 0.41 |
| Historical | test | 432 | 7.23 (6.00 to 8.40) | 11.69 (9.12 to 13.81) | 0.738 (0.690 to 0.794) | 16.6 | 0.327 (0.237 to 0.405) | 0.68 |

At T-15 the bar to beat for the point forecast is persistence, the lowest validation MAE, which scores 4.25 (3.20 to 5.23) minutes on test. For the delay bands it is persistence, the lowest validation Brier score. A model has to beat these on the test block with an interval that does not overlap theirs to count as an improvement.
