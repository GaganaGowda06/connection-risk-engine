# Baseline scores

Delay is the minutes between scheduled and actual arrival at the route destination. Training covers 2026-04-15 to 2026-09-22 (159 days), validation 2026-09-24 to 2026-10-07, and test 2026-10-09 to 2026-10-15, with one day dropped between blocks. Intervals are nominal 80% intervals. Brackets are 95% bootstrap intervals from 1000 resamples of whole service dates. Cancelled services have no delay and are not scored.

The delay bands are 0-14, 15-29, 30-59, 60-119, 120+ minutes, and they are provisional until the per-operator rules are checked in Stage 5. Timetable and persistence give a hard band with probability 1, so their log loss is dominated by the 1e-15 clip. The historical band frequencies are smoothed by half a count per band. The timetable has no interval. The persistence interval is its point plus the 10th and 90th percentile of training residuals per horizon.

## T-120 minutes

| Baseline | Block | Rows | MAE | RMSE | 80% coverage | Mean width | Brier | Log loss |
|---|---|---|---|---|---|---|---|---|
| Timetable | validation | 977 | 5.66 (4.78 to 6.73) | 9.10 (7.69 to 10.60) | n/a | n/a | 0.170 (0.116 to 0.235) | 2.93 |
| Timetable | test | 430 | 8.74 (7.38 to 10.21) | 13.34 (10.86 to 15.98) | n/a | n/a | 0.353 (0.240 to 0.455) | 6.10 |
| Persistence | validation | 977 | 5.57 (4.67 to 6.62) | 9.10 (7.59 to 10.66) | 0.868 (0.835 to 0.895) | 22.0 | 0.174 (0.114 to 0.245) | 3.00 |
| Persistence | test | 430 | 7.49 (5.88 to 8.86) | 11.19 (8.43 to 13.20) | 0.830 (0.764 to 0.885) | 22.0 | 0.363 (0.240 to 0.465) | 6.27 |
| Historical | validation | 977 | 5.18 (4.72 to 5.69) | 7.73 (6.70 to 8.84) | 0.816 (0.785 to 0.845) | 22.5 | 0.173 (0.135 to 0.218) | 0.38 |
| Historical | test | 430 | 6.94 (5.56 to 8.42) | 11.33 (8.80 to 13.99) | 0.844 (0.782 to 0.902) | 22.4 | 0.304 (0.220 to 0.391) | 0.59 |

At T-120 the bar to beat for the point forecast is historical, the lowest validation MAE, which scores 6.94 (5.56 to 8.42) minutes on test. For the delay bands it is timetable, the lowest validation Brier score. A model has to beat these on the test block with an interval that does not overlap theirs to count as an improvement.

## T-90 minutes

| Baseline | Block | Rows | MAE | RMSE | 80% coverage | Mean width | Brier | Log loss |
|---|---|---|---|---|---|---|---|---|
| Timetable | validation | 977 | 5.66 (4.78 to 6.73) | 9.10 (7.69 to 10.60) | n/a | n/a | 0.170 (0.116 to 0.235) | 2.93 |
| Timetable | test | 430 | 8.74 (7.38 to 10.21) | 13.34 (10.86 to 15.98) | n/a | n/a | 0.353 (0.240 to 0.455) | 6.10 |
| Persistence | validation | 977 | 5.53 (4.67 to 6.55) | 9.02 (7.57 to 10.53) | 0.875 (0.841 to 0.903) | 22.0 | 0.174 (0.110 to 0.248) | 3.00 |
| Persistence | test | 430 | 7.23 (5.82 to 8.47) | 10.88 (8.36 to 12.86) | 0.830 (0.788 to 0.868) | 22.0 | 0.358 (0.232 to 0.457) | 6.18 |
| Historical | validation | 977 | 5.18 (4.72 to 5.69) | 7.73 (6.70 to 8.84) | 0.816 (0.785 to 0.845) | 22.5 | 0.173 (0.135 to 0.218) | 0.38 |
| Historical | test | 430 | 6.94 (5.56 to 8.42) | 11.33 (8.80 to 13.99) | 0.844 (0.782 to 0.902) | 22.4 | 0.304 (0.220 to 0.391) | 0.59 |

At T-90 the bar to beat for the point forecast is historical, the lowest validation MAE, which scores 6.94 (5.56 to 8.42) minutes on test. For the delay bands it is timetable, the lowest validation Brier score. A model has to beat these on the test block with an interval that does not overlap theirs to count as an improvement.

## T-60 minutes

| Baseline | Block | Rows | MAE | RMSE | 80% coverage | Mean width | Brier | Log loss |
|---|---|---|---|---|---|---|---|---|
| Timetable | validation | 977 | 5.66 (4.78 to 6.73) | 9.10 (7.69 to 10.60) | n/a | n/a | 0.170 (0.116 to 0.235) | 2.93 |
| Timetable | test | 430 | 8.74 (7.38 to 10.21) | 13.34 (10.86 to 15.98) | n/a | n/a | 0.353 (0.240 to 0.455) | 6.10 |
| Persistence | validation | 977 | 4.41 (3.76 to 5.16) | 7.48 (6.32 to 8.55) | 0.892 (0.867 to 0.916) | 18.0 | 0.166 (0.111 to 0.230) | 2.86 |
| Persistence | test | 430 | 6.56 (5.24 to 7.88) | 10.30 (7.78 to 12.28) | 0.795 (0.719 to 0.861) | 18.0 | 0.344 (0.229 to 0.437) | 5.94 |
| Historical | validation | 977 | 5.18 (4.72 to 5.69) | 7.73 (6.70 to 8.84) | 0.816 (0.785 to 0.845) | 22.5 | 0.173 (0.135 to 0.218) | 0.38 |
| Historical | test | 430 | 6.94 (5.56 to 8.42) | 11.33 (8.80 to 13.99) | 0.844 (0.782 to 0.902) | 22.4 | 0.304 (0.220 to 0.391) | 0.59 |

At T-60 the bar to beat for the point forecast is persistence, the lowest validation MAE, which scores 6.56 (5.24 to 7.88) minutes on test. For the delay bands it is persistence, the lowest validation Brier score. A model has to beat these on the test block with an interval that does not overlap theirs to count as an improvement.

## T-30 minutes

| Baseline | Block | Rows | MAE | RMSE | 80% coverage | Mean width | Brier | Log loss |
|---|---|---|---|---|---|---|---|---|
| Timetable | validation | 977 | 5.66 (4.78 to 6.73) | 9.10 (7.69 to 10.60) | n/a | n/a | 0.170 (0.116 to 0.235) | 2.93 |
| Timetable | test | 430 | 8.74 (7.38 to 10.21) | 13.34 (10.86 to 15.98) | n/a | n/a | 0.353 (0.240 to 0.455) | 6.10 |
| Persistence | validation | 977 | 3.21 (2.69 to 3.75) | 6.02 (4.84 to 7.06) | 0.846 (0.814 to 0.874) | 12.0 | 0.127 (0.082 to 0.178) | 2.19 |
| Persistence | test | 430 | 4.71 (3.56 to 5.87) | 8.36 (5.66 to 10.38) | 0.791 (0.741 to 0.826) | 12.0 | 0.302 (0.205 to 0.389) | 5.22 |
| Historical | validation | 977 | 5.18 (4.72 to 5.69) | 7.73 (6.70 to 8.84) | 0.816 (0.785 to 0.845) | 22.5 | 0.173 (0.135 to 0.218) | 0.38 |
| Historical | test | 430 | 6.94 (5.56 to 8.42) | 11.33 (8.80 to 13.99) | 0.844 (0.782 to 0.902) | 22.4 | 0.304 (0.220 to 0.391) | 0.59 |

At T-30 the bar to beat for the point forecast is persistence, the lowest validation MAE, which scores 4.71 (3.56 to 5.87) minutes on test. For the delay bands it is persistence, the lowest validation Brier score. A model has to beat these on the test block with an interval that does not overlap theirs to count as an improvement.

## T-15 minutes

| Baseline | Block | Rows | MAE | RMSE | 80% coverage | Mean width | Brier | Log loss |
|---|---|---|---|---|---|---|---|---|
| Timetable | validation | 977 | 5.66 (4.78 to 6.73) | 9.10 (7.69 to 10.60) | n/a | n/a | 0.170 (0.116 to 0.235) | 2.93 |
| Timetable | test | 430 | 8.74 (7.38 to 10.21) | 13.34 (10.86 to 15.98) | n/a | n/a | 0.353 (0.240 to 0.455) | 6.10 |
| Persistence | validation | 977 | 3.01 (2.55 to 3.50) | 5.71 (4.45 to 6.76) | 0.897 (0.872 to 0.919) | 12.0 | 0.098 (0.062 to 0.140) | 1.70 |
| Persistence | test | 430 | 4.13 (3.12 to 5.19) | 7.71 (5.03 to 9.81) | 0.844 (0.815 to 0.874) | 12.0 | 0.265 (0.173 to 0.346) | 4.58 |
| Historical | validation | 977 | 5.18 (4.72 to 5.69) | 7.73 (6.70 to 8.84) | 0.816 (0.785 to 0.845) | 22.5 | 0.173 (0.135 to 0.218) | 0.38 |
| Historical | test | 430 | 6.94 (5.56 to 8.42) | 11.33 (8.80 to 13.99) | 0.844 (0.782 to 0.902) | 22.4 | 0.304 (0.220 to 0.391) | 0.59 |

At T-15 the bar to beat for the point forecast is persistence, the lowest validation MAE, which scores 4.13 (3.12 to 5.19) minutes on test. For the delay bands it is persistence, the lowest validation Brier score. A model has to beat these on the test block with an interval that does not overlap theirs to count as an improvement.

## T-5 minutes

| Baseline | Block | Rows | MAE | RMSE | 80% coverage | Mean width | Brier | Log loss |
|---|---|---|---|---|---|---|---|---|
| Timetable | validation | 977 | 5.66 (4.78 to 6.73) | 9.10 (7.69 to 10.60) | n/a | n/a | 0.170 (0.116 to 0.235) | 2.93 |
| Timetable | test | 430 | 8.74 (7.38 to 10.21) | 13.34 (10.86 to 15.98) | n/a | n/a | 0.353 (0.240 to 0.455) | 6.10 |
| Persistence | validation | 977 | 2.77 (2.41 to 3.16) | 5.13 (3.99 to 6.17) | 0.915 (0.898 to 0.931) | 12.0 | 0.100 (0.063 to 0.142) | 1.73 |
| Persistence | test | 430 | 3.78 (2.81 to 4.67) | 7.35 (4.76 to 9.32) | 0.874 (0.851 to 0.904) | 12.0 | 0.237 (0.142 to 0.310) | 4.10 |
| Historical | validation | 977 | 5.18 (4.72 to 5.69) | 7.73 (6.70 to 8.84) | 0.816 (0.785 to 0.845) | 22.5 | 0.173 (0.135 to 0.218) | 0.38 |
| Historical | test | 430 | 6.94 (5.56 to 8.42) | 11.33 (8.80 to 13.99) | 0.844 (0.782 to 0.902) | 22.4 | 0.304 (0.220 to 0.391) | 0.59 |

At T-5 the bar to beat for the point forecast is persistence, the lowest validation MAE, which scores 3.78 (2.81 to 4.67) minutes on test. For the delay bands it is persistence, the lowest validation Brier score. A model has to beat these on the test block with an interval that does not overlap theirs to count as an improvement.
