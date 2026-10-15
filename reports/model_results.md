# Model results

MLflow run f88ed558467c4ef8aed34a29926dbe45. The model was trained on 61 complete days (2026-07-18 to 2026-09-28), tuned and calibrated on validation (2026-09-30 to 2026-10-06), and scored once on test (2026-10-08 to 2026-10-14). Brackets are 95% bootstrap intervals over service dates. Intervals are nominal 80% intervals. Delay is minutes late at the route destination, and cancelled services are not scored.

The interval comes from LightGBM quantile models at 0.1, 0.5 and 0.9 with the horizon as a feature, widened by a conformal offset computed separately for each horizon and each observed lateness group at T (not started, 0 to 4, 5 to 14, and 15 or more minutes late). The lateness group uses only what is known at T. The target per horizon was chosen by validation pinball loss between the direct delay and the delay minus the lateness at T: T-120: residual, T-60: residual, T-15: direct. Band probabilities use temperature calibration. The delay bands are provisional.

## T-120 minutes, test block

| Method | MAE | 80% coverage | Mean width | Brier | Log loss |
|---|---|---|---|---|---|
| Timetable | 8.88 (7.59 to 10.14) | n/a | n/a | 0.366 (0.258 to 0.460) | 6.32 |
| Persistence | 7.61 (6.09 to 8.90) | 0.801 (0.741 to 0.861) | 19.0 | 0.380 (0.262 to 0.469) | 6.56 |
| Historical | 7.14 (5.90 to 8.32) | 0.840 (0.792 to 0.895) | 20.3 | 0.318 (0.236 to 0.389) | 0.62 |
| Model, global CQR | 6.37 (5.30 to 7.39) | 0.780 (0.721 to 0.851) | 18.5 | 0.305 (0.209 to 0.383) | 0.60 |
| Model, Mondrian CQR | 6.37 (5.30 to 7.39) | 0.801 (0.744 to 0.870) | 19.3 | 0.305 (0.209 to 0.383) | 0.60 |

Coverage by observed lateness at T, which is what the conformal guarantee covers:

| Method | not_started | 0-4 | 5-14 | 15+ |
|---|---|---|---|---|
| Timetable | n/a | n/a | n/a | n/a |
| Persistence | 0.84 (n=147, width 19.0) | 0.85 (n=213, width 19.0) | 0.65 (n=55, width 19.0) | 0.35 (n=17, width 19.0) |
| Historical | 0.83 (n=147, width 19.8) | 0.86 (n=213, width 20.4) | 0.89 (n=55, width 21.3) | 0.53 (n=17, width 21.6) |
| Model, global CQR | 0.81 (n=147, width 21.6) | 0.76 (n=213, width 14.5) | 0.82 (n=55, width 18.5) | 0.71 (n=17, width 42.3) |
| Model, Mondrian CQR | 0.81 (n=147, width 21.2) | 0.76 (n=213, width 14.7) | 0.91 (n=55, width 23.6) | 0.88 (n=17, width 47.4) |

Coverage by true delay band, which is what passengers feel:

| Method | 0-14 | 15-29 | 30-59 | 60-119 | 120+ |
|---|---|---|---|---|---|
| Timetable | n/a | n/a | n/a | n/a | n/a |
| Persistence | 0.87 (n=353, width 19.0) | 0.57 (n=60, width 19.0) | 0.22 (n=18, width 19.0) | 1.00 (n=1, width 19.0) | n/a |
| Historical | 0.95 (n=353, width 20.4) | 0.47 (n=60, width 20.4) | 0.00 (n=18, width 19.0) | 0.00 (n=1, width 24.0) | n/a |
| Model, global CQR | 0.87 (n=353, width 17.6) | 0.43 (n=60, width 19.3) | 0.28 (n=18, width 30.6) | 0.00 (n=1, width 61.8) | n/a |
| Model, Mondrian CQR | 0.88 (n=353, width 18.4) | 0.50 (n=60, width 20.2) | 0.33 (n=18, width 32.8) | 0.00 (n=1, width 66.9) | n/a |

## T-60 minutes, test block

| Method | MAE | 80% coverage | Mean width | Brier | Log loss |
|---|---|---|---|---|---|
| Timetable | 8.88 (7.59 to 10.14) | n/a | n/a | 0.366 (0.258 to 0.460) | 6.32 |
| Persistence | 6.73 (5.41 to 7.92) | 0.745 (0.676 to 0.802) | 16.0 | 0.361 (0.261 to 0.433) | 6.24 |
| Historical | 7.14 (5.90 to 8.32) | 0.840 (0.792 to 0.895) | 20.3 | 0.318 (0.236 to 0.389) | 0.62 |
| Model, global CQR | 5.65 (4.63 to 6.61) | 0.764 (0.714 to 0.823) | 15.1 | 0.280 (0.199 to 0.350) | 0.57 |
| Model, Mondrian CQR | 5.65 (4.63 to 6.61) | 0.766 (0.715 to 0.828) | 15.7 | 0.280 (0.199 to 0.350) | 0.57 |

Coverage by observed lateness at T, which is what the conformal guarantee covers:

| Method | not_started | 0-4 | 5-14 | 15+ |
|---|---|---|---|---|
| Timetable | n/a | n/a | n/a | n/a |
| Persistence | 0.50 (n=2, width 16.0) | 0.81 (n=302, width 16.0) | 0.66 (n=103, width 16.0) | 0.36 (n=25, width 16.0) |
| Historical | 0.50 (n=2, width 18.5) | 0.85 (n=302, width 20.1) | 0.88 (n=103, width 20.9) | 0.56 (n=25, width 20.6) |
| Model, global CQR | 0.00 (n=2, width 40.8) | 0.74 (n=302, width 13.6) | 0.86 (n=103, width 15.6) | 0.68 (n=25, width 29.2) |
| Model, Mondrian CQR | 0.00 (n=2, width 40.5) | 0.73 (n=302, width 13.3) | 0.91 (n=103, width 18.2) | 0.72 (n=25, width 31.9) |

Coverage by true delay band, which is what passengers feel:

| Method | 0-14 | 15-29 | 30-59 | 60-119 | 120+ |
|---|---|---|---|---|---|
| Timetable | n/a | n/a | n/a | n/a | n/a |
| Persistence | 0.82 (n=353, width 16.0) | 0.45 (n=60, width 16.0) | 0.22 (n=18, width 16.0) | 0.00 (n=1, width 16.0) | n/a |
| Historical | 0.95 (n=353, width 20.4) | 0.47 (n=60, width 20.4) | 0.00 (n=18, width 19.0) | 0.00 (n=1, width 24.0) | n/a |
| Model, global CQR | 0.85 (n=353, width 14.4) | 0.43 (n=60, width 15.0) | 0.28 (n=18, width 27.1) | 0.00 (n=1, width 57.1) | n/a |
| Model, Mondrian CQR | 0.85 (n=353, width 14.8) | 0.43 (n=60, width 16.0) | 0.33 (n=18, width 28.6) | 0.00 (n=1, width 59.7) | n/a |

## T-15 minutes, test block

| Method | MAE | 80% coverage | Mean width | Brier | Log loss |
|---|---|---|---|---|---|
| Timetable | 8.88 (7.59 to 10.14) | n/a | n/a | 0.366 (0.258 to 0.460) | 6.32 |
| Persistence | 4.25 (3.20 to 5.23) | 0.789 (0.768 to 0.816) | 11.0 | 0.278 (0.188 to 0.351) | 4.80 |
| Historical | 7.14 (5.90 to 8.32) | 0.840 (0.792 to 0.895) | 20.3 | 0.318 (0.236 to 0.389) | 0.62 |
| Model, global CQR | 4.06 (3.28 to 4.87) | 0.775 (0.731 to 0.831) | 12.1 | 0.213 (0.145 to 0.274) | 0.43 |
| Model, Mondrian CQR | 4.06 (3.28 to 4.87) | 0.766 (0.726 to 0.815) | 11.7 | 0.213 (0.145 to 0.274) | 0.43 |

Coverage by observed lateness at T, which is what the conformal guarantee covers:

| Method | not_started | 0-4 | 5-14 | 15+ |
|---|---|---|---|---|
| Timetable | n/a | n/a | n/a | n/a |
| Persistence | n/a | 0.78 (n=241, width 11.0) | 0.82 (n=151, width 11.0) | 0.70 (n=40, width 11.0) |
| Historical | n/a | 0.85 (n=241, width 19.7) | 0.94 (n=151, width 21.2) | 0.42 (n=40, width 20.8) |
| Model, global CQR | n/a | 0.72 (n=241, width 9.8) | 0.84 (n=151, width 11.7) | 0.85 (n=40, width 27.0) |
| Model, Mondrian CQR | n/a | 0.75 (n=241, width 10.1) | 0.82 (n=151, width 11.3) | 0.68 (n=40, width 22.7) |

Coverage by true delay band, which is what passengers feel:

| Method | 0-14 | 15-29 | 30-59 | 60-119 | 120+ |
|---|---|---|---|---|---|
| Timetable | n/a | n/a | n/a | n/a | n/a |
| Persistence | 0.85 (n=353, width 11.0) | 0.62 (n=60, width 11.0) | 0.28 (n=18, width 11.0) | 0.00 (n=1, width 11.0) | n/a |
| Historical | 0.95 (n=353, width 20.4) | 0.47 (n=60, width 20.4) | 0.00 (n=18, width 19.0) | 0.00 (n=1, width 24.0) | n/a |
| Model, global CQR | 0.82 (n=353, width 10.7) | 0.60 (n=60, width 13.4) | 0.56 (n=18, width 32.1) | 0.00 (n=1, width 54.1) | n/a |
| Model, Mondrian CQR | 0.82 (n=353, width 10.7) | 0.52 (n=60, width 12.0) | 0.56 (n=18, width 29.5) | 0.00 (n=1, width 49.7) | n/a |

Delay bands where the model interval covers under 70% of at least 30 test rows: T-120 band 15-29 at 0.50 (n=60); T-60 band 15-29 at 0.43 (n=60); T-15 band 15-29 at 0.52 (n=60).

## Limitations

These results cover one route, BHM to EUS. BHM to MAN and AST to EUS via BHM have not been ingested yet, so nothing here says the model transfers to them. The history runs from mid July to mid October, so it covers summer and the start of autumn but no winter, leaf-fall or storm season, and a seasonal shift would change both the delays and the calibration. The test block is seven days, so the bootstrap intervals are wide and the late bands hold few rows; a band with fewer than 30 rows is not flagged and should not be read as covered. The conformal guarantee holds on average within each observed lateness group if the days are exchangeable, which a time split only approximates, and it does not hold within the true delay bands. The delay bands are provisional until the per-operator compensation rules are checked.
