# Live evaluation

The champion model, MLflow run 4c88d55bbbed49018e3ce05e8199ce5b (registry version 2), was scored on service dates 2026-10-16 to 2026-10-22 of captured Darwin data, routes bhm-eus and ast-bhm. The model stayed frozen: it was trained and calibrated on dates up to 2026-10-15, and no model, recalibration or threshold was changed for this evaluation. The alert threshold is the frozen tau of 0.22. The model files and the registry alias were unchanged before and after the run (SHA-256 of every file in the model folder).

## Method

The API did not log predictions during the week, so every prediction was rebuilt after the fact as a point-in-time replay. For a service and a prediction time T, the live state was rebuilt from the Darwin messages whose message timestamp is at or before T, and no others: the events, the schedule versions (so a cancellation that arrived after T is not known at T), and the age of the newest message. That is stricter than the HSP rule used in training, which keeps events whose actual time is at or before T. The stop table, the features and the prediction come from the code the API runs: `LiveStore.stop_table`, `FeatureService.batch` (the training `build_snapshots`), `RiskService._predict` and, for connections, `RiskService.connection`. Features were built at 120, 60 and 15 minutes before the scheduled arrival. The 28 day history features come from the HSP history up to 2026-10-15, as they do in the API, so for later dates the history window holds fewer days (mean history rows per day are in the JSON). Actual arrival times come from Darwin. Message timestamps are Darwin message times, not the time the capture received the message.
As a check on the replay, 60 replayed predictions were also run through `RiskService.risk`, the code behind POST /risk: 60 of 60 matched on the median delay to 0.1 minute and 60 of 60 on P(delay of at least 15) to four places.

## Capture gaps

A gap is any stretch of message time longer than 5 minutes with no Darwin message. Over the window there were 78 gaps totalling 88.1 hours, the longest 382 minutes, found from 6,543,751 distinct message times. 74 of the gaps are 20 minutes or longer. Several hourly raw files are missing altogether. Hours missing per local day:

| Service date | Hours missing, whole day | Hours missing, 05:00 to 24:00 |
|---|---|---|
| 2026-10-16 | 3.2 | 3.2 |
| 2026-10-17 | 11.4 | 11.4 |
| 2026-10-18 | 12.7 | 10.2 |
| 2026-10-19 | 19.0 | 15.2 |
| 2026-10-20 | 15.0 | 11.4 |
| 2026-10-21 | 14.5 | 12.6 |
| 2026-10-22 | 12.3 | 9.9 |
| Total | 88.1 | 74.0 |

## Services kept and excluded

A service is excluded at a horizon when the window from T to its arrival overlaps a gap. The window ends at the scheduled arrival, or at the message that carried the actual arrival when that is later. The counts are per (service, route, horizon), and a service that calls at both AST, BHM and EUS counts once under each route, as in training. After the gap rule, a service is also left out when it was not yet known at T, was already cancelled or already arrived at T, or has no actual arrival (cancelled later, or the arrival was never captured).

| Route | Horizon | Services | Excluded for gap | Kept | Unknown at T | Cancelled at T | Arrived by T | No actual arrival | Scored |
|---|---|---|---|---|---|---|---|---|---|
| bhm-eus | T-120 | 363 | 301 | 62 | 0 | 3 | 0 | 3 | 48 |
| bhm-eus | T-60 | 363 | 285 | 78 | 0 | 3 | 0 | 5 | 61 |
| bhm-eus | T-15 | 363 | 245 | 118 | 0 | 6 | 1 | 15 | 96 |
| ast-bhm | T-120 | 561 | 464 | 97 | 12 | 3 | 0 | 11 | 71 |
| ast-bhm | T-60 | 561 | 442 | 119 | 16 | 7 | 0 | 10 | 82 |
| ast-bhm | T-15 | 561 | 363 | 198 | 0 | 13 | 0 | 12 | 138 |

Services scored by service date (kept and with an actual arrival):

| Service date | bhm-eus T-120 | bhm-eus T-60 | bhm-eus T-15 | ast-bhm T-120 | ast-bhm T-60 | ast-bhm T-15 |
|---|---|---|---|---|---|---|
| 2026-10-16 | 41 | 46 | 48 | 53 | 59 | 64 |
| 2026-10-17 | 5 | 5 | 7 | 16 | 18 | 22 |
| 2026-10-18 | 0 | 3 | 7 | 0 | 3 | 10 |
| 2026-10-19 | 0 | 0 | 1 | 0 | 0 | 2 |
| 2026-10-20 | 2 | 6 | 21 | 0 | 0 | 11 |
| 2026-10-21 | 0 | 0 | 4 | 0 | 0 | 11 |
| 2026-10-22 | 0 | 1 | 8 | 2 | 2 | 18 |

## Check of the actual times against HSP

HSP for 2026-10-16 was fetched for both routes into a separate data directory and compared with the Darwin actual arrival at the route destination, service by service on the same rid. Of 118 arrivals that both feeds recorded, 118 were identical to the minute (100.0%).

| Route | HSP services | Also in capture | Both have actual | Identical to the minute | Within 60 s | Actual only in HSP | Actual only in Darwin |
|---|---|---|---|---|---|---|---|
| bhm-eus | 70 | 69 | 57 | 100.0% | 100.0% | 12 | 0 |
| ast-bhm | 75 | 75 | 61 | 100.0% | 100.0% | 14 | 0 |

Across every stop of the services in both feeds, 3315 actual times were compared: 100.0% identical and 100.0% within 60 seconds. 733 had an actual only in HSP and 2 only in Darwin.

## Arrival ranges and probabilities by route and horizon

MAE is in minutes against the actual arrival delay. The validation bar is the baseline with the lowest MAE on the validation block of the champion run, the same bar used in reports/model_results.md. Paired differences are the model's MAE minus the baseline's, so a negative number favours the model. Intervals are 95% bootstrap intervals over service dates; with fewer than five dates no interval is given, because resampling so few dates says nothing. The window holds at most seven service dates, and on the days with the most captured time one date can hold most of the rows, so every interval is wide.

| Route | Horizon | Rows | Dates | Model MAE | Validation bar | Bar MAE | Model minus bar | Against the best baseline on these rows |
|---|---|---|---|---|---|---|---|---|
| bhm-eus | T-120 | 48 | 3 | 6.18 | historical | 7.16 | -0.98 | persistence: -0.20 |
| bhm-eus | T-60 | 61 | 5 | 5.38 (4.14 to 6.46) | persistence | 5.72 (4.13 to 7.59) | -0.34 (-1.45 to 0.55) | persistence: -0.34 (-1.45 to 0.55) |
| bhm-eus | T-15 | 96 | 7 | 3.91 (2.77 to 5.49) | persistence | 3.99 (2.27 to 5.45) | -0.08 (-0.51 to 1.46) | persistence: -0.08 (-0.51 to 1.46) |
| ast-bhm | T-120 | 71 | 3 | 3.06 | historical | 2.79 | 0.27 | historical: 0.27 |
| ast-bhm | T-60 | 82 | 4 | 2.71 | historical | 2.65 | 0.06 | historical: 0.06 |
| ast-bhm | T-15 | 138 | 7 | 2.34 (1.67 to 2.68) | persistence | 2.80 (1.81 to 3.29) | -0.47 (-0.64 to -0.08) | historical: -0.24 (-0.53 to -0.09) |

Nominal 80% interval, coverage and mean width in minutes:

| Route | Horizon | Model coverage | Model width | Persistence coverage | Historical coverage |
|---|---|---|---|---|---|
| bhm-eus | T-120 | 0.771 | 18.5 | 0.750 | 0.750 |
| bhm-eus | T-60 | 0.787 (0.583 to 0.929) | 18.8 (14.4 to 26.0) | 0.639 (0.533 to 0.818) | 0.787 (0.583 to 0.929) |
| bhm-eus | T-15 | 0.823 (0.747 to 0.900) | 17.6 (14.2 to 26.2) | 0.771 (0.683 to 0.844) | 0.854 (0.784 to 0.958) |
| ast-bhm | T-120 | 0.746 | 6.9 | 0.986 | 0.803 |
| ast-bhm | T-60 | 0.829 | 7.7 | 0.927 | 0.805 |
| ast-bhm | T-15 | 0.775 (0.723 to 0.850) | 6.8 (5.3 to 7.5) | 0.913 (0.895 to 0.965) | 0.848 (0.800 to 0.942) |

P(delay of at least 15 minutes), model against the historical baseline:

| Route | Horizon | Delays of 15 or more | Model Brier | Historical Brier | Model ECE | Historical ECE | ECE bins |
|---|---|---|---|---|---|---|---|
| bhm-eus | T-120 | 9 of 48 | 0.1013 | 0.1325 | 0.0842 | 0.1404 | 4 |
| bhm-eus | T-60 | 11 of 61 | 0.0813 (0.0121 to 0.0991) | 0.1341 (0.0232 to 0.1514) | 0.0567 (0.0325 to 0.1752) | 0.1460 (0.1273 to 0.2195) | 6 |
| bhm-eus | T-15 | 15 of 96 | 0.0940 (0.0348 to 0.1680) | 0.1262 (0.0493 to 0.1471) | 0.0945 (0.0449 to 0.2271) | 0.1606 (0.1131 to 0.1954) | 9 |
| ast-bhm | T-120 | 1 of 71 | 0.0140 | 0.0141 | 0.0199 | 0.0209 | 7 |
| ast-bhm | T-60 | 1 of 82 | 0.0122 | 0.0122 | 0.0167 | 0.0203 | 8 |
| ast-bhm | T-15 | 1 of 138 | 0.0071 (0.0000 to 0.0274) | 0.0073 (0.0002 to 0.0279) | 0.0065 (0.0015 to 0.0273) | 0.0169 (0.0108 to 0.0359) | 10 |

ECE uses equal-count bins, ten where there are enough rows and fewer otherwise. With so few delays of 15 or more, ECE and Brier for this event rest on a small number of events.

Coverage of the model's 80% interval by observed lateness at T:

| Route | Horizon | not_started | 0-4 | 5-14 | 15+ |
|---|---|---|---|---|---|
| bhm-eus | T-120 | 0.75 (n=20, too few for an interval) | 0.79 (n=24, too few for an interval) | 0.50 (n=2, too few for an interval) | 1.00 (n=2, too few for an interval) |
| bhm-eus | T-60 | 1.00 (n=1, too few for an interval) | 0.80 (0.55 to 0.91) (n=41) | 0.71 (n=17, too few for an interval) | 1.00 (n=2, too few for an interval) |
| bhm-eus | T-15 | 1.00 (n=4, too few for an interval) | 0.78 (0.68 to 0.86) (n=67) | 0.89 (n=18, too few for an interval) | 1.00 (n=7, too few for an interval) |
| ast-bhm | T-120 | 0.75 (n=71) | n/a (n=0) | n/a (n=0) | n/a (n=0) |
| ast-bhm | T-60 | 0.82 (n=80) | 1.00 (n=2, too few for an interval) | n/a (n=0) | n/a (n=0) |
| ast-bhm | T-15 | 0.75 (n=4, too few for an interval) | 0.77 (0.70 to 0.87) (n=128) | 1.00 (n=6, too few for an interval) | n/a (n=0) |

## Day by day MAE

Model MAE with the validation bar's MAE in brackets, and the number of services.

| Service date | bhm-eus T-120 | bhm-eus T-60 | bhm-eus T-15 | ast-bhm T-120 | ast-bhm T-60 | ast-bhm T-15 |
|---|---|---|---|---|---|---|
| 2026-10-16 | 6.57 (7.51), n=41 | 5.32 (5.65), n=46 | 3.13 (3.72), n=48 | 3.44 (3.15), n=53 | 3.02 (2.97), n=59 | 2.78 (3.47), n=64 |
| 2026-10-17 | 5.09 (6.10), n=5 | 4.76 (3.20), n=5 | 3.48 (3.29), n=7 | 1.59 (1.50), n=16 | 1.65 (1.67), n=18 | 1.55 (1.86), n=22 |
| 2026-10-18 | none | 3.40 (4.67), n=3 | 2.04 (1.71), n=7 | none | 1.60 (1.67), n=3 | 2.71 (2.60), n=10 |
| 2026-10-19 | none | none | 5.50 (2.00), n=1 | none | none | 0.78 (1.50), n=2 |
| 2026-10-20 | 0.85 (2.50), n=2 | 7.44 (8.50), n=6 | 6.10 (6.62), n=21 | none | none | 1.40 (1.18), n=11 |
| 2026-10-21 | none | none | 1.94 (2.62), n=4 | none | none | 1.92 (2.18), n=11 |
| 2026-10-22 | none | 4.80 (8.00), n=1 | 5.66 (2.25), n=8 | 4.76 (3.50), n=2 | 4.69 (3.50), n=2 | 2.54 (3.22), n=18 |

## Connections: AST to BHM inbound, BHM to EUS outbound

A pair is one AST to BHM service with one BHM to EUS service scheduled 10 to 30 minutes after it is due at Birmingham, the pair definition of the historical backtest, so no pair is infeasible on paper. The minimum connection time is assumed, not published. There were 533 such pairs in the window, 190 with a known outcome (both trains ran and have an actual time), and a miss rate of 12.1% among them. Each forecast is the API's `RiskService.connection` answer with the state known k minutes before the outbound leaves: the recalibration setting, the outbound delay history, the tail model and the alert flag are those the API uses. The baseline is the historical miss frequency by scheduled gap bucket, fitted on 18270 pairs from 159 training days only.

Why pairs were refused or not answered at T-15: 47 x no planned outbound departs at that time; 9 x the HH:MM is cancelled; 8 x the inbound service is cancelled (reason N); 1 x service does not call at a configured route. Of the 65 such pairs, 34 had a gap in the 60 minutes before T.

Why pairs were refused or not answered at T-60: 14 x service does not call at a configured route; 4 x the HH:MM is cancelled; 3 x the inbound service is cancelled (reason N). Of the 37 such pairs, 11 had a gap in the 60 minutes before T.

| Forecast | Feasible pairs | Excluded for gap | Answered | Refused or other | Answered with known outcome | Of those, inbound not yet arrived |
|---|---|---|---|---|---|---|
| T-15 | 533 | 341 | 127 | 65 | 112 | 52 |
| T-60 | 533 | 412 | 84 | 37 | 71 | 71 |

| Forecast | Pairs | Realised miss rate | Mean predicted | Model Brier | Baseline Brier | Model ECE | Baseline ECE |
|---|---|---|---|---|---|---|---|
| T-60 before outbound | 71 | 14.1% | 10.1% | 0.1106 | 0.1115 | 0.0607 | 0.0778 |
| T-15 before outbound (inbound not yet arrived) | 52 | 30.8% | 24.4% | 0.2228 (0.0609 to 0.2866) | 0.2340 (0.1358 to 0.3193) | 0.1737 (0.1440 to 0.2894) | 0.2071 (0.1072 to 0.3292) |
| T-15 before outbound (all pairs) | 112 | 14.3% | 11.3% | 0.1034 (0.0156 to 0.1533) | 0.1090 (0.0257 to 0.1634) | 0.0868 (0.0397 to 0.1475) | 0.1067 (0.0290 to 0.1770) |

The all-pairs T-15 row includes pairs whose inbound had already arrived, for which the delay is known, so it overstates what the model adds; the not-yet-arrived row is the fair test.

Alert at the frozen tau of 0.22 (false alert cost 1, missed alert cost 5, both assumed):

| Pairs | Count | Misses | Alerts | True alerts | Precision | Recall |
|---|---|---|---|---|---|---|
| T-60 before outbound | 71 | 10 | 14 | 5 | 0.357 | 0.500 |
| T-15 before outbound (inbound not yet arrived) | 52 | 16 | 23 | 8 | 0.348 | 0.500 |
| T-15 before outbound (all pairs) | 112 | 16 | 23 | 8 | 0.348 | 0.500 |
| both forecast times pooled (all pairs) | 183 | 26 | 37 | 13 | 0.351 | 0.500 |

## Live against the test block

Each live number next to the test-block number of the same champion run, from reports/model_results.md and the run's results.json. 'Similar' means the 95% intervals overlap, which with seven or fewer dates in each block says only that the data cannot separate them. 'Better' and 'worse' mean the intervals do not overlap. For coverage the labels are about distance from the nominal 80%.

| Route | Horizon | Metric | Live | Test | Live against test |
|---|---|---|---|---|---|
| bhm-eus | T-120 | MAE, minutes | 6.176 | 6.271 (4.940 to 7.513) | not testable (fewer than 5 dates), live lower |
| bhm-eus | T-120 | MAE minus validation bar | -0.981 | -0.671 (-1.689 to 0.098) | not testable (fewer than 5 dates), live lower |
| bhm-eus | T-120 | 80% coverage | 0.771 | 0.814 (0.744 to 0.883) | not testable (fewer than 5 dates), live lower |
| bhm-eus | T-120 | 80% interval width | 18.492 | 18.545 (17.584 to 19.788) | not testable (fewer than 5 dates), live lower |
| bhm-eus | T-120 | P(15+) Brier | 0.101 | 0.141 (0.092 to 0.186) | not testable (fewer than 5 dates), live lower |
| bhm-eus | T-120 | P(15+) ECE | 0.084 | 0.083 (0.049 to 0.140) | not testable (fewer than 5 dates), live higher |
| bhm-eus | T-120 | P(15+) Brier, historical baseline | 0.133 | 0.146 (0.108 to 0.186) | not testable (fewer than 5 dates), live lower |
| bhm-eus | T-60 | MAE, minutes | 5.382 (4.140 to 6.459) | 5.437 (4.438 to 6.579) | similar |
| bhm-eus | T-60 | MAE minus validation bar | -0.340 (-1.454 to 0.553) | -1.125 (-1.597 to -0.430) | similar |
| bhm-eus | T-60 | 80% coverage | 0.787 (0.583 to 0.929) | 0.812 (0.749 to 0.881) | similar |
| bhm-eus | T-60 | 80% interval width | 18.796 (14.360 to 25.965) | 16.481 (15.518 to 17.823) | similar |
| bhm-eus | T-60 | P(15+) Brier | 0.081 (0.012 to 0.099) | 0.132 (0.091 to 0.172) | similar |
| bhm-eus | T-60 | P(15+) ECE | 0.057 (0.033 to 0.175) | 0.077 (0.055 to 0.130) | similar |
| bhm-eus | T-60 | P(15+) Brier, historical baseline | 0.134 (0.023 to 0.151) | 0.146 (0.108 to 0.186) | similar |
| bhm-eus | T-15 | MAE, minutes | 3.912 (2.775 to 5.493) | 3.540 (2.831 to 4.459) | similar |
| bhm-eus | T-15 | MAE minus validation bar | -0.078 (-0.515 to 1.458) | -0.588 (-0.943 to -0.207) | similar |
| bhm-eus | T-15 | 80% coverage | 0.823 (0.747 to 0.900) | 0.840 (0.817 to 0.864) | similar |
| bhm-eus | T-15 | 80% interval width | 17.647 (14.151 to 26.190) | 11.654 (10.814 to 12.328) | worse |
| bhm-eus | T-15 | P(15+) Brier | 0.094 (0.035 to 0.168) | 0.090 (0.061 to 0.122) | similar |
| bhm-eus | T-15 | P(15+) ECE | 0.094 (0.045 to 0.227) | 0.049 (0.040 to 0.094) | similar |
| bhm-eus | T-15 | P(15+) Brier, historical baseline | 0.126 (0.049 to 0.147) | 0.146 (0.108 to 0.186) | similar |
| ast-bhm | T-120 | MAE, minutes | 3.057 | 2.298 (1.912 to 2.684) | not testable (fewer than 5 dates), live higher |
| ast-bhm | T-120 | MAE minus validation bar | 0.269 | -0.008 (-0.083 to 0.073) | not testable (fewer than 5 dates), live higher |
| ast-bhm | T-120 | 80% coverage | 0.746 | 0.830 (0.789 to 0.866) | not testable (fewer than 5 dates), live lower |
| ast-bhm | T-120 | 80% interval width | 6.867 | 7.084 (6.943 to 7.203) | not testable (fewer than 5 dates), live lower |
| ast-bhm | T-120 | P(15+) Brier | 0.014 | 0.003 (0.000 to 0.010) | not testable (fewer than 5 dates), live higher |
| ast-bhm | T-120 | P(15+) ECE | 0.020 | 0.009 (0.005 to 0.014) | not testable (fewer than 5 dates), live higher |
| ast-bhm | T-120 | P(15+) Brier, historical baseline | 0.014 | 0.003 (0.000 to 0.010) | not testable (fewer than 5 dates), live higher |
| ast-bhm | T-60 | MAE, minutes | 2.706 | 2.301 (1.963 to 2.623) | not testable (fewer than 5 dates), live higher |
| ast-bhm | T-60 | MAE minus validation bar | 0.059 | -0.005 (-0.074 to 0.033) | not testable (fewer than 5 dates), live higher |
| ast-bhm | T-60 | 80% coverage | 0.829 | 0.788 (0.754 to 0.820) | not testable (fewer than 5 dates), live higher |
| ast-bhm | T-60 | 80% interval width | 7.717 | 7.402 (6.918 to 7.928) | not testable (fewer than 5 dates), live higher |
| ast-bhm | T-60 | P(15+) Brier | 0.012 | 0.003 (0.000 to 0.010) | not testable (fewer than 5 dates), live higher |
| ast-bhm | T-60 | P(15+) ECE | 0.017 | 0.007 (0.004 to 0.014) | not testable (fewer than 5 dates), live higher |
| ast-bhm | T-60 | P(15+) Brier, historical baseline | 0.012 | 0.003 (0.000 to 0.010) | not testable (fewer than 5 dates), live higher |
| ast-bhm | T-15 | MAE, minutes | 2.337 (1.668 to 2.677) | 1.933 (1.653 to 2.224) | similar |
| ast-bhm | T-15 | MAE minus validation bar | -0.467 (-0.636 to -0.075) | -0.255 (-0.369 to -0.154) | similar |
| ast-bhm | T-15 | 80% coverage | 0.775 (0.723 to 0.850) | 0.805 (0.761 to 0.841) | similar |
| ast-bhm | T-15 | 80% interval width | 6.761 (5.301 to 7.492) | 6.198 (5.745 to 6.645) | similar |
| ast-bhm | T-15 | P(15+) Brier | 0.007 (0.000 to 0.027) | 0.003 (0.000 to 0.010) | similar |
| ast-bhm | T-15 | P(15+) ECE | 0.007 (0.001 to 0.027) | 0.003 (0.001 to 0.010) | similar |
| ast-bhm | T-15 | P(15+) Brier, historical baseline | 0.007 (0.000 to 0.028) | 0.003 (0.000 to 0.010) | similar |

| Connection forecast | Metric | Live | Test | Live against test |
|---|---|---|---|---|
| T-60 before outbound | P(miss) Brier | 0.1106 | 0.0525 (0.0395 to 0.0687) | not testable (fewer than 5 dates), live higher |
| T-60 before outbound | P(miss) ECE | 0.0607 | 0.0235 (0.0152 to 0.0448) | not testable (fewer than 5 dates), live higher |
| T-60 before outbound | P(miss) Brier, gap-bucket baseline | 0.1115 | 0.0528 (0.0382 to 0.0706) | not testable (fewer than 5 dates), live higher |
| T-15 before outbound (inbound not yet arrived) | P(miss) Brier | 0.2228 (0.0609 to 0.2866) | 0.1012 (0.0734 to 0.1422) | similar |
| T-15 before outbound (inbound not yet arrived) | P(miss) ECE | 0.1737 (0.1440 to 0.2894) | 0.0813 (0.0688 to 0.1326) | worse |
| T-15 before outbound (inbound not yet arrived) | P(miss) Brier, gap-bucket baseline | 0.2340 (0.1358 to 0.3193) | 0.1304 (0.0956 to 0.1819) | similar |
| T-15 before outbound (all pairs) | P(miss) Brier | 0.1034 (0.0156 to 0.1533) | 0.0406 (0.0287 to 0.0566) | similar |
| T-15 before outbound (all pairs) | P(miss) ECE | 0.0868 (0.0397 to 0.1475) | 0.0285 (0.0156 to 0.0466) | similar |
| T-15 before outbound (all pairs) | P(miss) Brier, gap-bucket baseline | 0.1090 (0.0257 to 0.1634) | 0.0528 (0.0382 to 0.0706) | similar |

Verdicts over all comparisons above: 25 similar, 16 not testable (fewer than 5 dates), live higher, 8 not testable (fewer than 5 dates), live lower, 2 worse.

## Sensitivity: no gap in the hour before T

The gap rule above looks only at the window from T to arrival. A gap that ended shortly before T can still have hidden the messages about the stops a train had already passed, so a train that was running can look not started in the replayed state. The share of predictions with no last stop known is shown for the main set, for this stricter set and for the test block. The stricter set keeps only predictions with no gap in the 60 minutes before T. Both sets are reported; the stricter one is a check on the main numbers.

| Route | Horizon | Rows kept of main | Dates | Not started: main / stricter / test | Model MAE | Test MAE | MAE, live against test | Model minus bar | Coverage | P(15+) Brier |
|---|---|---|---|---|---|---|---|---|---|---|
| bhm-eus | T-120 | 43 of 48 | 3 | 42% / 42% / 35% | 5.77 | 6.27 (4.94 to 7.51) | not testable (fewer than 5 dates), live lower | -1.17 | 0.767 | 0.0833 |
| bhm-eus | T-60 | 50 of 61 | 3 | 2% / 2% / 0% | 4.98 | 5.44 (4.44 to 6.58) | not testable (fewer than 5 dates), live lower | 0.30 | 0.760 | 0.0646 |
| bhm-eus | T-15 | 66 of 96 | 5 | 4% / 0% / 0% | 3.01 (2.62 to 3.95) | 3.54 (2.83 to 4.46) | similar | -0.28 (-0.64 to -0.02) | 0.833 (0.571 to 0.900) | 0.0437 (0.0001 to 0.0591) |
| ast-bhm | T-120 | 63 of 71 | 2 | 100% / 100% / 100% | 2.85 | 2.30 (1.91 to 2.68) | not testable (fewer than 5 dates), live higher | 0.18 | 0.762 | 0.0158 |
| ast-bhm | T-60 | 71 of 82 | 3 | 98% / 97% / 99% | 2.84 | 2.30 (1.96 to 2.62) | not testable (fewer than 5 dates), live higher | 0.05 | 0.817 | 0.0141 |
| ast-bhm | T-15 | 84 of 138 | 5 | 3% / 4% / 1% | 2.55 (1.50 to 2.96) | 1.93 (1.65 to 2.22) | similar | -0.58 (-0.71 to -0.04) | 0.762 (0.632 to 0.875) | 0.0117 (0.0000 to 0.0505) |

The same check for the connection forecasts, keeping pairs with no gap in the hour before T:

| Forecast | Pairs | Realised miss rate | Mean predicted | Model Brier | Baseline Brier | Model ECE | Baseline ECE |
|---|---|---|---|---|---|---|---|
| T-60 before outbound | 59 | 15.3% | 9.6% | 0.1169 | 0.1216 | 0.0617 | 0.0859 |
| T-15 before outbound (inbound not yet arrived) | 35 | 28.6% | 22.7% | 0.2293 | 0.2283 | 0.1652 | 0.2043 |
| T-15 before outbound (all pairs) | 77 | 13.0% | 10.3% | 0.1042 (0.0198 to 0.1379) | 0.1042 (0.0068 to 0.1528) | 0.0764 (0.0228 to 0.1596) | 0.0845 (0.0324 to 0.1677) |

## Limits of this evaluation

Scored predictions are concentrated on the dates with the least capture loss. In the six route and horizon cells the busiest single date holds between 46% and 85% of the scored rows.

The capture lost 88.1 hours of message time in the window, so the services kept are those that happened to fall in captured stretches, mostly on the days with the fewest gaps. The window is at most seven service dates, in autumn, so there is no winter or storm season, and bootstrap intervals over so few dates are wide. The history features come from HSP up to 2026-10-15 and do not include the Darwin days of the window, as in the API. The connection pairs use the assumed 10 minute connection time. The replayed state uses Darwin message times and cannot say how late the live service would have received a message. Nothing here explains why live differs from test where it does.
