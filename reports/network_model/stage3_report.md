# Stage 3: route-agnostic model

Run 71d0bc8d7f8e4803963f9af3ecd8d1ed (training commit e639e68). Option: residual over shrunk persistence, one LightGBM for all routes, no route, station or operator identity. Test block 2026-10-10 to 2026-10-16, held-out routes only (9713 rows over six horizons), scored once. Intervals are 95% date-bootstrap CIs.

## Test 2: MAE, model minus baseline (paired, minutes)

| baseline | minutes before | n | model MAE | baseline MAE | difference [95% CI] |
|---|---|---|---|---|---|
| persistence | 15 | 1619 | 2.192 | 2.405 | -0.213 [-0.364, -0.114] |
| persistence | 60 | 1619 | 3.551 | 3.819 | -0.268 [-0.374, -0.169] |
| shrunk_persistence | 15 | 1619 | 2.192 | 2.405 | -0.213 [-0.364, -0.114] |
| shrunk_persistence | 60 | 1619 | 3.551 | 3.685 | -0.134 [-0.240, -0.045] |
| historical | 15 | 1619 | 2.192 | 4.035 | -1.843 [-2.308, -1.431] |
| historical | 60 | 1619 | 3.551 | 4.035 | -0.484 [-0.731, -0.255] |

## Tests 3 and 4 on held-out routes together

| method | minutes before | MAE | raw 80% coverage | Brier | ECE |
|---|---|---|---|---|---|
| model | 15 | 2.19 | 0.737 [0.714, 0.761] | 0.021 [0.015, 0.027] | 0.014 [0.011, 0.019] |
| model | 60 | 3.55 | 0.770 [0.749, 0.795] | 0.039 [0.030, 0.050] | 0.015 [0.011, 0.026] |
| persistence | 15 | 2.41 | 0.850 [0.828, 0.872] | 0.023 [0.016, 0.029] | 0.018 [0.011, 0.026] |
| persistence | 60 | 3.82 | 0.785 [0.769, 0.798] | 0.046 [0.033, 0.057] | 0.025 [0.018, 0.039] |
| shrunk_persistence | 15 | 2.41 | 0.850 [0.827, 0.873] | 0.023 [0.016, 0.029] | 0.018 [0.011, 0.025] |
| shrunk_persistence | 60 | 3.69 | 0.757 [0.736, 0.780] | 0.045 [0.032, 0.056] | 0.019 [0.013, 0.034] |
| historical | 15 | 4.03 | 0.858 [0.835, 0.881] | 0.051 [0.033, 0.069] | 0.017 [0.008, 0.038] |
| historical | 60 | 4.03 | 0.858 [0.835, 0.880] | 0.051 [0.034, 0.069] | 0.017 [0.007, 0.037] |

Paired Brier difference, model minus baseline, P(delay >= 15):

- against historical at 15 minutes: -0.0301 [-0.0437, -0.0163]
- against historical at 60 minutes: -0.0115 [-0.0196, -0.0045]
- against persistence at 15 minutes: -0.0021 [-0.0044, 0.0002]
- against persistence at 60 minutes: -0.0069 [-0.0095, -0.0040]

## Per held-out route (model, then shrunk persistence)

Greater Anglia (lst-cbg) and Chiltern (myb-bmo) never appear in training. bri-wsm still carries about 2 percent of stops with arrival or departure inversions of two minutes or more (Stage 1), so its actual times carry extra noise.

| route | minutes before | n | model MAE | shrunk MAE | model coverage | model Brier | model ECE |
|---|---|---|---|---|---|---|---|
| lst-cbg | 15 | 281 | 2.02 | 2.58 | 0.701 | 0.005 | 0.009 |
| lst-cbg | 60 | 281 | 3.20 | 3.52 | 0.808 | 0.026 | 0.018 |
| myb-bmo | 15 | 193 | 1.66 | 1.66 | 0.788 | 0.011 | 0.007 |
| myb-bmo | 60 | 193 | 2.52 | 2.17 | 0.855 | 0.020 | 0.032 |
| bhm-liv | 15 | 182 | 3.02 | 3.25 | 0.500 | 0.026 | 0.032 |
| bhm-liv | 60 | 182 | 4.19 | 4.07 | 0.637 | 0.042 | 0.046 |
| man-shf | 15 | 337 | 2.73 | 2.98 | 0.754 | 0.039 | 0.025 |
| man-shf | 60 | 337 | 4.24 | 4.76 | 0.795 | 0.068 | 0.028 |
| glq-abd | 15 | 105 | 1.61 | 1.39 | 0.867 | 0.023 | 0.043 |
| glq-abd | 60 | 105 | 3.33 | 3.09 | 0.848 | 0.024 | 0.043 |
| bri-cdf | 15 | 237 | 2.22 | 2.37 | 0.768 | 0.020 | 0.035 |
| bri-cdf | 60 | 237 | 4.13 | 4.14 | 0.658 | 0.051 | 0.044 |
| bri-wsm | 15 | 284 | 1.73 | 1.92 | 0.796 | 0.017 | 0.013 |
| bri-wsm | 60 | 284 | 2.97 | 3.19 | 0.792 | 0.025 | 0.023 |

## Break it (validation rows, held-out routes)

- full model MAE {'15': 2.149, '60': 3.237}; shrunk persistence {'15': 2.354, '60': 3.45}
- without the what-is-ahead features: {'15': 2.183, '60': 3.299}
- shuffled route, rid and operator labels, largest prediction change: 0.0
- missing hour: {'day': '2026-09-28', 'window_utc': '10:00-11:00', 'held_out_rows_after_gap': 87, 'all_predictions_finite': True, 'mae_after_gap_normal': 2.446, 'mae_after_gap_missing': 3.051, 'rows_with_no_actual_normal': 13, 'rows_with_no_actual_missing': 19}
- no trains ahead: {'rows_with_nothing_ahead': 240, 'mae_those_rows_model': 1.8, 'mae_those_rows_shrunk': 1.768, 'all_ahead_features_zeroed_mae': {'15': 2.242, '60': 3.333}, 'finite': True}
- feature importance (share of gain, top 10): {'fastest_min_remaining': 0.1676, 'fc_minus_now_min': 0.0994, 'slack_min': 0.0993, 'sched_min_remaining': 0.0597, 'dest_late_mean_min': 0.0526, 'stops_remaining': 0.0478, 'n_actuals': 0.0459, 'assoc_in_turn_min': 0.0365, 'fc_late_min': 0.0349, 'late_now_min': 0.033}
