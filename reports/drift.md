# Feature drift

Generated 2026-10-16T10:16:45+00:00. Live serving features from 2026-10-14 to 2026-10-16 (184 services, features built 60 minutes before arrival from Darwin) are compared with the training snapshot (22424 rows at the same horizon). PSI uses 10 quantile bins from training, plus a bin for missing values. Under 0.1 is ok, 0.1 to 0.25 is watch, and 0.25 or more is act.

Summary: 10 ok, 1 watch, 5 act.

| Feature | PSI | Status | Training missing | Live missing |
|---|---|---|---|---|
| day_of_week | 6.131 | act | 0% | 0% |
| toc_code | 3.572 | act | 0% | 39% |
| hist_p80_28d | 1.332 | act | 1% | 3% |
| hist_n_28d | 1.140 | act | 0% | 0% |
| hist_median_28d | 0.522 | act | 1% | 3% |
| last_stop_lateness_min | 0.101 | watch | 57% | 56% |
| sched_minutes_remaining | 0.057 | ok | 57% | 56% |
| lateness_change_last_2 | 0.044 | ok | 59% | 57% |
| stops_remaining | 0.042 | ok | 57% | 56% |
| n_stops_passed | 0.037 | ok | 57% | 56% |
| last_stop_seq | 0.034 | ok | 57% | 56% |
| minutes_since_last_stop | 0.030 | ok | 57% | 56% |
| hour_of_day | 0.026 | ok | 0% | 0% |
| dest_hour | 0.024 | ok | 0% | 0% |
| minutes_to_arrival | 0.000 | ok | 0% | 0% |
| is_bank_holiday | 0.000 | ok | 0% | 0% |

Live rows with toc_code missing, by service date:

| Service date | Live services | Missing |
|---|---|---|
| 2026-10-15 | 131 | 55% |
| 2026-10-16 | 53 | 0% |

Live rows with hist_p80_28d missing, by service date:

| Service date | Live services | Missing |
|---|---|---|
| 2026-10-15 | 131 | 4% |
| 2026-10-16 | 53 | 0% |

Live rows with hist_median_28d missing, by service date:

| Service date | Live services | Missing |
|---|---|---|
| 2026-10-15 | 131 | 4% |
| 2026-10-16 | 53 | 0% |

Live rows with last_stop_lateness_min missing, by service date:

| Service date | Live services | Missing |
|---|---|---|
| 2026-10-15 | 131 | 55% |
| 2026-10-16 | 53 | 58% |

Live rows with sched_minutes_remaining missing, by service date:

| Service date | Live services | Missing |
|---|---|---|
| 2026-10-15 | 131 | 55% |
| 2026-10-16 | 53 | 58% |

Live rows with lateness_change_last_2 missing, by service date:

| Service date | Live services | Missing |
|---|---|---|
| 2026-10-15 | 131 | 55% |
| 2026-10-16 | 53 | 60% |

Live rows with stops_remaining missing, by service date:

| Service date | Live services | Missing |
|---|---|---|
| 2026-10-15 | 131 | 55% |
| 2026-10-16 | 53 | 58% |

Live rows with n_stops_passed missing, by service date:

| Service date | Live services | Missing |
|---|---|---|
| 2026-10-15 | 131 | 55% |
| 2026-10-16 | 53 | 58% |

Live rows with last_stop_seq missing, by service date:

| Service date | Live services | Missing |
|---|---|---|
| 2026-10-15 | 131 | 55% |
| 2026-10-16 | 53 | 58% |

Live rows with minutes_since_last_stop missing, by service date:

| Service date | Live services | Missing |
|---|---|---|
| 2026-10-15 | 131 | 55% |
| 2026-10-16 | 53 | 58% |

Darwin capture started on 2026-10-15 at 10:31 UTC, so the live window holds fewer than 7 days of services. Daily and weekly patterns in the live sample are partial, and PSI on a short window is noisy. Services that were scheduled before the capture began have no schedule message in the raw data, so their operator is missing; see the table by service date.
