# Feature drift

Generated 2026-10-15T19:58:12+00:00. Live serving features from 2026-10-14 to 2026-10-15 (53 services, features built 60 minutes before arrival from Darwin) are compared with the training snapshot (6027 rows at the same horizon). PSI uses 10 quantile bins from training, plus a bin for missing values. Under 0.1 is ok, 0.1 to 0.25 is watch, and 0.25 or more is act.

Summary: 1 ok, 4 watch, 10 act.

| Feature | PSI | Status | Training missing | Live missing |
|---|---|---|---|---|
| day_of_week | 8.099 | act | 0% | 0% |
| hist_n_28d | 4.553 | act | 0% | 0% |
| hist_median_28d | 4.461 | act | 1% | 0% |
| hist_p80_28d | 4.070 | act | 1% | 0% |
| toc_code | 1.368 | act | 0% | 17% |
| last_stop_lateness_min | 0.352 | act | 2% | 2% |
| n_stops_passed | 0.351 | act | 2% | 2% |
| stops_remaining | 0.339 | act | 2% | 2% |
| hour_of_day | 0.299 | act | 0% | 0% |
| sched_minutes_remaining | 0.268 | act | 2% | 2% |
| last_stop_seq | 0.238 | watch | 2% | 2% |
| dest_hour | 0.215 | watch | 0% | 0% |
| minutes_since_last_stop | 0.208 | watch | 2% | 2% |
| lateness_change_last_2 | 0.182 | watch | 7% | 2% |
| is_bank_holiday | 0.000 | ok | 0% | 0% |

Darwin capture started on 2026-10-15, so the live window holds fewer than 7 days of services. Daily and weekly patterns in the live sample are partial, and PSI on a short window is noisy.
