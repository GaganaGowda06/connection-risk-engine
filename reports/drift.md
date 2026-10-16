# Feature drift

Generated 2026-10-16T09:28:03+00:00. Live serving features from 2026-10-14 to 2026-10-16 (175 services, features built 60 minutes before arrival from Darwin) are compared with the training snapshot (22424 rows at the same horizon). PSI uses 10 quantile bins from training, plus a bin for missing values. Under 0.1 is ok, 0.1 to 0.25 is watch, and 0.25 or more is act.

Summary: 11 ok, 0 watch, 5 act.

| Feature | PSI | Status | Training missing | Live missing |
|---|---|---|---|---|
| day_of_week | 6.178 | act | 0% | 0% |
| toc_code | 3.794 | act | 0% | 41% |
| hist_p80_28d | 1.398 | act | 1% | 3% |
| hist_n_28d | 1.104 | act | 0% | 0% |
| hist_median_28d | 0.561 | act | 1% | 3% |
| last_stop_lateness_min | 0.099 | ok | 57% | 56% |
| sched_minutes_remaining | 0.074 | ok | 57% | 56% |
| hour_of_day | 0.051 | ok | 0% | 0% |
| dest_hour | 0.049 | ok | 0% | 0% |
| stops_remaining | 0.047 | ok | 57% | 56% |
| lateness_change_last_2 | 0.044 | ok | 59% | 57% |
| minutes_since_last_stop | 0.043 | ok | 57% | 56% |
| n_stops_passed | 0.040 | ok | 57% | 56% |
| last_stop_seq | 0.038 | ok | 57% | 56% |
| minutes_to_arrival | 0.000 | ok | 0% | 0% |
| is_bank_holiday | 0.000 | ok | 0% | 0% |

Darwin capture started on 2026-10-15, so the live window holds fewer than 7 days of services. Daily and weekly patterns in the live sample are partial, and PSI on a short window is noisy.
