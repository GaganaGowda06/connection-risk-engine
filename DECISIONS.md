# Decisions

- 2026-10-14 Package name: rail_risk, src layout.
- 2026-10-14 Routes: BHM to EUS, BHM to MAN, and AST to EUS via BHM.
- 2026-10-14 Repo will be public on GitHub later.
- 2026-10-14 Terraform is plan-only, no AWS.
- 2026-10-14 Interval method: LightGBM quantile models with conformalized quantile regression; fallback empirical residual quantiles per route and hour.
- 2026-10-14 gitleaks via system binary plus full-history CI scan.
- 2026-10-14 Repo public on GitHub; commits use the noreply email.
- 2026-10-14 CI runners pinned to ubuntu-24.04 so the 2026-10-19 ubuntu-latest switch to Ubuntu 26 cannot break CI unannounced.
- 2026-10-15 Cancellation labelling unresolved: no cancelled service appeared in 25 sampled BHM to EUS services, so HSP cancellation representation is unknown. Reason codes are numeric strings on late stops. Needs a larger sample using one hour windows.
- 2026-10-15 Darwin captured unfiltered and resumable via committed offsets; route filtering in Spark.
- 2026-10-15 HSP backfill cost, measured on bhm-eus for 2026-09-30 to 2026-10-02, hours 06-22: 48 windows and 222 services (74 per weekday) took 593 s over 274 HTTP calls (48 metrics, 222 details, 4 retries), about 2.2 s per call at the 1 s gap. Result: 2,574 stop rows, 0 quarantined, 3 candidate cancellations. One window hit HTTP 503 three times, finished on rerun, and a further rerun made 0 calls.
- 2026-10-15 HSP backfill projection: one route-day is about 90 calls and 200 s, so 12 months by 3 routes (1,095 route-days) is about 99,000 calls and 60 hours of API time, an upper bound because weekends carry fewer services and ast-eus rids shared with bhm-eus are fetched once. Volume per day is untested outside these 3 weekdays.
- 2026-10-15 Recommended backfill span: 6 months, bhm-eus and bhm-man first, in 30 day chunks overnight (about 20 hours for those two, 30 for all three), then extend to 12 months only if the delay model needs the seasonality.
- 2026-10-15 HSP route membership: a rid is fetched once and linked to every route that returns it (rid_routes), so reruns for another route add membership without refetching.
- 2026-10-15 HSP timestamps are stored as UTC, converted from Europe/London. Ambiguous local times (clocks back) take the first occurrence unless that would run backwards within the service, then the second. Non-existent local times (clocks forward) use the pre-change offset, which shifts them forward by the gap.
- 2026-10-15 Stage 2 summary: HSP and Darwin both load into validated Parquet with write-once raw files, a manifest per source, UTC timestamps and a quarantine table that keeps rejected rows. HSP: 12 months by 3 routes costs about 60 hours of API time, backfill of bhm-eus and bhm-man running newest first. Darwin: about 1.2 MB per minute stored, parsed at 3 to 5 s per hourly file, unfiltered, with route filtering left to Spark. Cancellation labels will come from Darwin schedule cancelReason and per-stop can flags, checked against HSP candidate_cancelled.
- 2026-10-15 Features: PySpark 4.2.0 on JDK 17, local[4] with 6 GB driver. Snapshots are one row per service, route and horizon (T-120, T-60, T-15) at T = scheduled destination arrival minus the horizon. Live state uses only actuals at or before T, historical features use only services dated 1 to 28 days earlier, and leakage tests (synthetic and 50 real rows) must stay at zero mismatches. Interim HSP files are normalised through Arrow before Spark reads them, because pandas writes nanosecond timestamps and null-typed columns that Spark rejects.
