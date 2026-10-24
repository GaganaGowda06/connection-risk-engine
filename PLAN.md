# PLAN

Network-wide rebuild of the Connection Risk Engine. GOAL.md holds the nine pass/fail tests; every stage below says which it must pass. Current stage: **Stage 1 (data foundation)**, sub-step 1.1 done (probe and mirror).

## Approach (from Stage 0 research, decisions and evidence in DECISIONS.md)
- History: the community Darwin Push Port archive (ilovetrains.co.uk), mirrored to disk. It is unverified for licence and has no SLA, so everything needed is mirrored.
- Window: 2025-12-01 to 2026-02-28 (February held out as the winter stress test) and 2026-07-25 to 2026-10-23.
- Model: one route-agnostic LightGBM with quantiles 0.1, 0.5 and 0.9, a calibrated P(delay at least 15) and a P(miss) classifier. No route_id. Features: segment, station, network state, weather. Cross-conformal calibration on leave-route-group-out folds, with Mondrian groups only on transferable variables (length class, horizon, delay band). Fallback: a slack-recovery model.
- Minimum connection times: parse the RDG MSN and TSI files, never hard-code. BHM reads as 12 minutes in MSN, unverified until parsed.
- Weather: Met Office DataHub for live use; the 2026 history source is open. Hosting: Oracle Cloud Always Free (2 OCPU, 12 GB), moved forward to the Oracle capture host step so capture stops depending on the laptop. Keep resident memory above about 2.4 GB to avoid idle reclaim.
- Out-of-time week: 2026-10-17 to 2026-10-23 is split=oot_week. It is never trained, tuned or calibrated on (GOAL test 6a) and is replayed through the service in Stage 6.
- Capture stopgap: our own capture started on the Mac on 2026-10-24 and records the full GB Darwin feed (about 3.9 GB a day), with route filtering at parse time. Start `nohup caffeinate -i scripts/run_capture.sh > capture.log 2>&1 &`, check `pgrep -fl capture_darwin` and `tail logs/capture.log`, stop `pkill -INT -f scripts/capture_darwin.py`.

## Routes
25 routes across 11 operators are planned. Stage 0 named only the seven held-out routes below; the other 18 and the operator count are not yet in the repo and are fixed in Stage 1.2. Nothing here is a trained-on claim until that list is written to config/routes.yaml.

| Route | Description | Held out |
|---|---|---|
| LST-CBG | London Liverpool Street to Cambridge | yes |
| MYB-BMO | London Marylebone to Birmingham Moor Street | yes |
| BHM-LIV | Birmingham New Street to Liverpool Lime Street | yes |
| MAN-SHF | Manchester Piccadilly to Sheffield | yes |
| GLQ-ABD | Glasgow Queen Street to Aberdeen | yes |
| BRI-CDF | Bristol Temple Meads to Cardiff Central | yes |
| BRI-WSM | Bristol Temple Meads to Weston-super-Mare | yes |
| BHM-EUS, AST-BHM, BHM-MAN (in config today, trained on by the old champion) | Birmingham to Euston, Aston to Birmingham, Birmingham to Manchester | no |
| 15 further routes | to be named in Stage 1.2 | no |

Held-out routes never enter training, tuning or calibration folds, and no feature is fitted on them.

## Stages

### Stage 1: data foundation (GOAL tests 1, 7, 8) - CURRENT
Scope: mirror the archive (1.1, in progress), parse Darwin XML v18 into per-stop tables with one set of cleaning checks and no route-specific code, parse MSN and TSI for minimum connection times, pick the 25 routes, load timetable and planned-closure information.
Split: all dates from 2026-10-17 to 2026-10-23 are marked split=oot_week in the clean tables and excluded from every fit; February 2026 is the winter hold-out.
Done means: at least 20 routes, 5 operators, short, medium and long journeys and a major hub in the clean tables (test 1); every route passes the same checks for types, timezones, station matching, and planned closure versus missing data (test 7); every source and licence is recorded and no number is unsourced (test 8).
Stage 1.1 result: see DECISIONS.md 2026-10-24.

### Oracle capture host (GOAL test 6b)
Scope: provision Oracle Cloud Always Free (2 OCPU, 12 GB, resident memory above about 2.4 GB to avoid idle reclaim), run the Darwin capture there, and stop depending on the laptop.
Done means: capture running on Oracle; the Mac capture stopped only after an overlap check shows no hole at the switch; capture loss measured against the archive for the same hours and stated with the days observed (target 7 days, under 1% loss).
Open issues carried to 6b: the Mac capture on 2026-10-16 was short in hours 12 to 16 (message ratios 0.89, 0.32, 0.57, 0.74, 0.75 against the archive) and had 2.4 to 2.5 times the archive's messages in hours 17 and 18 (duplicates or redelivery, cause not checked). Both must be understood before the loss figure is trusted.

### Stage 2: baselines (supports tests 2 to 5)
Scope: persistence, historical route-hour baseline and gap-bucket baseline on every route, fitted on earlier dates only; split=oot_week rows excluded.
Done means: baseline tables per route and horizon with bootstrap CIs by service date, reproducible from one command, so tests 2 to 5 have a fixed bar.

### Stage 3: route-agnostic LightGBM (test 2)
Scope: one model, no route_id, leave-route-group-out training; split=oot_week never used.
Done means: on held-out routes, arrival MAE beats persistence and the historical baseline at 15 and 60 minutes, paired 95% CI below zero, with run ids recorded.

### Stage 4: calibration (tests 3, 4, plus a separate February stress report)
Scope: cross-conformal ranges with transferable Mondrian groups, calibrated P(delay at least 15); split=oot_week never used for calibration.
Done means: 80% ranges cover 0.78 to 0.82 on held-out routes overall and no route below 0.70 (test 3); P(delay at least 15) beats the historical Brier with ECE at or below 0.05 (test 4); February 2026 is scored once, reported separately, and never used to tune.

### Stage 5: connections at BHM, MAN and BRI (test 5)
Scope: P(miss) classifier with parsed minimum connection times.
Done means: P(miss) beats the gap-bucket baseline on Brier at two or more hubs, ECE at or below 0.05 on pairs whose inbound has not yet arrived.

### Stage 6: live service on an always-on host (tests 6a, 6b, 9)
Scope: the API on the Oracle host beside the capture, weather from the Met Office DataHub, and a replay of the split=oot_week week (2026-10-17 to 2026-10-23) through the service.
Done means: the out-of-time week replayed through the service and scored once (test 6a); the capture record from the Oracle step stated honestly in the README with actual days and loss (test 6b); p95 under 2 seconds, visible degradation when data is stale, automatic recovery after a restart (test 9).

### Stage 7: traceability and demo (test 8)
Scope: every number on screen links to a model output or a verified rule; Delay Repay amounts only for verified operators.
Done means: an audit of the demo with no untraced number, and the README states every GOAL test with evidence and run ids.
