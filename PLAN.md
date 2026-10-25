# PLAN

Network-wide rebuild of the Connection Risk Engine. GOAL.md holds the nine pass/fail tests; every stage below says which it must pass. Current stage: **Stage 5 (connections)**; Stages 1 to 4 are met.

## Approach (from Stage 0 research, decisions and evidence in DECISIONS.md)
- History: the community Darwin Push Port archive (ilovetrains.co.uk), mirrored to disk. It is unverified for licence and has no SLA, so everything needed is mirrored.
- Window: 2025-12-01 to 2026-02-28 (February held out as the winter stress test) and 2026-07-25 to 2026-10-23.
- Model: one route-agnostic LightGBM with quantiles 0.1, 0.5 and 0.9, a calibrated P(delay at least 15) and a P(miss) classifier. No route_id. Features: segment, station, network state, weather. Cross-conformal calibration on leave-route-group-out folds, with Mondrian groups only on transferable variables (length class, horizon, delay band). Fallback: a slack-recovery model.
- Minimum connection times: parse the RDG MSN and TSI files, never hard-code. BHM reads as 12 minutes in MSN, unverified until parsed.
- Weather: Met Office DataHub for live use; the 2026 history source is open. Hosting: Oracle Cloud Always Free (2 OCPU, 12 GB), moved forward to the Oracle capture host step so capture stops depending on the laptop. Keep resident memory above about 2.4 GB to avoid idle reclaim.
- Out-of-time week: 2026-10-17 to 2026-10-23 is split=oot_week. It is never trained, tuned or calibrated on (GOAL test 6a) and is replayed through the service in Stage 6.
- Capture stopgap: our own capture started on the Mac on 2026-10-24 and records the full GB Darwin feed (about 3.9 GB a day), with route filtering at parse time. Start `nohup caffeinate -i scripts/run_capture.sh > capture.log 2>&1 &`, check `pgrep -fl capture_darwin` and `tail logs/capture.log`, stop `pkill -INT -f scripts/capture_darwin.py`.

## Routes
The 25-route table below was set by me on 2026-10-25 and is written to config/routes.yaml, the single source (route codes appear nowhere else in src/). 11 operators run on 10 Darwin TOC codes: London Northwestern and West Midlands Railway both use LM. Three older ids (ast-eus-via-bhm, bhm-lds, bhm-glc) stay in the file flagged `legacy` for the two-route stepping-stone code; they are not part of the 25 and are left out of coverage.

| Hub | Route | Operator | Length | Set |
|---|---|---|---|---|
| London | EUS-MAN | Avanti West Coast | Long | Train |
| London | KGX-LDS | LNER | Long | Train |
| London | PAD-BRI | GWR | Medium | Train |
| London | EUS-WFJ | London Northwestern | Short | Train |
| London | LST-CBG | Greater Anglia | Medium | Held out |
| London | MYB-BMO | Chiltern | Long | Held out |
| Birmingham | BHM-EUS | Avanti West Coast | Medium | Train |
| Birmingham | AST-BHM | West Midlands Railway | Short | Train |
| Birmingham | BHM-MAN | CrossCountry | Medium | Train |
| Birmingham | BHM-BRI | CrossCountry | Medium | Train |
| Birmingham | BHM-LIV | London Northwestern | Medium | Held out |
| Birmingham | BHM-WVH | West Midlands Railway | Short | Train |
| Manchester | MAN-LDS | TransPennine Express | Medium | Train |
| Manchester | MCV-LIV | Northern | Medium | Train |
| Manchester | MAN-SHF | Northern | Medium | Held out |
| Manchester | MAN-MIA | Northern | Short | Train |
| Manchester | MAN-GLC | TransPennine Express | Long | Train |
| Glasgow | GLC-EUS | Avanti West Coast | Long | Train |
| Glasgow | GLQ-EDB | ScotRail | Medium | Train |
| Glasgow | GLC-PYG | ScotRail | Short | Train |
| Glasgow | GLC-AYR | ScotRail | Medium | Train |
| Glasgow | GLQ-ABD | ScotRail | Long | Held out |
| Bristol | BRI-CDF | GWR | Medium | Held out |
| Bristol | BRI-BTH | GWR | Short | Train |
| Bristol | BRI-WSM | GWR | Short | Held out |

Interchange pairs (also in config/routes.yaml): BHM (BRI-BHM CrossCountry into BHM-EUS; MAN-BHM CrossCountry into BHM-LIV), MAN (LDS-MAN TransPennine into MAN-MIA), BRI (PAD-BRI into BRI-CDF).

Held-out routes never enter training, tuning or calibration folds, and no feature is fitted on them.

## Stages

### Stage 1: data foundation (GOAL tests 1, 7, 8) - MET 2026-10-25
Scope: mirror the archive (1.1, in progress), parse Darwin XML v18 into per-stop tables with one set of cleaning checks and no route-specific code, parse MSN and TSI for minimum connection times, pick the 25 routes, load timetable and planned-closure information.
Split: all dates from 2026-10-17 to 2026-10-23 are marked split=oot_week in the clean tables and excluded from every fit; February 2026 is the winter hold-out.
Done means: at least 20 routes, 5 operators, short, medium and long journeys and a major hub in the clean tables (test 1); every route passes the same checks for types, timezones, station matching, and planned closure versus missing data (test 7); every source and licence is recorded and no number is unsourced (test 8).
Progress (details and evidence in DECISIONS.md, 2026-10-24):
- 1.1 done: archive probe and mirror.
- 1.2 done: mirror of the whole window (4,342 files) plus six boundary and clock-change days, manifest and integrity check; network-wide streaming extraction to Parquet (schedules, schedule_points, stop_updates, stop_actuals, associations, deactivations) with the split column; TIPLOC to CRS from open files; one pandera cleaning module run on 187 service dates; coverage; HSP spot-check (730 of 730 arrivals and departures exact on 50 trains of three held-out routes); weather source recommendation (Open-Meteo, not ingested).
- 1.3 done (2026-10-25): 25-route table in config/routes.yaml; coverage (reports/stage1_coverage.csv); sequence-flag cause found and a general resolution rule applied (DECISIONS.md). MSN and TSI parsed in Stage 2 (data/ref/mct.csv). Still to do, not blocking: the rest of 2026-10-24, planned-closure and engineering information if a source is found.
- Stage 1 done-means, checked 2026-10-25:
  - Test 1 (breadth): MET. 25 routes, 11 operators (10 TOC codes), 5 hubs, short, medium and long, four interchange pairs; every route has trains in all 7 window months (reports/stage1_coverage.csv). Thin: mcv-liv (21 trains in its quietest month), man-glc (54).
  - Test 7 (data quality): MET with a caveat. The same pandera checks run on every route and the network (reports/stage1_cleaning.csv), a test fails on any CRS or route literal in the clean code. PLAN set no numeric pass threshold, so "passes" means checked and published; residual flags: duplicates 19,608 of 1.78 billion updates, station matching 100% on route services, impossible negative dwell 31,443 of 4.76 million stops network-wide (bri-wsm 175 of 7,864 is the highest route), 2.4% of passenger schedules are missing data.
  - Test 8 (trust): MET by my decision of 2026-10-25. The archive's own terms are unstated; the data is the RDM Darwin Push Port product under the RDM Open licence (already recorded), and the archive (ilovetrains.co.uk) is credited in the README and DECISIONS.md. An email to its owner is optional. Verified: station list ODbL 1.0, NaPTAN OGL, Open-Meteo CC-BY 4.0 non-commercial for development. MIDAS and Meteostat stay unverified and unused.
  - Stage 1 is met; GOAL.md's last line is Stage 2. The three legacy routes stay as they are.
Weather: Open-Meteo (ERA5) is the recommended history source (free for development, paid key before commercial use, newest 5 days null, modelled not observed); not ingested until Stage 3 needs it.
Extract layout: data/interim/pport/<table>/service_date=D/<hour file>.parquet, read model data only through rail_risk.ingest.pport_extract.train_pool_table (train_pool rows only).

### Oracle capture host (GOAL test 6b)
Status: DROPPED (2026-10-25): live capture is stopped for good, 6b is not run and the README says so. Earlier status: BLOCKED (2026-10-25). Oracle Cloud signup failed twice with a generic account-creation error. Mac capture continues as a free stopgap under launchd with a daily loss check; 6b fails by design while capture depends on the laptop. Reopens on a successful signup from my own device or another free always-on host.
Scope: provision Oracle Cloud Always Free (2 OCPU, 12 GB, resident memory above about 2.4 GB to avoid idle reclaim), run the Darwin capture there, and stop depending on the laptop.
Done means: capture running on Oracle; the Mac capture stopped only after an overlap check shows no hole at the switch; capture loss measured against the archive for the same hours and stated with the days observed (target 7 days, under 1% loss).
Open issues carried to 6b: the Mac capture on 2026-10-16 was short in hours 12 to 16 (message ratios 0.89, 0.32, 0.57, 0.74, 0.75 against the archive) and had 2.4 to 2.5 times the archive's messages in hours 17 and 18 (duplicates or redelivery, cause not checked). Both must be understood before the loss figure is trusted.

### Stage 2: baselines (supports tests 2 to 5) - MET 2026-10-25
Scope: persistence, historical route-hour baseline, shrunk persistence and gap-bucket baseline on every route or interchange pair, fitted on earlier dates only; split=oot_week and winter_test rows excluded.
Done means: baseline tables per route and horizon with bootstrap CIs by service date, reproducible from one command, so tests 2 to 5 have a fixed bar. Met: `rail-risk evaluate-baselines` writes reports/baselines/stage2_baselines.{csv,md,json} and stage2_gap_bucket.csv (run 8d3c1769aec3, commit 0e74539; restated 2026-10-25 as run 0e3633af27af on commit 664185a after a target fix, see DECISIONS.md).
Evidence (DECISIONS.md, 2026-10-25): DuckDB chosen over Polars and Spark; split train 2025-12-01 to 2026-09-23, validation 2026-09-25 to 2026-10-08, test 2026-10-10 to 2026-10-16; leakage check 0 mismatches on 894 real snapshots; MCT from RJTTF995.ZIP (BHM 12, MAN 10, BRI 10, no TSI overrides); gap-bucket Brier 0.085 (BHM), 0.096 (MAN), 0.052 (BRI) with the ECE of each recorded; shrunk persistence is the strongest point forecast at 60 minutes.
Caveats: the NRDP licence text as accepted was not readable (README and DECISIONS say so); the NRDP account needs a download by 2026-11-24 and the portal closes later in 2026; no inbound arrived early enough for the not-yet-arrived subset to differ from all pairs.

### Stage 3: route-agnostic LightGBM (test 2) - MET 2026-10-25
Scope: one model, no route_id, leave-route-group-out training; split=oot_week never used.
Done means: on held-out routes, arrival MAE beats persistence and the historical baseline at 15 and 60 minutes, paired 95% CI below zero, with run ids recorded. Met (and also against shrunk persistence, which I added at Stage 3): MLflow run 71d0bc8d7f8e4803963f9af3ecd8d1ed, `rail-risk train-network` then `rail-risk evaluate-network --run-id RUN`, reports/network_model/stage3_report.md. Test block, held-out routes: MAE 2.19 and 3.55 against persistence 2.41 and 3.82, shrunk persistence 2.41 and 3.69, historical 4.04; all six paired CIs below zero.
Note on PLAN vs Stage 3 scope: PLAN set persistence and historical as the bar; Stage 3 also gates on shrunk persistence (the stricter bar) and asks for coverage, Brier and ECE now. Both are reported.
Also found: test 3 NOT met by raw ranges (coverage 0.737 at 15 and 0.770 at 60; bhm-liv 0.500 at 15, bri-cdf 0.658 at 60), test 4 met on this block (Brier 0.021 and 0.039 against 0.051, ECE 0.014 and 0.015). Stage 4 carries test 3.
Caveats: a Stage 2 target defect (day-late duplicate rows) was found and fixed, the Stage 2 bar restated (DECISIONS.md); weather not used (not ingested); what-is-ahead features add only about 2 percent; the worst errors are incidents announced after T; myb-bmo is worse than shrunk persistence at 60 minutes; tuned at the grid edge.

### Stage 4: calibration (tests 3, 4, plus a separate February stress report) - MET 2026-10-25
Scope: out-of-route CQR ranges followed online per route (ACI, gamma 0.05, chosen on validation), calibrated P(delay at least 15); split=oot_week never used; February scored once for the stress report and never used to tune.
Done means: 80% ranges cover 0.78 to 0.82 on held-out routes overall and no route below 0.70 (test 3); P(delay at least 15) beats the historical Brier with ECE at or below 0.05 (test 4); February 2026 scored once. Met: model run 1e197857e96a41e8ad5d350d204307ed and calibration run c0a08ab4dfd8453380d3459a537ab27d (both from clean commits); test 3 met on the SECOND scoring of the test block (the first, static CQR, failed on bhm-liv): overall 0.803 [0.788, 0.817], lowest route 0.791 (bhm-liv); test 4 met (Brier 0.021 and 0.040 against 0.051, ECE 0.013 and 0.012); February adaptive overall 0.801, lowest route 0.797.
Cause of the earlier failure, tested: not seasonal drift outside the training dates. bhm-liv changed abruptly in mid-September together with mcv-liv (both end at Liverpool), trains gaining time, no retiming; lst-cbg is under-covered all winter including inside the train block.
Limits: the ranges need a live feed of arrival outcomes and a route's first day is the static CQR; an incident is followed, not anticipated; the model does not beat the history-aware shrunk persistence at 60 minutes on the test block; the Liverpool-end cause is unknown.
Also done: permanent target sanity check, adaptive leakage test on real data (0 of 360 reruns differ), failure tests, wider grid (no change).

### Stage 5: connections at BHM, MAN and BRI (test 5)
Scope: P(miss) classifier with parsed minimum connection times.
Done means: P(miss) beats the gap-bucket baseline on Brier at two or more hubs, ECE at or below 0.05 on pairs whose inbound has not yet arrived.

### Stage 6: live service on an always-on host (tests 6a, 6b, 9)
Scope: the API on the Oracle host beside the capture, weather from the Met Office DataHub, and a replay of the split=oot_week week (2026-10-17 to 2026-10-23) through the service.
Done means: the out-of-time week replayed through the service and scored once (test 6a); the capture record from the Oracle step stated honestly in the README with actual days and loss (test 6b); p95 under 2 seconds, visible degradation when data is stale, automatic recovery after a restart (test 9).

### Stage 7: traceability and demo (test 8)
Scope: every number on screen links to a model output or a verified rule; Delay Repay amounts only for verified operators.
Done means: an audit of the demo with no untraced number, and the README states every GOAL test with evidence and run ids.
