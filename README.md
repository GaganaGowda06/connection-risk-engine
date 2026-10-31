# Connection Risk Engine

Calibrated arrival ranges and connection risk for GB trains, built on open Darwin data and tested on routes the model never trained on.

## What it does

For a passenger: a train that shows "on time" until the last minute is replaced by an honest range ("expected 09:49 to 09:54", built so that about 8 in 10 trains land inside it), the chance of being 15 minutes or more late, and the chance of missing a connection at Birmingham New Street, Manchester Piccadilly or Bristol Temple Meads. The network service prices nothing: it returns ranges and probabilities, never a money amount.

For an engineer: one LightGBM model with no route, station or operator identity predicts arrival delay for any route. Online conformal calibration keeps the 80% ranges honest per route. A second model scores connection pairs with parsed minimum connection times. A FastAPI service answers `/risk`, `/connection`, `/inbound` and `/health`, and flags stale input.

## Results

All tests run on routes held out of training. Test block: 2026-10-10 to 2026-10-16. Intervals are 95% bootstrap CIs over service dates. GOAL.md defines each test.

| GOAL test | Result | Key numbers | Evidence |
|---|---|---|---|
| 1 Breadth | Met | 25 routes, 11 operators, 5 hubs, short, medium and long; 7 held-out routes | `config/routes.yaml`, `reports/stage1_coverage.csv` |
| 2 Accuracy | Met | MAE 2.22 (15 min) and 3.67 (60 min) against persistence 2.41 and 3.82 and historical 4.03; all paired CIs below zero | `reports/network_model/stage4/test_results.json`, run 905346cb (retrained clean as 1e197857) |
| 3 Ranges | Met on the second scoring | 80% coverage 0.803 overall [0.788, 0.817], worst route 0.791 (per-route ACI). Static CQR scored first and left bhm-liv at 0.58 | `reports/network_model/stage4b/second_scoring_results.json`, run 1e197857 |
| 4 Late risk | Met | Brier 0.021 and 0.040 against historical 0.051; ECE 0.013 and 0.012 | same file; calibration run c0a08ab4 |
| 5 Connections | Met after a label fix | Beats the gap-bucket baseline at BHM, MAN and BRI, all horizons, CIs below zero; ECE at most 0.029 | `reports/connections/stage5/test_relabel_comparison.md`, run cd4ac6e9 |
| 6a Out of time | Met | 11 of 13 metrics within the test CIs, the other 2 better than expected | `reports/serving/oot_table_6a.md`, runs 1e197857, c0a08ab4, cd4ac6e9 |
| 6b Live pipeline | Not met | The laptop capture lost data while the Mac slept; 54% to 80% of messages missing per day, measured on 17 to 23 October | `reports/capture_loss/` |
| 7 Data quality | Met with caveats | The same checks run on every route; no route-specific code | `reports/stage1_cleaning.csv` |
| 8 Trust | Met for the network service and this README | All 55 numeric and time fields in `/health`, `/risk`, `/connection`, `/inbound` and `/services` name their source in the response (run ids, MCT file and sequence, message clock, guard file); every README result number found in the file it cites; no Delay Repay amount is returned | `tests/test_traceability.py`, `scripts/audit_readme_numbers.py`, `reports/stage7_audit.md`; runs 1e197857, c0a08ab4, cd4ac6e9 |
| 9 Reliability | Met | p95 51 ms at 200 requests a second; capacity about 400 a second on one Mac; restart recovery 1.3 s; visible stale flag | `reports/serving/loadtest_w4/SUMMARY.md`, `recovery_after_sigkill.json`; same three runs. After the provenance block was added, a host run of the 4-worker service gave p95 13 ms at 200 requests a second with 0 errors (`reports/serving/loadtest_provenance_200rps.json`; a different setup from the Docker run, so not a like-for-like figure) |

Test 5 also beats a logistic-regression option on validation (Brier 0.043 against 0.059) and on the out-of-time week at BHM and MAN. At BRI the interval includes zero. A February 2026 winter stress run is reported on its own: adaptive coverage 0.801, worst route 0.797 (`reports/network_model/stage4b/second_scoring_results.json`).

## How it was evaluated

Routes are split by group. Seven routes (lst-cbg, myb-bmo, bhm-liv, man-shf, glq-abd, bri-cdf, bri-wsm) are never used to train, tune or calibrate. Dates are split in time: train 2025-12-01 to 2026-09-23, validation 2026-09-25 to 2026-10-08, test 2026-10-10 to 2026-10-16. The most recent archive week, 2026-10-17 to 2026-10-23, is kept out of everything and replayed through the service once. February 2026 is scored once and never tuned on. Differences against baselines are paired bootstrap intervals over service dates. Each test block was scored once; the one exception is the second scoring in test 3, recorded in DECISIONS.md.

## How it works

History comes from the community Darwin archive. DuckDB builds point-in-time snapshots from Parquet at 5 to 120 minutes before arrival, using only messages stamped before the forecast time; leakage tests rerun snapshots on real data. LightGBM predicts a residual over shrunk persistence. Conformal quantile regression sets the ranges, and adaptive conformal inference (ACI) updates them per route as outcomes arrive. An isotonic-calibrated classifier gives P(late). Connections use a LightGBM classifier with a physical estimate from the arrival quantiles as its main feature. MLflow tracks runs. The network service does not use the `champion` alias: it names the three frozen runs by id in `src/rail_risk/serving/artifacts.py` and serves a SQLite ledger of scores that the replay engine computed from them. Only the legacy two-route API loads its model by the `champion` alias. FastAPI serves the ledger in Docker. A freshness guard flags silent feeds and blends toward the gap-bucket baseline. Every response has a `provenance` block that names the source of each numeric field, and the service refuses (503) a ledger scored by other runs than the three it names, or a connection answer whose MCT file is unknown. GitHub Actions runs tests and lint. Spark builds features only for the older two-route model and the original API wiring; the network model, the connection model and the replay service use DuckDB.

## Limitations

- Incidents that start after the prediction cannot be foreseen. On the 18 services that ran 45 minutes or more late in one incident, coverage was 0.44.
- At 60 minutes the model only ties a route-history persistence baseline (difference -0.016 [-0.094, +0.058]).
- On the one real outage in the archive, the stale-data guard scored worse than no guard on raw labels (Brier 0.0861 against 0.0379). Many of those labels are "miss" only because the archive lacks the arrival.
- P(miss) calibration drifted out of time (ECE up to 0.035) but stayed under 0.05.
- It does not foresee a late outbound train saving a connection.
- Only three hubs were tested for connections.
- Live capture is not proven (test 6b). My laptop capture lost 54% to 80% of messages per day, measured on 17 to 23 October, while the Mac slept, and no always-on host was run.

## In progress

I am capturing live train position from the Network Rail Train Describer feed. No result is claimed yet.

## Run it

You need Docker (Colima works). No keys, no data and no Java are needed for this path. From a fresh clone:

```
git clone https://github.com/GaganaGowda06/connection-risk-engine.git
cd connection-risk-engine
docker compose up --build api
curl -s localhost:8000/health
curl -s -X POST localhost:8000/risk -H 'content-type: application/json' -d '{"rid":"202610178700432"}'
curl -s -X POST localhost:8000/connection -H 'content-type: application/json' -d '{"in_rid":"202610176700683","out_rid":"202610176742156"}'
docker compose down
```

Stop anything already listening on port 8000 first. With Colima, clone under your home folder, because that is the only host path it shares with containers. This starts the Stage 6 network service, the SQLite-backed FastAPI with `/risk`, `/connection`, `/inbound`, `/health` and the freshness guard. If the real replay ledger (`data/interim/serve/state_oot`, built from your own RDM data by the stage scripts) is missing, it serves the committed sample ledger in `data/sample/`: derived scores for one replay hour (2026-10-17 18:00 to 19:00 UTC), about 2.5 MB, no raw Darwin or Train Describer data. Only services scored in that hour answer. `/health` and every response carry the run ids 1e197857 (arrival), c0a08ab4 (calibration) and cd4ac6e9 (P(miss)) and the `stale` flag. There is no new web page for it.

To test the code: `uv sync`, `cp .env.example .env`, `uv run pytest`. Java 17 is needed only for the legacy Spark code and the tests that cover it (`brew install openjdk@17` on a Mac); the network service and its image do not use it.

Rebuilding results needs your own RDM keys and data: `rail-risk evaluate-baselines`, `train-network`, `evaluate-network --run-id RUN`, then the scripts in `scripts/` (for example `stage4_calibrate.py`, `stage4_score_test.py`, `stage5_build.py`, `make_sample_state.py`). PLAN.md and DECISIONS.md record the order and the run ids.

## Legacy

The two-route model that came before the network rebuild is kept, and its reports are in `reports/legacy/`. Its Spark API and Streamlit page run under a compose profile: `docker compose --profile legacy up --build legacy-api legacy-ui`, with the API on port 8100 and the page on 8501. Its model is loaded by the MLflow `champion` alias. Its synthetic demo, which needs Java 17:

```
uv run rail-risk check
uv run python scripts/make_fixture_store.py /tmp/rail-demo
RAIL_DATA_DIR=/tmp/rail-demo/data RAIL_MLFLOW_DIR=/tmp/rail-demo/mlruns uv run rail-risk serve --port 8011
curl -s localhost:8011/health
```

The demo store is synthetic, so `/health` reports `degraded`.

## Data and licences

- Darwin Push Port under the RDM Open licence. Darwin may not be resold unmodified, so commercial users should check that condition.
- Historical archive: the community Darwin archive at ilovetrains.co.uk, credited here. It states no terms of its own.
- Powered by National Rail Enquiries, for the RDG timetable feed (minimum connection times).
- Station codes: NaPTAN (Open Government Licence) joined to the davwheat/uk-railway-stations list (ODbL).
- Network Rail open data (Train Describer): I have not read its licence text yet.
- No raw rail data is committed. Code is under the MIT licence.

## Delay Repay

The network service does not price Delay Repay and returns no money amounts. Only the legacy two-route API uses `src/rail_risk/repay.py`, which reads `config/delay_repay.yaml` and prices a payout as a share of the ticket price you enter, by minutes late at the final destination. Only Avanti West Coast is priced. Its rules were read from the operator's own page on 2026-10-15. West Midlands Railway's rules were also read that day, by hand, but they are not used: TOC code LM is shared with London Northwestern Railway, whose rules are unverified, and a TOC code cannot tell them apart, so LM answers "not priced: rules unverified". An unverified or unknown operator raises an error rather than a number. The rules were last checked on 2026-10-15 and are not re-checked until a service prices from them.
