# Connection Risk Engine

Calibrated UK rail arrival ranges and connection risk on open Darwin and HSP data.
Routes: BHM to EUS, BHM to MAN, AST to EUS via BHM (see config/routes.yaml).

## Commands
- Setup: `uv sync`, then `cp .env.example .env` and fill in keys
- Test: `uv run pytest`
- Lint: `uv run ruff check . && uv run ruff format --check .`
- Run: `uv run rail-risk check`
- Java: PySpark 4.2 needs JDK 17. `brew install openjdk@17`; `src/rail_risk/spark.py` sets
  JAVA_HOME to `/opt/homebrew/opt/openjdk@17` when it is unset. CI uses Temurin 17.
- Build features: `uv run rail-risk build-features [--start D --end D] [--skip-darwin]` (parses Darwin first; snapshots at 5, 15, 30, 60, 90 and 120 minutes before arrival, with `minutes_to_arrival` as a feature)
- Baselines: `uv run rail-risk evaluate-baselines` (writes reports/baselines/ROUTE/baselines.md (the .json is git-ignored, over 500 KB), one set per route)
- Train: `uv run rail-risk train` (smoke below `min_train_days` or on a dirty tree; one model over the routes in config/split.yaml, scored per route; logs to mlruns/mlflow.db, MLflow sqlite backend). Reportable runs write reports/model_results.md, so commit first.
- Connection: `uv run rail-risk connection --inbound-rid RID [--outbound-dep HH:MM] --date D --fare 50 --ticket single` (no outbound: first feasible)
- Backtest: `uv run rail-risk backtest-connections` (ast-bhm inbound against bhm-eus outbound, out of sample on validation and test dates; writes reports/connection_backtest.json, the connections section of reports/model_results.md, and alert thresholds from config/alerts.yaml; run after `train`)
- Serve: `uv run rail-risk serve` (localhost:8000; loads model alias `champion`, set with `rail-risk promote --run-id RUN`)
  - `curl -s -X POST localhost:8000/risk -H 'content-type: application/json' -d '{"rid":"202610158700587","fare":50,"ticket":"single"}'`
  - `curl -s localhost:8000/health`
  - Connection (Aston to Euston via New Street): `curl -s localhost:8000/inbound` lists inbound services, then `curl -s -X POST localhost:8000/connection -H 'content-type: application/json' -d '{"inbound_rid":"RID","outbound_departure":"10:06","fare":50,"ticket":"single"}'` (omit outbound_departure for the first feasible train). The alert flag uses tau from config/alerts.yaml; 503 means the outbound history did not load.
- UI: start `rail-risk serve`, then `uv run rail-risk ui` (Streamlit on localhost:8501, calls the API at `RAIL_API_URL`, default http://127.0.0.1:8000). Two tabs: Train arrival and My connection.
- Bench: `uv run python scripts/bench_api.py` (p50 and p95 against a running API; never part of CI)
- Tip: run `rail-risk parse-darwin` before `serve`; unparsed closed hours are parsed at startup and add about 8 s
- Docker: `docker compose up --build api ui` (API on 127.0.0.1:8000, page on 127.0.0.1:8501; data/ and mlruns/ mounted read-only; needs Colima or Docker). The capture service is a profile and must not run beside a host capture.
  - Images: `docker build --target api -t rail-risk-api .` and `--target ui`. The API image carries a jlinked Java 17 runtime for Spark; no secret is ever in an image.
- Terraform (plan only, no AWS account): `cd infra && terraform init -backend=false && terraform validate`; offline plan steps are in infra/README.md
- Live eval: `uv run rail-risk live-eval --start D --end D --expect-run-id RUN [--hsp-check-dir DIR --hsp-check-date D]` (point-in-time replay of the frozen champion on captured Darwin data, writes reports/live_eval.json and live_eval.md; fetch an HSP check day with `RAIL_DATA_DIR=DIR uv run rail-risk ingest-hsp ...` into a scratch dir, never into data/, or the training split shifts)
- Stress test: `uv run rail-risk stress-eval [--skip-build]` (frozen champion on the gap-free week of its own routes and on the five hub routes of config/stress.yaml; writes reports/stress_eval.json and .md; fetch new routes first with scripts/run_backfill.sh)
- Drift: `uv run rail-risk drift` (PSI of live Darwin serving features against training; writes reports/drift.json and drift.md; 0.1 watch, 0.25 act)
- Delay Repay: `src/rail_risk/repay.py` reads `config/delay_repay.yaml`; unverified TOCs raise, fares are inputs
- Parse Darwin: `uv run rail-risk parse-darwin [--date YYYY-MM-DD]` (closed hourly files, once each)

## Committing
- One-time: `uv run pre-commit install`
- Check everything: `uv run pre-commit run --all-files`
- Hooks scan staged changes with gitleaks and block `.env` and `data/raw/`. Never bypass with `--no-verify`.

## Data rules
- Never edit `data/raw/` or `.env`. A hook in `.claude/hooks/protect_paths.py` blocks it.
- `data/` is git-ignored except `data/.gitkeep`.
- Secrets live only in `.env`; settings use the `RAIL_` prefix.

## Running the Darwin capture
- Keep the Mac plugged in with the lid open. Sleep pauses it, and it resumes from committed offsets.
- Start: `nohup ./scripts/run_capture.sh > /dev/null 2>&1 &`
- Stop: `pkill -INT -f scripts/capture_darwin.py` (flushes, commits offsets, closes cleanly)
- Check: `tail logs/capture.log` and `ls -l data/raw/darwin/$(date -u +%F)/`
- Anything older than the topic's retention is lost if the capture is down too long.

## Running the HSP backfill
- Start: `nohup ./scripts/run_backfill.sh 2026-04-15 > /dev/null 2>&1 &` (newest first, 30 day chunks)
- Stop: `pkill -f run_backfill.sh; pkill -f "rail-risk ingest-hsp"`. Restart with the same command.
- Order is `ROUTES` in the script: bhm-eus, ast-bhm, bhm-man. Override with `ROUTES="..."`.
- Check: `uv run rail-risk status` and `tail logs/backfill.log`
- Finished windows are skipped on restart, so nothing is refetched. Exit 2 means retry later.
- To extend further back, rerun with an older first date. Roughly 200 s per route-day.

## Stage plan
All stages are done: capture and ingest, Spark features, quantile models with conformal ranges, connection simulator and alerts, API and UI, Docker, Terraform plan-only, drift report, README. Open items are in DECISIONS.md.

## Working rules
- Run and test everything yourself.
- Test small before any large run.
- Say failures plainly with cause and next step.
- Label anything unrun UNTESTED.
- Keep secrets out of git and check before each commit.
- Confirm before destructive or externally visible actions.
- Keep replies short, answer first.
- Write prose with no em dashes and no decorative dividers.
