# Stage 7 audit: traceability of the network service (GOAL test 8)

GOAL check: test 8 is the target and is met for the network service responses and the README
result numbers (below). Tests 1 to 5, 6a, 7 and 9 are as in the README. Test 6b is not met.

Scope: the "screen" is the network service (`/health`, `/risk`, `/connection`, `/inbound`) answering
from the committed sample ledger. The legacy Streamlit page and the legacy two-route API are out
of scope.

## Field inventory (sample ledger, one answer per endpoint)

Numeric and time fields, counted by distinct path (`snapshots[]` and `latest` count separately).

| endpoint | fields | source stated in the response, before | after |
|---|---|---|---|
| /health | 5 | 0 | 5 |
| /risk | 15 | 0 | 15 |
| /connection | 15 | 0 | 15 |
| /inbound | 6 | 0 | 6 |
| total | 41 | 0 | 41 |

Before this change the responses named the run ids only. Reading the code traced 37 of the 41
fields to a model output, the message clock or the guard; 4 had no recorded source: `mct_min` and
`slack_min` (in `snapshots[]` and `latest`), because neither the response nor the ledger recorded
which MCT file they came from. Now each response carries `provenance.fields` (field to source and
ledger column) and `provenance.sources` (run ids, MCT file `RJTTF995.ZIP` sequence 995 generated
2026-10-20, message clock, guard file and its threshold). The ledger records the runs that scored
it and the MCT file once per replay (meta key `provenance`).

## Delay Repay

The network service does not price Delay Repay: no fare, price or amount is returned, and no
serving module imports `repay.py` (tested). Rules were therefore not re-verified today. They stay
as recorded in `config/delay_repay.yaml` (checked 2026-10-15, now 15 days old). The legacy API
still prices from `repay.py`, keyed by TOC code, so TOC code LM maps to West Midlands Railway
rules; that is a latent risk for the legacy path and is not changed here.

## README numbers

`scripts/audit_readme_numbers.py`: all checks pass (35 of 35 at the time of writing) (31 numbers found in the file they cite at
the README's precision, plus derived checks for "11 of 13", 25 routes / 11 operators / 5 hubs, and
the 2.5 MB sample). Corrections made while tracing: the file for the 60 minute route-history
persistence difference is `stage4/test_results.json`, not the stage 4b file; 0.44 is in
`stage4b/adaptive_break_it.json`. The check proves a number appears in the cited file at that
precision, not that the sentence describing it is right.

Open finding: "54% to 80% of messages per day" matches `reports/capture_loss/` for 17 to 23
October only. The later Darwin files (not yet committed) show 86% on 25 October and 100% on 26 to
29 October, because the Darwin capture was not running.

## Break tests (`tests/test_traceability.py`)

| failure | result |
|---|---|
| no MCT source (not in the ledger, no `data/ref/mct.csv`) | `/connection` 503 "mct_min has no source"; `/health` degraded with the reason; `/risk` still answers |
| MCT table missing at engine start | `MctError` "no MCT value is assumed" |
| ledger scored by a different P(miss) run | `/risk`, `/connection`, `/inbound` 503 naming `pmiss_model`; `/health` degraded |
| ledger that records no runs (the real replay ledger) | served; `sources.ledger_runs.matches_service` is null with a note |
| stale feed (180 minutes silent) | `freshness.stale` true with "no message for N min"; `/health` degraded |
| Delay Repay rules file older than 30 days | not applicable: the service prices nothing |

## Not tested

- Semantic correctness of each source label (the test checks that an entry exists and names a source key).
- The real replay ledger has no `provenance` stamp; it is checked against `data/ref/mct.csv` instead.
- Docker image from a fresh clone with the new code (run after the commit, see the commit notes).
