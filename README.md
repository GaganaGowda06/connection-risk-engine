# Connection Risk Engine

This project will produce, for each UK train on a small set of routes, a calibrated arrival-time range, the probability the train is cancelled, the probability a passenger misses a connection, and the expected Delay Repay in pounds. It is built on open Darwin and Historical Service Performance (HSP) data from Rail Delivery Group. The first routes are Birmingham New Street to London Euston, Birmingham New Street to Manchester Piccadilly, and Aston to London Euston via Birmingham New Street.

## Status

Stage 1 of 7: skeleton, config validation, secret scanning and CI. Nothing is modelled yet, and there are no results to report.

## Setup and tests

```
uv sync
cp .env.example .env
uv run pytest
uv run rail-risk check
```

Install the commit hooks once with `uv run pre-commit install`.
