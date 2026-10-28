#!/bin/sh
# Entry point of the compose `api` service. Serves the real replay ledger when it exists; if it is
# missing (a fresh clone) it serves the committed sample ledger in data/sample instead, from a
# writable copy, because the server adds indexes on start.
NAME="${REPLAY_NAME:-oot}"
REAL="/app/data/interim/serve/state_${NAME}"
if [ -f "${REAL}/state.sqlite" ]; then
    STATE="${REAL}"
    FLAGS="${SERVE_FLAGS---no-replay}"
else
    echo "real ledger ${REAL} not found: serving the sample ledger (one replay hour)"
    mkdir -p /tmp/sample_state
    cp /app/data/sample/state_oot/state.sqlite /tmp/sample_state/state.sqlite
    STATE=/tmp/sample_state
    FLAGS="--no-replay"
fi
exec python -m rail_risk.serving.run --serve \
    --split "${REPLAY_SPLIT:-oot_week}" \
    --start "${REPLAY_START:-2026-10-17T00:00}" --end "${REPLAY_END:-2026-10-24T04:00}" \
    --state "${STATE}" --store-root "/app/data/interim/serve/store_${NAME}" \
    --die-after-hours "${REPLAY_DIE:-0}" --skip ${REPLAY_SKIP:-} \
    --workers "${SERVE_WORKERS:-4}" ${FLAGS}
