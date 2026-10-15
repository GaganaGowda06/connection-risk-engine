#!/usr/bin/env bash
# HSP backfill, newest first, in 30 day chunks, for each route. Safe to stop and restart:
# finished windows are in the manifest and are skipped, and raw responses are write-once.
#   usage: scripts/run_backfill.sh OLDEST [NEWEST]    (dates YYYY-MM-DD; NEWEST defaults to yesterday)
#   env:   ROUTES="bhm-eus bhm-man"
set -uo pipefail
cd "$(dirname "$0")/.."
OLDEST="${1:?usage: run_backfill.sh OLDEST [NEWEST]}"
NEWEST="${2:-$(date -v-1d +%F)}"
ROUTES="${ROUTES:-bhm-eus bhm-man}"
mkdir -p logs
exec >> logs/backfill.log 2>&1
caffeinate -i -w $$ &   # no idle sleep while this script is alive

chunks=$(uv run python - "$OLDEST" "$NEWEST" <<'PY'
import sys
from datetime import date, timedelta
oldest, newest = (date.fromisoformat(a) for a in sys.argv[1:3])
end = newest
while end >= oldest:
    start = max(oldest, end - timedelta(days=29))
    print(start, end)
    end = start - timedelta(days=1)
PY
)
echo "$(date -u +%FT%TZ) backfill start routes=[$ROUTES] oldest=$OLDEST newest=$NEWEST"
for route in $ROUTES; do
  while read -r start end; do
    echo "$(date -u +%FT%TZ) chunk route=$route $start..$end"
    uv run rail-risk ingest-hsp --route "$route" --start "$start" --end "$end" --newest-first
    rc=$?
    if [ "$rc" -eq 1 ]; then
      echo "$(date -u +%FT%TZ) aborting: ingest-hsp exited 1 (config or HTTP error)"
      exit 1
    fi  # rc 2 means some windows failed; they are retried on the next run
  done <<< "$chunks"
done
echo "$(date -u +%FT%TZ) backfill finished"
