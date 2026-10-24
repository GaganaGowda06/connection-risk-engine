#!/usr/bin/env bash
# Darwin archive mirror, newest first, one request at a time. Safe to stop and restart:
# fetched hours are in data/interim/archive_manifest.sqlite and are skipped.
#   usage: nohup scripts/run_archive_mirror.sh > /dev/null 2>&1 &
#   stop:  pkill -f run_archive_mirror.sh; pkill -f scripts/mirror_archive.py
set -uo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs
exec >> logs/archive_mirror.log 2>&1
caffeinate -i -w $$ &   # no idle sleep while this script is alive
uv run python scripts/mirror_archive.py "$@"
