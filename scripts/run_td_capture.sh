#!/usr/bin/env bash
# Run the TD/TSR capture until stopped, preventing idle sleep, appending logs to logs/td_capture.log.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs
exec caffeinate -dims uv run python scripts/capture_td.py "$@" >> logs/td_capture.log 2>&1
