#!/usr/bin/env bash
# Run the Darwin capture until stopped, preventing idle sleep, appending logs to logs/capture.log.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs
exec caffeinate -ims uv run python scripts/capture_darwin.py "$@" >> logs/capture.log 2>&1
